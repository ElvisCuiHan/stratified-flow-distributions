#!/usr/bin/env python3
"""Run a lightweight JMLR-style density-estimation benchmark.

This is the first executable benchmark scaffold for the ML submission track.
It compares the true SFD density as an oracle upper-bound against Gaussian and
Gaussian-mixture baselines using held-out log score. The output schema is
designed so stronger estimators can be added without changing downstream
tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tmp" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / "tmp" / "cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "cache"))

import numpy as np
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, multivariate_normal

from flowdistribution import cartoon_face, ecg_like, maze_graph, snowflake, torus_knot


DEFAULT_OUT = ROOT / "outputs" / "jmlr_density_benchmark"
DEFAULT_PROTOCOL_VERSION = "jmlr_frozen_benchmark_v1"

RUN_COLUMNS = [
    "protocol_version",
    "scenario",
    "scenario_family",
    "rep",
    "n_train",
    "n_test",
    "method",
    "heldout_log_score",
    "fit_seconds",
    "eval_seconds",
    "seed",
    "hyperparameter",
    "status",
    "converged",
    "n_iter",
    "error_type",
    "error_message",
    "nonfinite_count",
    "final_objective",
]

SUMMARY_COLUMNS = [
    "protocol_version",
    "scenario",
    "scenario_family",
    "method",
    "mean_heldout_log_score",
    "se_heldout_log_score",
    "rank_within_scenario",
    "mean_fit_seconds",
    "mean_eval_seconds",
    "reps",
    "failure_rate",
    "warning_rate",
]

RUNTIME_COLUMNS = [
    "protocol_version",
    "scenario",
    "scenario_family",
    "method",
    "reps",
    "total_fit_seconds",
    "mean_fit_seconds",
    "total_eval_seconds",
    "mean_eval_seconds",
    "peak_memory_mb",
]


FACTORIES = {
    "snowflake": lambda seed: snowflake(d=3, sigma_shape=0.075, random_state=seed),
    "doraemon": lambda seed: cartoon_face(d=3, sigma_curve=0.070, sigma_blob=0.105, random_state=seed),
    "stylized_face": lambda seed: cartoon_face(d=3, sigma_curve=0.070, sigma_blob=0.105, random_state=seed),
    "maze": lambda seed: maze_graph(d=3, sigma_edge=0.085, sigma_atom=0.135, random_state=seed),
    "ecg": lambda seed: ecg_like(sigma_x=0.018, sigma_y0=0.095, sigma_y1=0.130, random_state=seed),
    "torus": lambda seed: torus_knot(random_state=seed),
}


def parse_int_list(text: str) -> list[int]:
    out = [int(piece) for piece in text.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("All integers must be positive.")
    return out


def parse_csv_list(text: str) -> list[str]:
    out = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return out


def parse_str_list(text: str) -> list[str]:
    out = [piece.strip() for piece in text.split(",") if piece.strip()]
    unknown = sorted(set(out) - set(FACTORIES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown scenarios: {unknown}")
    return out


def regularized_cov(x: np.ndarray, reg: float) -> np.ndarray:
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return cov + reg * np.eye(x.shape[1])


@dataclass
class GaussianFit:
    mean: np.ndarray
    cov: np.ndarray

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)


@dataclass
class GMMFit:
    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    converged: bool = True
    n_iter: int = 0
    final_objective: float | None = None

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for weight, mean, cov in zip(self.weights, self.means, self.covs):
            cols.append(np.log(max(weight, 1e-300)) + multivariate_normal.logpdf(x, mean=mean, cov=cov))
        return logsumexp(np.vstack(cols).T, axis=1)


@dataclass
class KDEFit:
    kde: gaussian_kde

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.kde.logpdf(x.T), dtype=float)


def fit_gaussian(x: np.ndarray, reg: float) -> GaussianFit:
    return GaussianFit(mean=np.mean(x, axis=0), cov=regularized_cov(x, reg))


def fit_kde(x: np.ndarray, *, random_state: int, jitter: float) -> KDEFit:
    try:
        return KDEFit(kde=gaussian_kde(x.T))
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(random_state)
        jittered = x + rng.normal(scale=jitter, size=x.shape)
        return KDEFit(kde=gaussian_kde(jittered.T))


def fit_gmm_em(
    x: np.ndarray,
    n_components: int,
    *,
    random_state: int,
    reg: float,
    max_iter: int,
    tol: float,
) -> GMMFit:
    rng = np.random.default_rng(random_state)
    n, d = x.shape
    k = min(n_components, n)
    init_idx = rng.choice(n, size=k, replace=False)
    means = x[init_idx].copy()
    base_cov = regularized_cov(x, reg)
    covs = np.repeat(base_cov[None, :, :], k, axis=0)
    weights = np.full(k, 1.0 / k)
    previous = -np.inf
    objective = -np.inf
    converged = False
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        log_cols = []
        for weight, mean, cov in zip(weights, means, covs):
            log_cols.append(np.log(max(weight, 1e-300)) + multivariate_normal.logpdf(x, mean=mean, cov=cov))
        log_resp = np.vstack(log_cols).T
        log_norm = logsumexp(log_resp, axis=1)
        objective = float(np.mean(log_norm))
        n_iter = iteration
        resp = np.exp(log_resp - log_norm[:, None])
        nk = np.maximum(resp.sum(axis=0), 1e-12)
        weights = nk / n
        means = (resp.T @ x) / nk[:, None]
        for j in range(k):
            centered = x - means[j]
            covs[j] = (centered * resp[:, [j]]).T @ centered / nk[j] + reg * np.eye(d)
        if np.isfinite(previous) and abs(objective - previous) < tol:
            converged = True
            break
        previous = objective

    final_objective = objective if np.isfinite(objective) else None
    return GMMFit(
        weights=weights,
        means=means,
        covs=covs,
        converged=converged,
        n_iter=n_iter,
        final_objective=final_objective,
    )


def mean_log_score(logpdf: np.ndarray) -> float:
    finite = np.asarray(logpdf, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def effective_scenario_family(args) -> str:
    scenario_family = getattr(args, "scenario_family", None)
    if scenario_family:
        return str(scenario_family)
    return "external_family" if getattr(args, "external_csv", None) else "builtin_density"


def density_row(
    args,
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    method: str,
    seed: int,
    hyperparameter: str,
) -> dict[str, object]:
    return {
        "protocol_version": getattr(args, "protocol_version", DEFAULT_PROTOCOL_VERSION),
        "scenario": scenario,
        "scenario_family": effective_scenario_family(args),
        "rep": rep,
        "n_train": n_train,
        "n_test": n_test,
        "method": method,
        "heldout_log_score": None,
        "fit_seconds": 0.0,
        "eval_seconds": 0.0,
        "seed": seed,
        "hyperparameter": hyperparameter,
        "status": "ok",
        "converged": True,
        "n_iter": 0,
        "error_type": "",
        "error_message": "",
        "nonfinite_count": 0,
        "final_objective": "",
    }


def clean_error_message(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:500]


def mark_warning(row: dict[str, object], error_type: str, error_message: str) -> None:
    if row.get("status") == "failed":
        return
    row["status"] = "warning"
    if row.get("error_type"):
        row["error_type"] = f"{row['error_type']};{error_type}"
    else:
        row["error_type"] = error_type
    if row.get("error_message"):
        row["error_message"] = f"{row['error_message']}; {error_message}"
    else:
        row["error_message"] = error_message


def mark_failure(row: dict[str, object], exc: BaseException) -> None:
    row["status"] = "failed"
    row["converged"] = False
    row["heldout_log_score"] = None
    row["error_type"] = type(exc).__name__
    row["error_message"] = clean_error_message(exc)


def assign_log_score(row: dict[str, object], logpdf: np.ndarray) -> None:
    values = np.asarray(logpdf, dtype=float)
    finite = np.isfinite(values)
    nonfinite_count = int(values.size - int(np.sum(finite)))
    row["nonfinite_count"] = nonfinite_count
    if values.size == 0:
        row["heldout_log_score"] = None
        mark_warning(row, "EmptyLogScore", "No held-out log densities were returned.")
    elif nonfinite_count:
        row["heldout_log_score"] = None
        mark_warning(row, "NonFiniteLogScore", f"{nonfinite_count} non-finite held-out log densities.")
    else:
        row["heldout_log_score"] = float(np.mean(values))


def numeric_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def parse_methods(text: str) -> tuple[bool, list[int], bool, bool]:
    methods = [piece.strip().lower().replace("-", "") for piece in text.split(",") if piece.strip()]
    if not methods:
        raise argparse.ArgumentTypeError("Expected at least one method.")
    gaussian = False
    gmm_components: list[int] = []
    kde = False
    oracle = False
    for method in methods:
        if method == "gaussian":
            gaussian = True
        elif method == "kde":
            kde = True
        elif method in {"oracle", "oraclesfd", "sfd"}:
            oracle = True
        elif method.startswith("gmm") and method[3:].isdigit():
            k = int(method[3:])
            if k <= 0:
                raise argparse.ArgumentTypeError("GMM component counts must be positive.")
            gmm_components.append(k)
        else:
            raise argparse.ArgumentTypeError(f"Unknown method: {method}")
    return gaussian, sorted(set(gmm_components)), kde, oracle


def configured_methods(args, *, include_oracle_default: bool) -> tuple[bool, list[int], bool, bool]:
    if args.methods:
        return parse_methods(args.methods)
    return True, args.gmm_components, args.kde, include_oracle_default


def append_baseline_rows(
    rows: list[dict[str, object]],
    train: np.ndarray,
    test: np.ndarray,
    *,
    scenario: str,
    rep: int,
    seed: int,
    args,
    include_gaussian: bool,
    gmm_components: list[int],
    include_kde: bool,
) -> None:
    n_train = int(train.shape[0])
    n_test = int(test.shape[0])

    if include_gaussian:
        row = density_row(
            args,
            scenario=scenario,
            rep=rep,
            n_train=n_train,
            n_test=n_test,
            method="Gaussian",
            seed=seed,
            hyperparameter=f"reg={args.reg:g}",
        )
        start = time.perf_counter()
        try:
            gaussian = fit_gaussian(train, args.reg)
            row["fit_seconds"] = time.perf_counter() - start
            eval_start = time.perf_counter()
            assign_log_score(row, gaussian.logpdf(test))
            row["eval_seconds"] = time.perf_counter() - eval_start
        except Exception as exc:
            row["fit_seconds"] = time.perf_counter() - start
            mark_failure(row, exc)
        rows.append(row)

    for k in gmm_components:
        row = density_row(
            args,
            scenario=scenario,
            rep=rep,
            n_train=n_train,
            n_test=n_test,
            method=f"GMM-{min(k, n_train)}",
            seed=seed,
            hyperparameter=f"K={k};reg={args.reg:g};max_iter={args.max_iter};tol={args.tol:g}",
        )
        start = time.perf_counter()
        try:
            gmm = fit_gmm_em(
                train,
                k,
                random_state=seed + 17 * k,
                reg=args.reg,
                max_iter=args.max_iter,
                tol=args.tol,
            )
            row["fit_seconds"] = time.perf_counter() - start
            row["converged"] = bool(gmm.converged)
            row["n_iter"] = int(gmm.n_iter)
            if gmm.final_objective is not None:
                row["final_objective"] = float(gmm.final_objective)
            if not gmm.converged:
                mark_warning(
                    row,
                    "ConvergenceWarning",
                    f"EM did not meet tolerance {args.tol:g} within {args.max_iter} iterations.",
                )
            eval_start = time.perf_counter()
            assign_log_score(row, gmm.logpdf(test))
            row["eval_seconds"] = time.perf_counter() - eval_start
        except Exception as exc:
            row["fit_seconds"] = time.perf_counter() - start
            mark_failure(row, exc)
        rows.append(row)

    if include_kde:
        row = density_row(
            args,
            scenario=scenario,
            rep=rep,
            n_train=n_train,
            n_test=n_test,
            method="KDE",
            seed=seed,
            hyperparameter=f"bandwidth=scott;jitter={args.kde_jitter:g}",
        )
        start = time.perf_counter()
        try:
            kde = fit_kde(train, random_state=seed + 7919, jitter=args.kde_jitter)
            row["fit_seconds"] = time.perf_counter() - start
            eval_start = time.perf_counter()
            assign_log_score(row, kde.logpdf(test))
            row["eval_seconds"] = time.perf_counter() - eval_start
        except Exception as exc:
            row["fit_seconds"] = time.perf_counter() - start
            mark_failure(row, exc)
        rows.append(row)


def run_one(args, scenario: str, rep: int, seed: int) -> list[dict[str, object]]:
    dist = FACTORIES[scenario](seed)
    rng = np.random.default_rng(seed + 1009)
    train = np.asarray(dist.rvs(size=args.n_train, random_state=rng), dtype=float)
    test = np.asarray(dist.rvs(size=args.n_test, random_state=rng), dtype=float)
    rows: list[dict[str, object]] = []
    include_gaussian, gmm_components, include_kde, include_oracle = configured_methods(
        args, include_oracle_default=True
    )

    if include_oracle:
        row = density_row(
            args,
            scenario=scenario,
            rep=rep,
            n_train=args.n_train,
            n_test=args.n_test,
            method="Oracle SFD",
            seed=seed,
            hyperparameter=f"n_grid={args.n_grid}",
        )
        start = time.perf_counter()
        try:
            oracle_logpdf = np.asarray(dist.logpdf(test, n_grid=args.n_grid), dtype=float)
            assign_log_score(row, oracle_logpdf)
            row["eval_seconds"] = time.perf_counter() - start
        except Exception as exc:
            row["eval_seconds"] = time.perf_counter() - start
            mark_failure(row, exc)
        rows.append(row)

    append_baseline_rows(
        rows,
        train,
        test,
        scenario=scenario,
        rep=rep,
        seed=seed,
        args=args,
        include_gaussian=include_gaussian,
        gmm_components=gmm_components,
        include_kde=include_kde,
    )

    return rows


def read_split_assignments(path: Path) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    assignments: dict[tuple[str, str], str] = {}
    metadata: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"scenario", "sample_id", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required split columns: {sorted(missing)}")
        for row in reader:
            key = (row["scenario"], row["sample_id"])
            assignments[key] = row["split"].strip().lower()
            for name in ("split_seed", "train_fraction", "split_rule"):
                if row.get(name) and name not in metadata:
                    metadata[name] = row[name]
    return assignments, metadata


def load_external_splits(args) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    if args.split_assignments is None:
        raise ValueError("--split-assignments is required with --external-csv.")
    features = args.features or []
    if not features:
        raise ValueError("--features is required with --external-csv.")
    assignments, split_metadata = read_split_assignments(args.split_assignments)
    scenario_filter = set(args.scenarios)
    by_scenario: dict[str, dict[str, list[list[float]]]] = {
        scenario: {"train": [], "test": []} for scenario in args.scenarios
    }

    with args.external_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"scenario", "sample_id", *features}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{args.external_csv} is missing required columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            scenario = row["scenario"]
            if scenario not in scenario_filter:
                continue
            sample_id = row["sample_id"]
            split = assignments.get((scenario, sample_id))
            if split not in {"train", "test"}:
                continue
            try:
                values = [float(row[name]) for name in features]
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric feature value in {args.external_csv} line {line_number}: {features}"
                ) from exc
            by_scenario[scenario][split].append(values)

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for scenario, splits in by_scenario.items():
        train = np.asarray(splits["train"], dtype=float)
        test = np.asarray(splits["test"], dtype=float)
        if train.size == 0 or test.size == 0:
            raise ValueError(f"Scenario {scenario!r} has train={train.shape[0]} test={test.shape[0]} rows.")
        arrays[scenario] = {"train": train, "test": test}
    return arrays, split_metadata


def run_external_one(args, scenario: str, rep: int, seed: int, train: np.ndarray, test: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    include_gaussian, gmm_components, include_kde, include_oracle = configured_methods(
        args, include_oracle_default=False
    )
    if include_oracle:
        raise ValueError("Oracle SFD is only available for built-in synthetic scenarios.")
    append_baseline_rows(
        rows,
        train,
        test,
        scenario=scenario,
        rep=rep,
        seed=seed,
        args=args,
        include_gaussian=include_gaussian,
        gmm_components=gmm_components,
        include_kde=include_kde,
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(r["scenario"], r["method"]) for r in rows})
    out = []
    for scenario, method in keys:
        sub = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
        scores = np.asarray(
            [score for score in (numeric_or_none(r.get("heldout_log_score")) for r in sub) if score is not None],
            dtype=float,
        )
        fit = np.asarray(
            [value for value in (numeric_or_none(r.get("fit_seconds")) for r in sub) if value is not None],
            dtype=float,
        )
        eval_time = np.asarray(
            [value for value in (numeric_or_none(r.get("eval_seconds")) for r in sub) if value is not None],
            dtype=float,
        )
        status = [str(r.get("status", "ok")) for r in sub]
        reps = len(sub)
        failures = sum(1 for item in status if item == "failed")
        warnings = sum(1 for item in status if item == "warning")
        out.append(
            {
                "protocol_version": sub[0].get("protocol_version", DEFAULT_PROTOCOL_VERSION),
                "scenario": scenario,
                "scenario_family": sub[0].get("scenario_family", ""),
                "method": method,
                "mean_heldout_log_score": float(np.mean(scores)) if scores.size else None,
                "se_heldout_log_score": float(np.std(scores, ddof=1) / np.sqrt(scores.size)) if scores.size > 1 else 0.0 if scores.size == 1 else None,
                "rank_within_scenario": 0,
                "mean_fit_seconds": float(np.mean(fit)) if fit.size else None,
                "mean_eval_seconds": float(np.mean(eval_time)) if eval_time.size else None,
                "reps": int(reps),
                "failure_rate": float(failures / reps) if reps else 0.0,
                "warning_rate": float(warnings / reps) if reps else 0.0,
            }
        )
    for scenario in sorted({str(r["scenario"]) for r in out}):
        scenario_rows = [r for r in out if r["scenario"] == scenario]
        scenario_rows = [r for r in scenario_rows if numeric_or_none(r.get("mean_heldout_log_score")) is not None]
        scenario_rows.sort(key=lambda r: float(r["mean_heldout_log_score"]), reverse=True)
        for rank, row in enumerate(scenario_rows, start=1):
            row["rank_within_scenario"] = rank
    return out


def summarize_runtime(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in summary_rows:
        reps = int(row["reps"])
        mean_fit = numeric_or_none(row.get("mean_fit_seconds"))
        mean_eval = numeric_or_none(row.get("mean_eval_seconds"))
        out.append(
            {
                "protocol_version": row.get("protocol_version", DEFAULT_PROTOCOL_VERSION),
                "scenario": row["scenario"],
                "scenario_family": row.get("scenario_family", ""),
                "method": row["method"],
                "reps": reps,
                "total_fit_seconds": mean_fit * reps if mean_fit is not None else None,
                "mean_fit_seconds": mean_fit,
                "total_eval_seconds": mean_eval * reps if mean_eval is not None else None,
                "mean_eval_seconds": mean_eval,
                "peak_memory_mb": None,
            }
        )
    return out


def shell_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(piece) for piece in argv)


def configured_method_names(args, *, include_oracle_default: bool) -> list[str]:
    include_gaussian, gmm_components, include_kde, include_oracle = configured_methods(
        args, include_oracle_default=include_oracle_default
    )
    names: list[str] = []
    if include_oracle:
        names.append("Oracle SFD")
    if include_gaussian:
        names.append("Gaussian")
    if include_kde:
        names.append("KDE")
    names.extend(f"GMM-{k}" for k in gmm_components)
    return names


def status_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "ok"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def parameters_payload(args, command: str, started_at: str | None = None, finished_at: str | None = None) -> dict[str, object]:
    external = args.external_csv is not None
    payload: dict[str, object] = {
        "protocol_version": args.protocol_version,
        "scenario_family": effective_scenario_family(args),
        "mode": "external_csv" if external else "synthetic",
        "scenarios": list(args.scenarios),
        "reps": int(args.reps),
        "seed": int(args.seed),
        "seed_formula": "seed + 10000 * rep + 101 * scenario_index, with rep starting at 1 and scenario_index starting at 0",
        "scenario_order": list(args.scenarios),
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "gmm_components": list(args.gmm_components),
        "kde": bool(args.kde),
        "methods": args.methods or "default",
        "method_names": configured_method_names(args, include_oracle_default=not external),
        "external_csv": str(args.external_csv) if args.external_csv else "",
        "split_assignments": str(args.split_assignments) if args.split_assignments else "",
        "features": list(args.features or []),
        "truth_columns": list(args.truth_columns or []),
        "kde_jitter": float(args.kde_jitter),
        "n_grid": int(args.n_grid),
        "reg": float(args.reg),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "output_dir": str(args.output_dir),
        "figure_dir": str(args.figure_dir),
        "figure_prefix": args.figure_prefix,
        "command": command,
        "failure_rules": {
            "method_failure": "write status=failed with error_type and error_message",
            "convergence_warning": "write status=warning with converged, n_iter, and final_objective when available",
            "nonfinite_log_score": "write heldout_log_score as missing, keep the row, and record nonfinite_count",
        },
        "output_files": {
            "heldout_log_scores": str(args.output_dir / "heldout_log_scores.csv"),
            "benchmark_summary": str(args.output_dir / "benchmark_summary.csv"),
            "runtime": str(args.output_dir / "runtime.csv"),
            "run_parameters_txt": str(args.output_dir / "run_parameters.txt"),
            "run_parameters_json": str(args.output_dir / "run_parameters.json"),
            "execution_report": str(args.output_dir / "execution_report.md"),
        },
    }
    if started_at:
        payload["started_at_utc"] = started_at
    if finished_at:
        payload["finished_at_utc"] = finished_at
    if external:
        payload["n_train_argument"] = int(args.n_train)
        payload["n_test_argument"] = int(args.n_test)
        payload["external_counts"] = getattr(args, "external_counts", {})
        payload["split_metadata"] = getattr(args, "split_metadata", {})
    return payload


def write_parameters(path: Path, args) -> None:
    output_dir = args.output_dir
    command = getattr(args, "command", "")
    lines = [
        "JMLR density benchmark scaffold",
        f"protocol_version={args.protocol_version}",
        f"scenario_family={effective_scenario_family(args)}",
        f"mode={'external_csv' if args.external_csv else 'synthetic'}",
        f"scenarios={','.join(args.scenarios)}",
        f"reps={args.reps}",
        f"seed={args.seed}",
        "seed_formula=seed + 10000 * rep + 101 * scenario_index, with rep starting at 1 and scenario_index starting at 0",
        f"gmm_components={','.join(map(str, args.gmm_components))}",
        f"kde={args.kde}",
        f"methods={args.methods or 'default'}",
        f"method_names={','.join(configured_method_names(args, include_oracle_default=args.external_csv is None))}",
        f"external_csv={args.external_csv or ''}",
        f"split_assignments={args.split_assignments or ''}",
        f"features={','.join(args.features or [])}",
        f"truth_columns={','.join(args.truth_columns or [])}",
        f"kde_jitter={args.kde_jitter}",
        f"n_grid={args.n_grid}",
        f"reg={args.reg}",
        f"max_iter={args.max_iter}",
        f"tol={args.tol}",
        f"output_dir={output_dir}",
        f"figure_dir={args.figure_dir}",
        f"figure_prefix={args.figure_prefix}",
        f"command={command}",
        "failure_rule.method_failure=status=failed with error_type and error_message",
        "failure_rule.convergence_warning=status=warning with converged, n_iter, and final_objective when available",
        "failure_rule.nonfinite_log_score=heldout_log_score missing with nonfinite_count recorded",
    ]
    if args.external_csv:
        lines.extend([f"n_train_argument={args.n_train}", f"n_test_argument={args.n_test}"])
    else:
        lines.extend([f"n_train={args.n_train}", f"n_test={args.n_test}"])
    for scenario, counts in sorted(getattr(args, "external_counts", {}).items()):
        lines.append(f"external_counts.{scenario}.n_train={counts['n_train']}")
        lines.append(f"external_counts.{scenario}.n_test={counts['n_test']}")
    for key, value in sorted(getattr(args, "split_metadata", {}).items()):
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


def write_parameters_json(path: Path, args) -> None:
    payload = parameters_payload(
        args,
        getattr(args, "command", ""),
        getattr(args, "started_at_utc", None),
        getattr(args, "finished_at_utc", None),
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_execution_report(
    path: Path,
    args,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
) -> None:
    counts = status_counts(rows)
    n_rows = len(rows)
    full_profile = (
        args.external_csv is None
        and int(args.n_train) == 1000
        and int(args.n_test) == 1000
        and int(args.reps) == 25
    )
    run_profile = "frozen paper-facing 25-rep target" if full_profile else "standard or diagnostic run, not the final 25-rep target"
    status_text = ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) if counts else "none"
    lines = [
        "# JMLR Frozen Benchmark Execution Report",
        "",
        f"- protocol_version: `{args.protocol_version}`",
        f"- scenario_family: `{effective_scenario_family(args)}`",
        f"- run_profile: {run_profile}",
        f"- started_at_utc: `{getattr(args, 'started_at_utc', '')}`",
        f"- finished_at_utc: `{getattr(args, 'finished_at_utc', '')}`",
        f"- command: `{getattr(args, 'command', '')}`",
        f"- seed_formula: `seed + 10000 * rep + 101 * scenario_index`",
        f"- scenarios: `{','.join(args.scenarios)}`",
        f"- methods: `{','.join(configured_method_names(args, include_oracle_default=args.external_csv is None))}`",
        f"- n_train: `{args.n_train}`",
        f"- n_test: `{args.n_test}`",
        f"- reps: `{args.reps}`",
        f"- status_counts: {status_text}",
        "",
        "## Output Files",
        "",
        f"- `heldout_log_scores.csv`: {n_rows} rows",
        f"- `benchmark_summary.csv`: {len(summary_rows)} rows",
        f"- `runtime.csv`: {len(runtime_rows)} rows",
        f"- `run_parameters.txt`",
        f"- `run_parameters.json`",
        "",
        "## Schema Status",
        "",
        f"- Per-run columns: `{','.join(RUN_COLUMNS)}`",
        f"- Summary columns: `{','.join(SUMMARY_COLUMNS)}`",
        f"- Runtime columns: `{','.join(RUNTIME_COLUMNS)}`",
        "",
        "## Claim Boundary",
        "",
        "- Oracle SFD rows are diagnostic upper-bound rows, not fitted-estimator performance evidence.",
        "- This report does not authorize manuscript performance-claim edits by itself.",
    ]
    path.write_text("\n".join(lines) + "\n")


def maybe_plot(out_dir: Path, figure_dir: Path, figure_prefix: str, summary_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
        from sfd_figure_style import panel_title, set_sfd_matplotlib_style
    except Exception:
        return

    set_sfd_matplotlib_style(plt, base_size=9.0, title_size=10.0, label_size=9.0, tick_size=8.0)
    scenario_order = ["snowflake", "stylized_face", "doraemon", "maze", "ecg", "torus"]
    scenario_order.extend(
        sorted({str(r["scenario"]) for r in summary_rows if str(r["scenario"]) not in scenario_order})
    )
    scenario_labels = {
        "snowflake": "Snowflake",
        "stylized_face": "Stylized face",
        "doraemon": "Stylized face",
        "maze": "Maze graph",
        "ecg": "ECG-like flow",
        "torus": "Torus knot",
    }
    method_order = ["Gaussian", "KDE", "GMM-2", "GMM-4", "GMM-8", "GMM-16", "GMM-32", "Oracle SFD"]
    scenarios = [s for s in scenario_order if any(str(r["scenario"]) == s for r in summary_rows)]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.0 * len(scenarios), 3.65), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]
    color = {
        "Gaussian": "#9fd8cb",
        "KDE": "#d7b56d",
        "GMM-2": "#91c4df",
        "GMM-4": "#b7a4df",
        "GMM-8": "#f7c979",
        "GMM-16": "#c7d77f",
        "GMM-32": "#d4a6a3",
        "Oracle SFD": "#f28e88",
    }
    for ax, scenario in zip(axes, scenarios):
        sub = [
            r for r in summary_rows
            if r["scenario"] == scenario and numeric_or_none(r.get("mean_heldout_log_score")) is not None
        ]
        sub.sort(key=lambda r: method_order.index(str(r["method"])) if str(r["method"]) in method_order else 999)
        x = np.arange(len(sub))
        y = [float(r["mean_heldout_log_score"]) for r in sub]
        yerr = [numeric_or_none(r.get("se_heldout_log_score")) or 0.0 for r in sub]
        ax.bar(
            x,
            y,
            yerr=yerr,
            color=[color.get(str(r["method"]), "#b8c0cc") for r in sub],
            edgecolor="white",
            linewidth=0.8,
            error_kw={"elinewidth": 0.8, "ecolor": "#324150", "capsize": 2.5},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["method"]) for r in sub], rotation=35, ha="right")
        panel_title(ax, scenario_labels.get(scenario, scenario.replace("_", " ").title()))
        ax.set_ylabel("held-out log score")
        ax.axhline(0.0, color="#53606b", linewidth=0.55, alpha=0.45)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        out_dir / f"{figure_prefix}.png",
        out_dir / f"{figure_prefix}.pdf",
        figure_dir / f"{figure_prefix}.png",
        figure_dir / f"{figure_prefix}.pdf",
    ]:
        fig.savefig(path, dpi=220 if path.suffix == ".png" else None)
    fig.savefig(out_dir / "fig_heldout_log_score_scaffold.png", dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output-dir", dest="output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol-version", type=str, default=DEFAULT_PROTOCOL_VERSION)
    parser.add_argument("--scenario-family", type=str)
    parser.add_argument("--external-csv", type=Path)
    parser.add_argument("--split-assignments", type=Path)
    parser.add_argument("--features", type=parse_csv_list)
    parser.add_argument("--truth-columns", type=parse_csv_list, default=[])
    parser.add_argument("--methods", type=str)
    parser.add_argument("--scenarios", type=parse_csv_list, default=parse_csv_list("snowflake,doraemon,maze,ecg,torus"))
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=250)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--gmm-components", type=parse_int_list, default=parse_int_list("2,4,8,16,32"))
    parser.add_argument("--kde", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kde-jitter", type=float, default=1e-6)
    parser.add_argument("--n-grid", type=int, default=96)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--figure-prefix", type=str, default="jmlr_density_benchmark_heldout_log_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_train <= 0 or args.n_test <= 0:
        raise ValueError("--n-train and --n-test must be positive.")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive.")
    if args.tol <= 0:
        raise ValueError("--tol must be positive.")
    args.command = shell_command([sys.executable, *sys.argv])
    args.started_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if args.external_csv is None:
        unknown = sorted(set(args.scenarios) - set(FACTORIES))
        if unknown:
            raise ValueError(f"Unknown synthetic scenarios: {unknown}")
    args.reps = args.reps if args.reps is not None else (1 if args.external_csv else 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    if args.external_csv:
        external_splits, split_metadata = load_external_splits(args)
        args.split_metadata = split_metadata
        args.external_counts = {
            scenario: {
                "n_train": int(splits["train"].shape[0]),
                "n_test": int(splits["test"].shape[0]),
            }
            for scenario, splits in external_splits.items()
        }
        for rep in range(1, args.reps + 1):
            for idx, scenario in enumerate(args.scenarios):
                seed = args.seed + 10000 * rep + 101 * idx
                train = external_splits[scenario]["train"]
                test = external_splits[scenario]["test"]
                print(
                    f"[benchmark] external scenario={scenario} rep={rep} seed={seed} "
                    f"n_train={train.shape[0]} n_test={test.shape[0]}",
                    flush=True,
                )
                rows.extend(run_external_one(args, scenario, rep, seed, train, test))
    else:
        args.split_metadata = {}
        args.external_counts = {}
        for rep in range(1, args.reps + 1):
            for idx, scenario in enumerate(args.scenarios):
                seed = args.seed + 10000 * rep + 101 * idx
                print(f"[benchmark] scenario={scenario} rep={rep} seed={seed}", flush=True)
                rows.extend(run_one(args, scenario, rep, seed))

    summary_rows = summarize(rows)
    runtime_rows = summarize_runtime(summary_rows)
    args.finished_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    write_csv(args.output_dir / "heldout_log_scores.csv", rows, RUN_COLUMNS)
    write_csv(args.output_dir / "benchmark_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(args.output_dir / "runtime.csv", runtime_rows, RUNTIME_COLUMNS)
    write_parameters(args.output_dir / "run_parameters.txt", args)
    write_parameters_json(args.output_dir / "run_parameters.json", args)
    write_execution_report(args.output_dir / "execution_report.md", args, rows, summary_rows, runtime_rows)
    maybe_plot(args.output_dir, args.figure_dir, args.figure_prefix, summary_rows)
    print(f"Wrote {args.output_dir / 'heldout_log_scores.csv'}")
    print(f"Wrote {args.output_dir / 'benchmark_summary.csv'}")
    print(f"Wrote {args.output_dir / 'runtime.csv'}")
    print(f"Wrote {args.output_dir / 'run_parameters.txt'}")
    print(f"Wrote {args.output_dir / 'run_parameters.json'}")
    print(f"Wrote {args.output_dir / 'execution_report.md'}")


if __name__ == "__main__":
    main()
