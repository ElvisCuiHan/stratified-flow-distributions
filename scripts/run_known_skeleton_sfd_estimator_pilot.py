#!/usr/bin/env python3
"""Run a known-skeleton weight-only fitted-SFD diagnostic.

This script fits only component weights for a fixed true SFD skeleton. The
built-in scenario factory defines the geometry, component families, and truth
labels. The output bundle is shaped toward the frozen JMLR benchmark schema,
but the estimator remains known-skeleton/weight-only and does not learn
topology or component geometry.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tmp" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / "tmp" / "cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "cache"))

import numpy as np
from scipy.special import logsumexp

from run_jmlr_density_benchmark import (
    FACTORIES,
    fit_gaussian,
    fit_gmm_em,
    fit_kde,
    mean_log_score,
    parse_int_list,
    parse_str_list,
)


PROTOCOL_VERSION = "jmlr_frozen_benchmark_v1"
SCENARIO_FAMILY = "built_in_mixed_support"
DEFAULT_OUT = ROOT / "outputs" / PROTOCOL_VERSION / "fitted_sfd_standard"
METHOD = "Known-skeleton weight-only SFD"
ESTIMATOR_SCOPE = (
    "known-skeleton/weight-only; component geometry and topology fixed to the "
    "true data-generating SFD"
)
SEED_FORMULA = "seed = base_seed + 10000 * rep + 101 * scenario_index_zero_based"


@dataclass
class WeightOnlySFD:
    weights: np.ndarray
    iterations: int
    converged: bool
    final_objective: float
    final_max_weight_change: float


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_array(values: list[object]) -> np.ndarray:
    arr = np.asarray([as_float(value) for value in values], dtype=float)
    return arr[np.isfinite(arr)]


def se(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def component_logdensities(dist, x: np.ndarray, n_grid: int) -> np.ndarray:
    cols = [comp.logpdf(x, n_grid=n_grid) for comp in dist.components]
    return np.vstack(cols).T


def fit_weight_only_sfd(
    log_density: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    min_weight: float,
) -> WeightOnlySFD:
    n_components = log_density.shape[1]
    weights = np.full(n_components, 1.0 / n_components)
    converged = False
    iterations = 0
    final_objective = float("nan")
    final_max_weight_change = float("inf")
    for iterations in range(1, max_iter + 1):
        log_resp = log_density + np.log(np.maximum(weights, min_weight))
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - log_norm)
        updated = np.maximum(resp.mean(axis=0), min_weight)
        updated = updated / updated.sum()
        final_objective = float(np.mean(log_norm))
        final_max_weight_change = float(np.max(np.abs(updated - weights)))
        if final_max_weight_change < tol:
            converged = True
            weights = updated
            break
        weights = updated
    return WeightOnlySFD(
        weights=weights,
        iterations=iterations,
        converged=converged,
        final_objective=final_objective,
        final_max_weight_change=final_max_weight_change,
    )


def logpdf_from_component_density(log_density: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return logsumexp(log_density + np.log(np.maximum(weights, 1e-300)), axis=1)


def responsibilities_from_component_density(log_density: np.ndarray, weights: np.ndarray) -> np.ndarray:
    log_resp = log_density + np.log(np.maximum(weights, 1e-300))
    log_norm = logsumexp(log_resp, axis=1, keepdims=True)
    return np.exp(log_resp - log_norm)


def score_and_nonfinite(logpdf: np.ndarray) -> tuple[float, int]:
    values = np.asarray(logpdf, dtype=float)
    return mean_log_score(values), int(np.size(values) - np.isfinite(values).sum())


def density_row(
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    method: str,
    heldout_log_score: float,
    fit_seconds: float,
    eval_seconds: float,
    seed: int,
    hyperparameter: str,
    status: str = "ok",
    converged: object = "",
    n_iter: object = "",
    error_type: str = "",
    error_message: str = "",
    nonfinite_count: int = 0,
    final_objective: object = "",
    final_max_weight_change: object = "",
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": scenario,
        "scenario_family": SCENARIO_FAMILY,
        "rep": rep,
        "n_train": n_train,
        "n_test": n_test,
        "method": method,
        "heldout_log_score": heldout_log_score,
        "fit_seconds": fit_seconds,
        "eval_seconds": eval_seconds,
        "seed": seed,
        "hyperparameter": hyperparameter,
        "status": status,
        "converged": converged,
        "n_iter": n_iter,
        "error_type": error_type,
        "error_message": error_message,
        "nonfinite_count": nonfinite_count,
        "final_objective": final_objective,
        "final_max_weight_change": final_max_weight_change,
        "estimator_scope": ESTIMATOR_SCOPE if method == METHOD else "",
    }


def failed_density_row(
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    method: str,
    seed: int,
    hyperparameter: str,
    exc: BaseException,
    fit_seconds: float = 0.0,
    eval_seconds: float = 0.0,
) -> dict[str, object]:
    return density_row(
        scenario=scenario,
        rep=rep,
        n_train=n_train,
        n_test=n_test,
        method=method,
        heldout_log_score=float("nan"),
        fit_seconds=fit_seconds,
        eval_seconds=eval_seconds,
        seed=seed,
        hyperparameter=hyperparameter,
        status="failed",
        converged=False,
        n_iter="",
        error_type=type(exc).__name__,
        error_message=str(exc)[:500],
    )


def component_masks(dist) -> tuple[np.ndarray, np.ndarray]:
    kinds = np.asarray([comp.kind for comp in dist.components], dtype=object)
    atom_mask = kinds == "gaussian"
    branch_mask = kinds == "tube"
    return atom_mask, branch_mask


def layer_name(comp) -> str:
    return str(getattr(comp, "metadata", {}).get("layer", ""))


def recovery_metrics(
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    seed: int,
    dist,
    fitted_weights: np.ndarray,
    resp_test: np.ndarray,
    test_labels: np.ndarray,
    fit: WeightOnlySFD,
) -> dict[str, object]:
    true_weights = np.asarray(dist.weights, dtype=float)
    diff = fitted_weights - true_weights
    one_hot = np.eye(len(true_weights))[test_labels]
    pred = np.argmax(resp_test, axis=1)
    atom_mask, branch_mask = component_masks(dist)
    atom_diff = diff[atom_mask]
    branch_diff = diff[branch_mask]
    branch_rows = branch_mask[test_labels]
    status = "ok" if fit.converged else "warning"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": scenario,
        "rep": rep,
        "method": METHOD,
        "branch_accuracy": float(np.mean(pred[branch_rows] == test_labels[branch_rows])) if np.any(branch_rows) else float("nan"),
        "stratum_accuracy": float(np.mean(pred == test_labels)),
        "atom_mass_bias": float(np.sum(fitted_weights[atom_mask]) - np.sum(true_weights[atom_mask])),
        "atom_mass_rmse": float(np.sqrt(np.mean(atom_diff**2))) if atom_diff.size else 0.0,
        "branch_mass_bias": float(np.sum(fitted_weights[branch_mask]) - np.sum(true_weights[branch_mask])),
        "branch_mass_rmse": float(np.sqrt(np.mean(branch_diff**2))) if branch_diff.size else 0.0,
        "latent_coordinate_rmse": float("nan"),
        "offtrajectory_variance_bias": float("nan"),
        "offtrajectory_variance_rmse": float("nan"),
        "status": status,
        "n_train": n_train,
        "n_test": n_test,
        "seed": seed,
        "n_components": int(len(true_weights)),
        "n_atom_like_components": int(np.sum(atom_mask)),
        "n_branch_components": int(np.sum(branch_mask)),
        "component_mass_rmse": float(np.sqrt(np.mean(diff**2))),
        "component_mass_mae": float(np.mean(np.abs(diff))),
        "component_mass_max_abs_bias": float(np.max(np.abs(diff))),
        "responsibility_brier": float(np.mean(np.sum((resp_test - one_hot) ** 2, axis=1))),
        "mean_true_component_responsibility": float(np.mean(resp_test[np.arange(test_labels.size), test_labels])),
        "em_iterations": fit.iterations,
        "em_converged": fit.converged,
        "final_objective": fit.final_objective,
        "final_max_weight_change": fit.final_max_weight_change,
        "truth_labels_available": True,
        "truth_alignment": "SFD.rvs(return_info=True).component_index aligned to fixed true component order.",
        "estimator_scope": ESTIMATOR_SCOPE,
        "note": "Known-skeleton diagnostic only; latent coordinate and off-trajectory variance are not estimated.",
    }


def failed_recovery_row(
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    seed: int,
    exc: BaseException,
) -> dict[str, object]:
    row = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": scenario,
        "rep": rep,
        "method": METHOD,
        "branch_accuracy": float("nan"),
        "stratum_accuracy": float("nan"),
        "atom_mass_bias": float("nan"),
        "atom_mass_rmse": float("nan"),
        "branch_mass_bias": float("nan"),
        "branch_mass_rmse": float("nan"),
        "latent_coordinate_rmse": float("nan"),
        "offtrajectory_variance_bias": float("nan"),
        "offtrajectory_variance_rmse": float("nan"),
        "status": "failed",
        "n_train": n_train,
        "n_test": n_test,
        "seed": seed,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:500],
        "estimator_scope": ESTIMATOR_SCOPE,
    }
    return row


def fitted_weight_rows(
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    seed: int,
    dist,
    fit: WeightOnlySFD,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    true_weights = np.asarray(dist.weights, dtype=float)
    train_counts = np.bincount(train_labels, minlength=len(true_weights))
    test_counts = np.bincount(test_labels, minlength=len(true_weights))
    for idx, (comp, true_weight, fitted_weight) in enumerate(zip(dist.components, true_weights, fit.weights)):
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": scenario,
                "scenario_family": SCENARIO_FAMILY,
                "rep": rep,
                "method": METHOD,
                "seed": seed,
                "n_train": n_train,
                "n_test": n_test,
                "component_index": idx,
                "component_name": comp.name,
                "component_kind": comp.kind,
                "component_layer": layer_name(comp),
                "true_weight": float(true_weight),
                "fitted_weight": float(fitted_weight),
                "weight_bias": float(fitted_weight - true_weight),
                "abs_weight_bias": float(abs(fitted_weight - true_weight)),
                "observed_train_fraction": float(train_counts[idx] / n_train),
                "observed_test_fraction": float(test_counts[idx] / n_test),
                "em_iterations": fit.iterations,
                "em_converged": fit.converged,
                "estimator_scope": ESTIMATOR_SCOPE,
            }
        )
    return rows


def responsibility_summary_rows(
    *,
    scenario: str,
    rep: int,
    n_test: int,
    seed: int,
    dist,
    resp_test: np.ndarray,
    test_labels: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    pred = np.argmax(resp_test, axis=1)
    max_resp = np.max(resp_test, axis=1)
    pred_counts = np.bincount(pred, minlength=len(dist.components))
    for idx, comp in enumerate(dist.components):
        mask = test_labels == idx
        if np.any(mask):
            true_resp = float(np.mean(resp_test[mask, idx]))
            mean_max = float(np.mean(max_resp[mask]))
            pred_as_true = float(np.mean(pred[mask] == idx))
            n_test_component = int(np.sum(mask))
        else:
            true_resp = float("nan")
            mean_max = float("nan")
            pred_as_true = float("nan")
            n_test_component = 0
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": scenario,
                "scenario_family": SCENARIO_FAMILY,
                "rep": rep,
                "method": METHOD,
                "seed": seed,
                "n_test": n_test,
                "true_component_index": idx,
                "true_component_name": comp.name,
                "true_component_kind": comp.kind,
                "true_component_layer": layer_name(comp),
                "n_test_component": n_test_component,
                "true_test_fraction": float(n_test_component / n_test),
                "predicted_test_fraction": float(pred_counts[idx] / n_test),
                "mean_responsibility_for_true_component": true_resp,
                "mean_max_responsibility": mean_max,
                "predicted_as_true_fraction": pred_as_true,
                "estimator_scope": ESTIMATOR_SCOPE,
            }
        )
    return rows


def convergence_row(
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    seed: int,
    dist,
    fit: WeightOnlySFD,
    args,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": scenario,
        "scenario_family": SCENARIO_FAMILY,
        "rep": rep,
        "method": METHOD,
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "status": "ok" if fit.converged else "warning",
        "converged": fit.converged,
        "n_iter": fit.iterations,
        "final_objective": fit.final_objective,
        "final_max_weight_change": fit.final_max_weight_change,
        "em_tol": args.em_tol,
        "em_max_iter": args.em_max_iter,
        "min_weight": args.min_weight,
        "n_components": len(dist.components),
        "estimator_scope": ESTIMATOR_SCOPE,
    }


def failed_convergence_row(
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    seed: int,
    exc: BaseException,
    args,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": scenario,
        "scenario_family": SCENARIO_FAMILY,
        "rep": rep,
        "method": METHOD,
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "status": "failed",
        "converged": False,
        "n_iter": "",
        "final_objective": "",
        "final_max_weight_change": "",
        "em_tol": args.em_tol,
        "em_max_iter": args.em_max_iter,
        "min_weight": args.min_weight,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:500],
        "estimator_scope": ESTIMATOR_SCOPE,
    }


def run_one(
    args,
    scenario: str,
    rep: int,
    seed: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    dist = FACTORIES[scenario](seed)
    rng = np.random.default_rng(seed + 1009)
    train, train_info = dist.rvs(size=args.n_train, random_state=rng, return_info=True)
    test, test_info = dist.rvs(size=args.n_test, random_state=rng, return_info=True)
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    train_labels = np.asarray(train_info["component_index"], dtype=int)
    test_labels = np.asarray(test_info["component_index"], dtype=int)
    rows: list[dict[str, object]] = []

    start = time.perf_counter()
    try:
        oracle_logpdf = np.asarray(dist.logpdf(test, n_grid=args.n_grid), dtype=float)
        eval_seconds = time.perf_counter() - start
        score, nonfinite = score_and_nonfinite(oracle_logpdf)
        rows.append(
            density_row(
                scenario=scenario,
                rep=rep,
                n_train=args.n_train,
                n_test=args.n_test,
                method="Oracle SFD",
                heldout_log_score=score,
                fit_seconds=0.0,
                eval_seconds=eval_seconds,
                seed=seed,
                hyperparameter=f"n_grid={args.n_grid}",
                converged=True,
                n_iter=0,
                nonfinite_count=nonfinite,
            )
        )
    except Exception as exc:  # pragma: no cover - exercised by integration failures
        rows.append(
            failed_density_row(
                scenario=scenario,
                rep=rep,
                n_train=args.n_train,
                n_test=args.n_test,
                method="Oracle SFD",
                seed=seed,
                hyperparameter=f"n_grid={args.n_grid}",
                exc=exc,
                eval_seconds=time.perf_counter() - start,
            )
        )

    try:
        start = time.perf_counter()
        train_log_density = component_logdensities(dist, train, args.n_grid)
        fit = fit_weight_only_sfd(
            train_log_density,
            max_iter=args.em_max_iter,
            tol=args.em_tol,
            min_weight=args.min_weight,
        )
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        test_log_density = component_logdensities(dist, test, args.n_grid)
        fixed_logpdf = logpdf_from_component_density(test_log_density, fit.weights)
        resp_test = responsibilities_from_component_density(test_log_density, fit.weights)
        eval_seconds = time.perf_counter() - start
        score, nonfinite = score_and_nonfinite(fixed_logpdf)
        rows.append(
            density_row(
                scenario=scenario,
                rep=rep,
                n_train=args.n_train,
                n_test=args.n_test,
                method=METHOD,
                heldout_log_score=score,
                fit_seconds=fit_seconds,
                eval_seconds=eval_seconds,
                seed=seed,
                hyperparameter=(
                    f"fixed_geometry=true;n_grid={args.n_grid};em_max_iter={args.em_max_iter};"
                    f"em_tol={args.em_tol:g};min_weight={args.min_weight:g}"
                ),
                status="ok" if fit.converged else "warning",
                converged=fit.converged,
                n_iter=fit.iterations,
                nonfinite_count=nonfinite,
                final_objective=fit.final_objective,
                final_max_weight_change=fit.final_max_weight_change,
            )
        )
        recovery = recovery_metrics(
            scenario=scenario,
            rep=rep,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=seed,
            dist=dist,
            fitted_weights=fit.weights,
            resp_test=resp_test,
            test_labels=test_labels,
            fit=fit,
        )
        weight_rows = fitted_weight_rows(
            scenario=scenario,
            rep=rep,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=seed,
            dist=dist,
            fit=fit,
            train_labels=train_labels,
            test_labels=test_labels,
        )
        responsibility_rows = responsibility_summary_rows(
            scenario=scenario,
            rep=rep,
            n_test=args.n_test,
            seed=seed,
            dist=dist,
            resp_test=resp_test,
            test_labels=test_labels,
        )
        convergence_rows = [
            convergence_row(
                scenario=scenario,
                rep=rep,
                n_train=args.n_train,
                n_test=args.n_test,
                seed=seed,
                dist=dist,
                fit=fit,
                args=args,
            )
        ]
    except Exception as exc:  # pragma: no cover - exercised by integration failures
        rows.append(
            failed_density_row(
                scenario=scenario,
                rep=rep,
                n_train=args.n_train,
                n_test=args.n_test,
                method=METHOD,
                seed=seed,
                hyperparameter=(
                    f"fixed_geometry=true;n_grid={args.n_grid};em_max_iter={args.em_max_iter};"
                    f"em_tol={args.em_tol:g};min_weight={args.min_weight:g}"
                ),
                exc=exc,
            )
        )
        recovery = failed_recovery_row(
            scenario=scenario,
            rep=rep,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=seed,
            exc=exc,
        )
        weight_rows = []
        responsibility_rows = []
        convergence_rows = [
            failed_convergence_row(
                scenario=scenario,
                rep=rep,
                n_train=args.n_train,
                n_test=args.n_test,
                seed=seed,
                exc=exc,
                args=args,
            )
        ]

    start = time.perf_counter()
    try:
        gaussian = fit_gaussian(train, args.reg)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        logpdf = gaussian.logpdf(test)
        eval_seconds = time.perf_counter() - start
        score, nonfinite = score_and_nonfinite(logpdf)
        rows.append(
            density_row(
                scenario=scenario,
                rep=rep,
                n_train=args.n_train,
                n_test=args.n_test,
                method="Gaussian",
                heldout_log_score=score,
                fit_seconds=fit_seconds,
                eval_seconds=eval_seconds,
                seed=seed,
                hyperparameter=f"reg={args.reg:g}",
                converged=True,
                n_iter=0,
                nonfinite_count=nonfinite,
            )
        )
    except Exception as exc:  # pragma: no cover - exercised by integration failures
        rows.append(
            failed_density_row(
                scenario=scenario,
                rep=rep,
                n_train=args.n_train,
                n_test=args.n_test,
                method="Gaussian",
                seed=seed,
                hyperparameter=f"reg={args.reg:g}",
                exc=exc,
                fit_seconds=time.perf_counter() - start,
            )
        )

    for k in args.gmm_components:
        start = time.perf_counter()
        method = f"GMM-{min(k, args.n_train)}"
        hyperparameter = f"K={k};reg={args.reg:g};max_iter={args.gmm_max_iter};tol={args.gmm_tol:g}"
        try:
            gmm = fit_gmm_em(
                train,
                k,
                random_state=seed + 17 * k,
                reg=args.reg,
                max_iter=args.gmm_max_iter,
                tol=args.gmm_tol,
            )
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            logpdf = gmm.logpdf(test)
            eval_seconds = time.perf_counter() - start
            score, nonfinite = score_and_nonfinite(logpdf)
            rows.append(
                density_row(
                    scenario=scenario,
                    rep=rep,
                    n_train=args.n_train,
                    n_test=args.n_test,
                    method=method,
                    heldout_log_score=score,
                    fit_seconds=fit_seconds,
                    eval_seconds=eval_seconds,
                    seed=seed,
                    hyperparameter=hyperparameter,
                    converged="",
                    n_iter="",
                    nonfinite_count=nonfinite,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by integration failures
            rows.append(
                failed_density_row(
                    scenario=scenario,
                    rep=rep,
                    n_train=args.n_train,
                    n_test=args.n_test,
                    method=method,
                    seed=seed,
                    hyperparameter=hyperparameter,
                    exc=exc,
                    fit_seconds=time.perf_counter() - start,
                )
            )

    if args.kde:
        start = time.perf_counter()
        try:
            kde = fit_kde(train, random_state=seed + 7919, jitter=args.kde_jitter)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            logpdf = kde.logpdf(test)
            eval_seconds = time.perf_counter() - start
            score, nonfinite = score_and_nonfinite(logpdf)
            rows.append(
                density_row(
                    scenario=scenario,
                    rep=rep,
                    n_train=args.n_train,
                    n_test=args.n_test,
                    method="KDE",
                    heldout_log_score=score,
                    fit_seconds=fit_seconds,
                    eval_seconds=eval_seconds,
                    seed=seed,
                    hyperparameter=f"bandwidth=scott;jitter={args.kde_jitter:g}",
                    converged=True,
                    n_iter=0,
                    nonfinite_count=nonfinite,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by integration failures
            rows.append(
                failed_density_row(
                    scenario=scenario,
                    rep=rep,
                    n_train=args.n_train,
                    n_test=args.n_test,
                    method="KDE",
                    seed=seed,
                    hyperparameter=f"bandwidth=scott;jitter={args.kde_jitter:g}",
                    exc=exc,
                    fit_seconds=time.perf_counter() - start,
                )
            )

    return rows, recovery, weight_rows, responsibility_rows, convergence_rows


def summarize_density(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(str(r["scenario"]), str(r["method"])) for r in rows})
    out = []
    for scenario, method in keys:
        sub = [r for r in rows if str(r["scenario"]) == scenario and str(r["method"]) == method]
        ok_scores = finite_array(
            [r["heldout_log_score"] for r in sub if str(r.get("status", "")) in {"ok", "warning"}]
        )
        fit = finite_array([r["fit_seconds"] for r in sub])
        eval_time = finite_array([r["eval_seconds"] for r in sub])
        statuses = [str(r.get("status", "")) for r in sub]
        out.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": scenario,
                "scenario_family": SCENARIO_FAMILY,
                "method": method,
                "mean_heldout_log_score": float(np.mean(ok_scores)) if ok_scores.size else float("nan"),
                "se_heldout_log_score": se(ok_scores),
                "rank_within_scenario": "",
                "mean_fit_seconds": float(np.mean(fit)) if fit.size else float("nan"),
                "mean_eval_seconds": float(np.mean(eval_time)) if eval_time.size else float("nan"),
                "reps": len(sub),
                "failure_rate": float(np.mean([status == "failed" for status in statuses])),
                "warning_rate": float(np.mean([status == "warning" for status in statuses])),
                "ok_reps": int(sum(status == "ok" for status in statuses)),
            }
        )
    for scenario in sorted({str(r["scenario"]) for r in out}):
        scenario_rows = [
            r for r in out
            if str(r["scenario"]) == scenario and np.isfinite(as_float(r["mean_heldout_log_score"]))
        ]
        scenario_rows.sort(key=lambda r: float(r["mean_heldout_log_score"]), reverse=True)
        for rank, row in enumerate(scenario_rows, start=1):
            row["rank_within_scenario"] = rank
    return out


def summarize_runtime(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(str(r["scenario"]), str(r["method"])) for r in rows})
    out = []
    for scenario, method in keys:
        sub = [r for r in rows if str(r["scenario"]) == scenario and str(r["method"]) == method]
        fit = finite_array([r["fit_seconds"] for r in sub])
        eval_time = finite_array([r["eval_seconds"] for r in sub])
        out.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": scenario,
                "scenario_family": SCENARIO_FAMILY,
                "method": method,
                "reps": len(sub),
                "total_fit_seconds": float(np.sum(fit)) if fit.size else 0.0,
                "mean_fit_seconds": float(np.mean(fit)) if fit.size else float("nan"),
                "total_eval_seconds": float(np.sum(eval_time)) if eval_time.size else 0.0,
                "mean_eval_seconds": float(np.mean(eval_time)) if eval_time.size else float("nan"),
                "peak_memory_mb": "",
            }
        )
    return out


def summarize_recovery(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_fields = [
        "branch_accuracy",
        "stratum_accuracy",
        "atom_mass_bias",
        "atom_mass_rmse",
        "branch_mass_bias",
        "branch_mass_rmse",
        "component_mass_rmse",
        "component_mass_mae",
        "component_mass_max_abs_bias",
        "responsibility_brier",
        "mean_true_component_responsibility",
        "em_iterations",
    ]
    out = []
    for scenario in sorted({str(r["scenario"]) for r in rows}):
        sub = [r for r in rows if str(r["scenario"]) == scenario]
        statuses = [str(r.get("status", "")) for r in sub]
        item: dict[str, object] = {
            "protocol_version": PROTOCOL_VERSION,
            "scenario": scenario,
            "method": METHOD,
            "reps": len(sub),
            "failure_rate": float(np.mean([status == "failed" for status in statuses])),
            "warning_rate": float(np.mean([status == "warning" for status in statuses])),
            "em_converged_reps": int(sum(bool(r.get("em_converged", False)) for r in sub)),
            "truth_labels_available": True,
            "estimator_scope": ESTIMATOR_SCOPE,
        }
        for field in metric_fields:
            vals = finite_array([r.get(field, float("nan")) for r in sub])
            item[f"mean_{field}"] = float(np.mean(vals)) if vals.size else float("nan")
            item[f"se_{field}"] = se(vals)
        out.append(item)
    return out


def write_parameters(path: Path, args, command: str, row_counts: dict[str, int] | None = None) -> None:
    row_counts = row_counts or {}
    lines = [
        "Known-skeleton weight-only fitted SFD diagnostic",
        f"protocol_version={PROTOCOL_VERSION}",
        f"scenario_family={SCENARIO_FAMILY}",
        f"estimator_scope={ESTIMATOR_SCOPE}",
        f"command={command}",
        f"python={sys.executable}",
        f"pythonpath={os.environ.get('PYTHONPATH', '')}",
        f"scenarios={','.join(args.scenarios)}",
        f"n_train={args.n_train}",
        f"n_test={args.n_test}",
        f"reps={args.reps}",
        f"base_seed={args.seed}",
        f"seed_formula={SEED_FORMULA}",
        f"method={METHOD}",
        f"gmm_components={','.join(map(str, args.gmm_components))}",
        f"kde={args.kde}",
        f"kde_jitter={args.kde_jitter}",
        f"n_grid={args.n_grid}",
        f"em_max_iter={args.em_max_iter}",
        f"em_tol={args.em_tol}",
        f"min_weight={args.min_weight}",
        f"reg={args.reg}",
        f"gmm_max_iter={args.gmm_max_iter}",
        f"gmm_tol={args.gmm_tol}",
        f"out={args.out}",
        f"figure_dir={args.figure_dir}",
        f"figure_prefix={args.figure_prefix}",
    ]
    for name, count in sorted(row_counts.items()):
        lines.append(f"rows.{name}={count}")
    path.write_text("\n".join(lines) + "\n")


def write_parameters_json(path: Path, args, command: str, row_counts: dict[str, int] | None = None) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario_family": SCENARIO_FAMILY,
        "estimator_scope": ESTIMATOR_SCOPE,
        "command": command,
        "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "scenarios": args.scenarios,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "reps": args.reps,
        "base_seed": args.seed,
        "seed_formula": SEED_FORMULA,
        "method": METHOD,
        "gmm_components": args.gmm_components,
        "kde": args.kde,
        "kde_jitter": args.kde_jitter,
        "n_grid": args.n_grid,
        "em_max_iter": args.em_max_iter,
        "em_tol": args.em_tol,
        "min_weight": args.min_weight,
        "reg": args.reg,
        "gmm_max_iter": args.gmm_max_iter,
        "gmm_tol": args.gmm_tol,
        "out": args.out,
        "figure_dir": args.figure_dir,
        "figure_prefix": args.figure_prefix,
        "row_counts": row_counts or {},
        "standard_speed_target": {"n_train": 1000, "n_test": 1000, "reps": 5},
        "paper_facing_target_reps": 25,
    }
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def maybe_plot(
    out_dir: Path,
    figure_dir: Path,
    figure_prefix: str,
    summary_rows: list[dict[str, object]],
    recovery_summary: list[dict[str, object]],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from sfd_figure_style import panel_title, set_sfd_matplotlib_style
    except Exception:
        return []

    set_sfd_matplotlib_style(plt, base_size=9.0, title_size=10.0, label_size=9.0, tick_size=8.0)
    scenario_order = ["snowflake", "stylized_face", "doraemon", "maze", "ecg", "torus"]
    scenario_labels = {
        "snowflake": "Snowflake",
        "stylized_face": "Stylized face",
        "doraemon": "Stylized face",
        "maze": "Maze",
        "ecg": "ECG",
        "torus": "Torus",
    }
    method_order = ["Gaussian", "KDE", "GMM-2", "GMM-4", "GMM-8", METHOD, "Oracle SFD"]
    color = {
        "Gaussian": "#8fd4c7",
        "KDE": "#a8d8f0",
        "GMM-2": "#b7d99c",
        "GMM-4": "#c7b6e6",
        "GMM-8": "#ffd08a",
        METHOD: "#2aa7a1",
        "Oracle SFD": "#f48b7a",
    }
    fresh_grid = "#dfeaed"
    ink = "#344955"
    rmse_fill = "#7ecdbb"
    accuracy_fill = "#ff8f7a"

    scenarios = [s for s in scenario_order if any(str(r["scenario"]) == s for r in summary_rows)]
    fig = plt.figure(figsize=(11.2, 7.4), constrained_layout=True, facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0])
    ax_score = fig.add_subplot(grid[0, :])
    ax_rmse = fig.add_subplot(grid[1, 0])
    ax_accuracy = fig.add_subplot(grid[1, 1])
    for ax in (ax_score, ax_rmse, ax_accuracy):
        ax.set_facecolor("#fbfdfd")

    x_base = np.arange(len(scenarios))
    width = 0.11
    for idx, method in enumerate(method_order):
        vals = []
        errs = []
        for scenario in scenarios:
            match = [
                r for r in summary_rows
                if str(r["scenario"]) == scenario and str(r["method"]) == method
            ]
            vals.append(float(match[0]["mean_heldout_log_score"]) if match else np.nan)
            errs.append(float(match[0]["se_heldout_log_score"]) if match else 0.0)
        offset = (idx - (len(method_order) - 1) / 2.0) * width
        ax_score.bar(
            x_base + offset,
            vals,
            width=width,
            yerr=errs,
            label=method,
            color=color.get(method, "#b8c0cc"),
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 0.7, "ecolor": ink, "capsize": 2.0},
        )
    ax_score.set_xticks(x_base)
    ax_score.set_xticklabels([scenario_labels.get(s, s) for s in scenarios])
    ax_score.set_ylabel("mean held-out log score")
    panel_title(ax_score, "Known-skeleton weight-only fitted SFD diagnostic")
    ax_score.grid(axis="y", color=fresh_grid, linewidth=0.85)
    ax_score.spines["top"].set_visible(False)
    ax_score.spines["right"].set_visible(False)
    ax_score.legend(ncols=4, fontsize=8, frameon=False, loc="lower left")

    rmse = []
    rmse_se = []
    accuracy_se = []
    accuracy = []
    for scenario in scenarios:
        match = [r for r in recovery_summary if str(r["scenario"]) == scenario]
        rmse.append(float(match[0]["mean_component_mass_rmse"]) if match else np.nan)
        rmse_se.append(float(match[0]["se_component_mass_rmse"]) if match else 0.0)
        accuracy.append(float(match[0]["mean_stratum_accuracy"]) if match else np.nan)
        accuracy_se.append(float(match[0]["se_stratum_accuracy"]) if match else 0.0)
    ax_rmse.bar(
        x_base,
        rmse,
        width=0.56,
        yerr=rmse_se,
        color=rmse_fill,
        edgecolor="white",
        linewidth=0.6,
        error_kw={"elinewidth": 0.8, "ecolor": ink, "capsize": 2.0},
    )
    ax_rmse.set_xticks(x_base)
    ax_rmse.set_xticklabels([scenario_labels.get(s, s) for s in scenarios], rotation=18, ha="right")
    ax_rmse.set_ylabel("component mass RMSE")
    panel_title(ax_rmse, "Weight recovery")
    ax_rmse.grid(axis="y", color=fresh_grid, linewidth=0.85)
    ax_rmse.spines["top"].set_visible(False)
    ax_rmse.spines["right"].set_visible(False)

    ax_accuracy.errorbar(
        x_base,
        accuracy,
        yerr=accuracy_se,
        fmt="o",
        color=accuracy_fill,
        ecolor=ink,
        elinewidth=0.8,
        capsize=2.5,
        markersize=5.5,
    )
    ax_accuracy.set_xticks(x_base)
    ax_accuracy.set_xticklabels([scenario_labels.get(s, s) for s in scenarios], rotation=18, ha="right")
    ax_accuracy.set_ylim(-0.02, 1.04)
    ax_accuracy.set_ylabel("stratum accuracy")
    panel_title(ax_accuracy, "Component-label diagnostic")
    ax_accuracy.grid(axis="y", color=fresh_grid, linewidth=0.85)
    ax_accuracy.spines["top"].set_visible(False)
    ax_accuracy.spines["right"].set_visible(False)

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / f"{figure_prefix}.png",
        out_dir / f"{figure_prefix}.pdf",
        figure_dir / f"{figure_prefix}.png",
        figure_dir / f"{figure_prefix}.pdf",
    ]
    unique_paths = []
    for path in paths:
        if path not in unique_paths:
            unique_paths.append(path)
    for path in unique_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            fig.savefig(path, dpi=220)
        else:
            fig.savefig(path)
    plt.close(fig)
    return unique_paths


def write_execution_report(
    path: Path,
    *,
    args,
    command: str,
    row_counts: dict[str, int],
    score_rows: list[dict[str, object]],
    recovery_rows: list[dict[str, object]],
    generated_files: list[Path],
) -> None:
    generated_utc = datetime.now(timezone.utc).isoformat()
    density_failures = sum(str(row.get("status")) == "failed" for row in score_rows)
    density_warnings = sum(str(row.get("status")) == "warning" for row in score_rows)
    recovery_failures = sum(str(row.get("status")) == "failed" for row in recovery_rows)
    recovery_warnings = sum(str(row.get("status")) == "warning" for row in recovery_rows)
    standard_speed = args.n_train == 1000 and args.n_test == 1000 and args.reps == 5
    lines = [
        "# Department B: Known-Skeleton Weight-Only Fitted SFD",
        "",
        f"Generated UTC: `{generated_utc}`",
        "",
        "## Scope",
        "",
        f"- Protocol label: `{PROTOCOL_VERSION}`",
        f"- Estimator label: `{ESTIMATOR_SCOPE}`",
        "- This is not topology learning and does not estimate component geometry.",
        "- These outputs are diagnostic/integration evidence until the full frozen protocol and audits pass.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Parameters",
        "",
        f"- Scenarios: `{','.join(args.scenarios)}`",
        f"- Sample sizes: `n_train={args.n_train}`, `n_test={args.n_test}`, `reps={args.reps}`",
        f"- Standard-speed Department-B target met: `{standard_speed}`",
        f"- Base seed: `{args.seed}`",
        f"- Seed formula: `{SEED_FORMULA}`",
        f"- Quadrature grid: `n_grid={args.n_grid}`",
        f"- EM: `max_iter={args.em_max_iter}`, `tol={args.em_tol}`, `min_weight={args.min_weight}`",
        f"- Baseline grid in this diagnostic: `GMM={','.join(map(str, args.gmm_components))}`, `KDE={args.kde}`",
        "",
        "## Outputs",
        "",
    ]
    for file_path in generated_files:
        key = file_path.name
        suffix = f" ({row_counts[key]} rows)" if key in row_counts else ""
        lines.append(f"- `{file_path}`{suffix}")
    if standard_speed:
        repetition_note = (
            "This standard diagnostic uses 5 repetitions; the frozen paper-facing target remains "
            "25 repetitions plus broader baselines, calibration extraction, and final audit."
        )
    else:
        repetition_note = (
            "This is a smaller diagnostic run; the Department-B standard profile remains "
            "n_train=1000, n_test=1000, reps=5, and the frozen paper-facing target remains 25 repetitions."
        )
    lines.extend(
        [
            "",
            "## Status Summary",
            "",
            f"- Density rows: `{len(score_rows)}`; failures: `{density_failures}`; warnings: `{density_warnings}`",
            f"- Recovery rows: `{len(recovery_rows)}`; failures: `{recovery_failures}`; warnings: `{recovery_warnings}`",
            "- Responsibility output is summarized by true component; full per-observation responsibility matrices are not written.",
            "",
            "## Claim Boundary And Blockers",
            "",
            "- The fitted SFD rows are known-skeleton/weight-only rows, not learned-topology SFD rows.",
            "- Latent-coordinate RMSE and off-trajectory variance metrics are present as schema columns but not estimated by this weight-only diagnostic.",
            f"- {repetition_note}",
            "- No manuscript performance claims were edited by this run.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scenarios", type=parse_str_list, default=parse_str_list("snowflake,doraemon,maze,ecg,torus"))
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--gmm-components", type=parse_int_list, default=parse_int_list("2,4,8"))
    parser.add_argument("--kde", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kde-jitter", type=float, default=1e-6)
    parser.add_argument("--n-grid", type=int, default=64)
    parser.add_argument("--em-max-iter", type=int, default=300)
    parser.add_argument("--em-tol", type=float, default=1e-6)
    parser.add_argument("--min-weight", type=float, default=1e-12)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--gmm-max-iter", type=int, default=80)
    parser.add_argument("--gmm-tol", type=float, default=1e-5)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--figure-prefix", type=str, default="known_skeleton_weight_only_sfd_standard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.figure_dir is None:
        args.figure_dir = args.out / "figures"
    command = shlex.join([sys.executable, *sys.argv])
    write_parameters(args.out / "run_parameters.txt", args, command)
    write_parameters_json(args.out / "run_parameters.json", args, command)

    score_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    responsibility_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    for rep in range(1, args.reps + 1):
        for idx, scenario in enumerate(args.scenarios):
            seed = args.seed + 10000 * rep + 101 * idx
            print(f"[known-skeleton] scenario={scenario} rep={rep} seed={seed}", flush=True)
            rows, recovery, weights, responsibilities, convergence = run_one(args, scenario, rep, seed)
            score_rows.extend(rows)
            recovery_rows.append(recovery)
            weight_rows.extend(weights)
            responsibility_rows.extend(responsibilities)
            convergence_rows.extend(convergence)

    summary_rows = summarize_density(score_rows)
    runtime_rows = summarize_runtime(score_rows)
    recovery_summary = summarize_recovery(recovery_rows)
    generated_files = [
        args.out / "heldout_log_scores.csv",
        args.out / "benchmark_summary.csv",
        args.out / "runtime.csv",
        args.out / "recovery_scores.csv",
        args.out / "recovery_summary.csv",
        args.out / "fitted_weights.csv",
        args.out / "responsibility_summary.csv",
        args.out / "convergence_metadata.csv",
        args.out / "run_parameters.txt",
        args.out / "run_parameters.json",
        args.out / "execution_report.md",
    ]
    write_csv(args.out / "heldout_log_scores.csv", score_rows)
    write_csv(args.out / "benchmark_summary.csv", summary_rows)
    write_csv(args.out / "runtime.csv", runtime_rows)
    write_csv(args.out / "recovery_scores.csv", recovery_rows)
    write_csv(args.out / "recovery_summary.csv", recovery_summary)
    write_csv(args.out / "fitted_weights.csv", weight_rows)
    write_csv(args.out / "responsibility_summary.csv", responsibility_rows)
    write_csv(args.out / "convergence_metadata.csv", convergence_rows)
    # Backward-compatible alias for the earlier pilot file name.
    write_csv(args.out / "recovery_metrics.csv", recovery_rows)
    generated_files.append(args.out / "recovery_metrics.csv")
    figure_paths = maybe_plot(args.out, args.figure_dir, args.figure_prefix, summary_rows, recovery_summary)
    generated_files.extend(figure_paths)

    row_counts = {
        "heldout_log_scores.csv": len(score_rows),
        "benchmark_summary.csv": len(summary_rows),
        "runtime.csv": len(runtime_rows),
        "recovery_scores.csv": len(recovery_rows),
        "recovery_summary.csv": len(recovery_summary),
        "fitted_weights.csv": len(weight_rows),
        "responsibility_summary.csv": len(responsibility_rows),
        "convergence_metadata.csv": len(convergence_rows),
        "recovery_metrics.csv": len(recovery_rows),
    }
    write_parameters(args.out / "run_parameters.txt", args, command, row_counts)
    write_parameters_json(args.out / "run_parameters.json", args, command, row_counts)
    write_execution_report(
        args.out / "execution_report.md",
        args=args,
        command=command,
        row_counts=row_counts,
        score_rows=score_rows,
        recovery_rows=recovery_rows,
        generated_files=generated_files,
    )
    print(f"Wrote {args.out / 'heldout_log_scores.csv'}")
    print(f"Wrote {args.out / 'benchmark_summary.csv'}")
    print(f"Wrote {args.out / 'runtime.csv'}")
    print(f"Wrote {args.out / 'recovery_scores.csv'}")
    print(f"Wrote {args.out / 'fitted_weights.csv'}")
    print(f"Wrote {args.out / 'responsibility_summary.csv'}")
    print(f"Wrote {args.out / 'execution_report.md'}")


if __name__ == "__main__":
    main()
