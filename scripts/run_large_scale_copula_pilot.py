#!/usr/bin/env python3
"""Run a deterministic copula-family pilot for the JMLR benchmark layer.

The pilot targets dependence-focused full-dimensional distributions. It uses
closed-form Gaussian, Student-t, and Clayton copula generators and densities,
then compares generic density estimators under the current held-out log-score
schema. Vine-copula generation is dependency-gated because it relies on
optional R packages and should not silently fall back to an unverified model.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tmp" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / "tmp" / "cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "cache"))

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import gamma as gamma_dist
from scipy.stats import gaussian_kde, lognorm, multivariate_normal, norm, t


PROTOCOL_VERSION = "jmlr_frozen_benchmark_v1"
SCENARIO_FAMILY = "external_copula_full_dimensional"
DEFAULT_OUT = ROOT / "outputs" / "jmlr_frozen_benchmark_v1" / "external_family" / "copula_family"
DEFAULT_FIGURE_PREFIX = "jmlr_large_scale_copula_heldout_log_score"
SCORE_FIELDNAMES = [
    "protocol_version",
    "scenario",
    "scenario_family",
    "dimension",
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
    "final_objective",
    "error_type",
    "error_message",
    "nonfinite_count",
]
SUMMARY_FIELDNAMES = [
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
RUNTIME_FIELDNAMES = [
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
REQUIRED_HELDOUT_COLUMNS = {
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
}


def ar1_corr(d: int, rho: float) -> np.ndarray:
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def clip_u(u: np.ndarray) -> np.ndarray:
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def multivariate_t_logpdf(x: np.ndarray, corr: np.ndarray, df: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    d = x.shape[1]
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        raise ValueError("Correlation matrix must be positive definite.")
    inv = np.linalg.inv(corr)
    quad = np.einsum("ij,jk,ik->i", x, inv, x)
    const = gammaln((df + d) / 2.0) - gammaln(df / 2.0)
    const -= 0.5 * (d * math.log(df * math.pi) + logdet)
    return const - 0.5 * (df + d) * np.log1p(quad / df)


@dataclass(frozen=True)
class CopulaScenario:
    name: str
    label: str
    dimension: int
    description: str
    parameter_summary: dict[str, object]
    sampler: Callable[[np.random.Generator, int], np.ndarray]
    logpdf: Callable[[np.ndarray], np.ndarray]


def make_gaussian_copula_lognormal() -> CopulaScenario:
    d = 10
    rho = 0.55
    corr = ar1_corr(d, rho)
    marginal_mu = np.linspace(-0.25, 0.35, d)
    marginal_sigma = np.linspace(0.35, 0.85, d)

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        z = rng.multivariate_normal(np.zeros(d), corr, size=n)
        u = clip_u(norm.cdf(z))
        cols = [
            lognorm.ppf(u[:, j], s=marginal_sigma[j], scale=math.exp(marginal_mu[j]))
            for j in range(d)
        ]
        return np.column_stack(cols)

    def logpdf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        log_margins = np.zeros(x.shape[0])
        z_cols = []
        for j in range(d):
            dist = lognorm(s=marginal_sigma[j], scale=math.exp(marginal_mu[j]))
            u_j = clip_u(dist.cdf(x[:, j]))
            z_cols.append(norm.ppf(u_j))
            log_margins += dist.logpdf(x[:, j])
        z = np.column_stack(z_cols)
        log_copula = multivariate_normal.logpdf(z, mean=np.zeros(d), cov=corr)
        log_copula -= np.sum(norm.logpdf(z), axis=1)
        return log_copula + log_margins

    return CopulaScenario(
        name="gaussian_copula_lognormal_10d",
        label="Gaussian copula lognormal 10d",
        dimension=d,
        description="Gaussian copula with AR(1) correlation and heterogeneous lognormal margins.",
        parameter_summary={
            "copula": "gaussian",
            "rho": rho,
            "margin": "lognormal",
            "margin_mu_min": float(marginal_mu.min()),
            "margin_mu_max": float(marginal_mu.max()),
            "margin_sigma_min": float(marginal_sigma.min()),
            "margin_sigma_max": float(marginal_sigma.max()),
        },
        sampler=sampler,
        logpdf=logpdf,
    )


def make_t_copula_t_margins() -> CopulaScenario:
    d = 10
    rho = 0.45
    copula_df = 4.5
    corr = ar1_corr(d, rho)
    margin_df = np.linspace(3.0, 8.0, d)
    margin_loc = np.linspace(-0.7, 0.7, d)
    margin_scale = np.linspace(0.65, 1.25, d)

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        z = rng.multivariate_normal(np.zeros(d), corr, size=n)
        chi = rng.chisquare(copula_df, size=n)
        y = z / np.sqrt(chi[:, None] / copula_df)
        u = clip_u(t.cdf(y, df=copula_df))
        cols = [
            margin_loc[j] + margin_scale[j] * t.ppf(u[:, j], df=margin_df[j])
            for j in range(d)
        ]
        return np.column_stack(cols)

    def logpdf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        log_margins = np.zeros(x.shape[0])
        y_cols = []
        for j in range(d):
            standardized = (x[:, j] - margin_loc[j]) / margin_scale[j]
            u_j = clip_u(t.cdf(standardized, df=margin_df[j]))
            y_cols.append(t.ppf(u_j, df=copula_df))
            log_margins += t.logpdf(standardized, df=margin_df[j]) - math.log(margin_scale[j])
        y = np.column_stack(y_cols)
        log_copula = multivariate_t_logpdf(y, corr, copula_df)
        log_copula -= np.sum(t.logpdf(y, df=copula_df), axis=1)
        return log_copula + log_margins

    return CopulaScenario(
        name="t_copula_t_margins_10d",
        label="t copula t margins 10d",
        dimension=d,
        description="Student-t copula with heavy-tailed, heterogeneous Student-t margins.",
        parameter_summary={
            "copula": "student_t",
            "rho": rho,
            "copula_df": copula_df,
            "margin": "student_t",
            "margin_df_min": float(margin_df.min()),
            "margin_df_max": float(margin_df.max()),
        },
        sampler=sampler,
        logpdf=logpdf,
    )


def make_clayton_copula_gamma() -> CopulaScenario:
    d = 5
    theta = 1.6
    shape = np.linspace(1.4, 3.0, d)
    scale = np.linspace(0.55, 1.25, d)

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        frailty = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)
        expo = rng.exponential(scale=1.0, size=(n, d))
        u = clip_u((1.0 + expo / frailty[:, None]) ** (-1.0 / theta))
        cols = [gamma_dist.ppf(u[:, j], a=shape[j], scale=scale[j]) for j in range(d)]
        return np.column_stack(cols)

    def logpdf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        log_margins = np.zeros(x.shape[0])
        u_cols = []
        for j in range(d):
            dist = gamma_dist(a=shape[j], scale=scale[j])
            u_j = clip_u(dist.cdf(x[:, j]))
            u_cols.append(u_j)
            log_margins += dist.logpdf(x[:, j])
        u = np.column_stack(u_cols)
        log_factor = sum(math.log1p(k * theta) for k in range(1, d))
        inner = np.sum(u ** (-theta), axis=1) - d + 1.0
        log_copula = log_factor
        log_copula += np.sum((-theta - 1.0) * np.log(u), axis=1)
        log_copula += -(d + 1.0 / theta) * np.log(inner)
        return log_copula + log_margins

    return CopulaScenario(
        name="clayton_copula_gamma_5d",
        label="Clayton copula gamma 5d",
        dimension=d,
        description="Clayton Archimedean copula with heterogeneous gamma margins.",
        parameter_summary={
            "copula": "clayton",
            "theta": theta,
            "margin": "gamma",
            "shape_min": float(shape.min()),
            "shape_max": float(shape.max()),
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
        },
        sampler=sampler,
        logpdf=logpdf,
    )


SCENARIO_FACTORIES = {
    "gaussian_copula_lognormal_10d": make_gaussian_copula_lognormal,
    "t_copula_t_margins_10d": make_t_copula_t_margins,
    "clayton_copula_gamma_5d": make_clayton_copula_gamma,
}


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
    converged: bool
    n_iter: int
    final_objective: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for weight, mean, cov in zip(self.weights, self.means, self.covs):
            cols.append(np.log(max(float(weight), 1e-300)) + multivariate_normal.logpdf(x, mean=mean, cov=cov))
        return logsumexp(np.vstack(cols).T, axis=1)


@dataclass
class KDEFit:
    kde: gaussian_kde

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.kde.logpdf(x.T), dtype=float)


def regularized_cov(x: np.ndarray, reg: float) -> np.ndarray:
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return cov + reg * np.eye(x.shape[1])


def fit_gaussian(x: np.ndarray, reg: float) -> GaussianFit:
    return GaussianFit(mean=np.mean(x, axis=0), cov=regularized_cov(x, reg))


def fit_kde(x: np.ndarray, *, random_state: int, jitter: float) -> KDEFit:
    try:
        return KDEFit(kde=gaussian_kde(x.T))
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(random_state)
        return KDEFit(kde=gaussian_kde((x + rng.normal(scale=jitter, size=x.shape)).T))


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
    covs = np.repeat(regularized_cov(x, reg)[None, :, :], k, axis=0)
    weights = np.full(k, 1.0 / k)
    previous = -np.inf
    converged = False
    objective = -np.inf
    iteration = 0

    for iteration in range(1, max_iter + 1):
        log_resp_cols = []
        for weight, mean, cov in zip(weights, means, covs):
            log_resp_cols.append(np.log(max(float(weight), 1e-300)) + multivariate_normal.logpdf(x, mean=mean, cov=cov))
        log_resp = np.vstack(log_resp_cols).T
        log_norm = logsumexp(log_resp, axis=1)
        objective = float(np.mean(log_norm))
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
    return GMMFit(weights=weights, means=means, covs=covs, converged=converged, n_iter=iteration, final_objective=objective)


def mean_log_score(logpdf: np.ndarray) -> float:
    values = np.asarray(logpdf, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def score_logpdf_values(logpdf: np.ndarray) -> tuple[float | str, int, str]:
    values = np.asarray(logpdf, dtype=float)
    finite = values[np.isfinite(values)]
    nonfinite_count = int(values.size - finite.size)
    if finite.size == 0:
        return "NA", nonfinite_count, "warning"
    status = "warning" if nonfinite_count else "ok"
    return float(np.mean(finite)), nonfinite_count, status


def parse_csv_list(text: str) -> list[str]:
    out = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return out


def parse_int_list(text: str) -> list[int]:
    out = [int(piece) for piece in text.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("All integers must be positive.")
    return sorted(set(out))


def parse_methods(text: str | None) -> tuple[bool, list[int], bool, bool]:
    if not text:
        return True, [2, 4, 8], True, True
    gaussian = False
    gmm_components: list[int] = []
    kde = False
    oracle = False
    for piece in [p.strip().lower().replace("-", "") for p in text.split(",") if p.strip()]:
        if piece == "gaussian":
            gaussian = True
        elif piece == "kde":
            kde = True
        elif piece in {"oracle", "copulaoracle"}:
            oracle = True
        elif piece.startswith("gmm") and piece[3:].isdigit():
            gmm_components.append(int(piece[3:]))
        else:
            raise argparse.ArgumentTypeError(f"Unknown method: {piece}")
    return gaussian, sorted(set(gmm_components)), kde, oracle


def inspect_r_packages(timeout_seconds: float) -> dict[str, object]:
    packages = ["rvinecopulib", "VineCopula", "copula"]
    cmd = [
        "Rscript",
        "--vanilla",
        "-e",
        (
            "pkgs <- c('rvinecopulib','VineCopula','copula'); "
            "for (p in pkgs) cat(p, requireNamespace(p, quietly=TRUE), '\\n')"
        ),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return {"status": "Rscript_not_found", "available_packages": [], "details": "Rscript is not on PATH."}
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "available_packages": [],
            "details": f"R package inventory exceeded {timeout_seconds:g} seconds.",
        }

    available = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].upper() == "TRUE":
            available.append(parts[0])
    status = "ok" if completed.returncode == 0 else f"returncode_{completed.returncode}"
    return {
        "status": status,
        "available_packages": available,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def inspect_vine_dependencies(timeout_seconds: float) -> dict[str, object]:
    python_candidates = ["pyvinecopulib", "vinecopulas", "copulas"]
    python_available = [name for name in python_candidates if importlib.util.find_spec(name) is not None]
    r_status = inspect_r_packages(timeout_seconds)
    r_available = list(r_status.get("available_packages", []))
    adapter_status = "not_implemented_in_this_runner"
    reliable_for_rows = False
    decision = "no_go"
    reason = (
        "No verified local vine-copula benchmark adapter is available. A reliable row requires "
        "a dependency plus deterministic sampling, fitted-density evaluation, marginal handling, "
        "and frozen-schema failure metadata."
    )
    if not python_available and not r_available:
        reason = (
            "No Python vine package among pyvinecopulib/vinecopulas/copulas and no R package among "
            "rvinecopulib/VineCopula/copula was detected within the dependency probe."
        )
    elif "pyvinecopulib" in python_available:
        reason = (
            "pyvinecopulib is importable, but this runner does not yet contain an audited adapter "
            "that transforms margins, fits the vine, evaluates full-dimensional log densities, and "
            "records convergence/failure metadata."
        )
    elif r_available:
        reason = (
            "An R vine/copula package is available, but this runner does not yet contain a scripted "
            "R bridge with deterministic data exchange, full-density evaluation, and frozen-schema "
            "failure handling."
        )
    plan = [
        "Select one implementation path: Python pyvinecopulib preferred for in-process deterministic rows; otherwise R rvinecopulib via a version-pinned Rscript bridge.",
        "Freeze a vine scenario name, dimension, pair-copula families, marginal distributions, seed formula, and train/test split rule.",
        "Implement marginal CDF/PDF transforms and verify full log density as log copula density plus marginal log densities on generated samples.",
        "Add rows with method names such as Vine copula oracle or Fitted vine copula only after density evaluation and failure metadata are audited.",
        "Rerun this script at n_train=1000, n_test=1000, reps=5 and keep the output under outputs/jmlr_frozen_benchmark_v1/external_family/copula_family/.",
    ]
    return {
        "decision": decision,
        "reliable_for_rows": reliable_for_rows,
        "reason": reason,
        "adapter_status": adapter_status,
        "python_candidates": python_candidates,
        "python_available": python_available,
        "r_candidates": ["rvinecopulib", "VineCopula", "copula"],
        "r_status": r_status,
        "r_available": r_available,
        "no_go_plan": plan,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_score(row: dict[str, object], column: str = "heldout_log_score") -> float | None:
    value = str(row.get(column, ""))
    if value in {"", "NA", "nan", "None"}:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if np.isfinite(score) else None


def score_method(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    dimension: int,
    rep: int,
    n_train: int,
    n_test: int,
    method: str,
    seed: int,
    hyperparameter: str,
    fit_seconds: float,
    eval_seconds: float,
    score: float | str,
    status: str = "ok",
    converged: bool | str = True,
    n_iter: int | str = 0,
    final_objective: float | str = "",
    error_type: str = "",
    error_message: str = "",
    nonfinite_count: int | str = 0,
) -> None:
    rows.append(
        {
            "protocol_version": PROTOCOL_VERSION,
            "scenario": scenario,
            "scenario_family": SCENARIO_FAMILY,
            "dimension": dimension,
            "rep": rep,
            "n_train": n_train,
            "n_test": n_test,
            "method": method,
            "heldout_log_score": score,
            "fit_seconds": fit_seconds,
            "eval_seconds": eval_seconds,
            "seed": seed,
            "hyperparameter": hyperparameter,
            "status": status,
            "converged": str(converged).lower() if isinstance(converged, bool) else converged,
            "n_iter": n_iter,
            "final_objective": final_objective,
            "error_type": error_type,
            "error_message": error_message,
            "nonfinite_count": nonfinite_count,
        }
    )


def append_failure_row(
    rows: list[dict[str, object]],
    *,
    scenario: CopulaScenario,
    rep: int,
    seed: int,
    n_train: int,
    n_test: int,
    method: str,
    hyperparameter: str,
    error: Exception,
    fit_seconds: float = 0.0,
    eval_seconds: float = 0.0,
) -> None:
    score_method(
        rows,
        scenario=scenario.name,
        dimension=scenario.dimension,
        rep=rep,
        n_train=n_train,
        n_test=n_test,
        method=method,
        seed=seed,
        hyperparameter=hyperparameter,
        fit_seconds=fit_seconds,
        eval_seconds=eval_seconds,
        score="NA",
        status="failed",
        converged=False,
        n_iter="",
        final_objective="",
        error_type=type(error).__name__,
        error_message=str(error)[:500],
        nonfinite_count="",
    )


def append_baseline_rows(
    rows: list[dict[str, object]],
    train: np.ndarray,
    test: np.ndarray,
    *,
    scenario: CopulaScenario,
    rep: int,
    seed: int,
    args,
    include_gaussian: bool,
    gmm_components: list[int],
    include_kde: bool,
    include_oracle: bool,
) -> None:
    n_train = int(train.shape[0])
    n_test = int(test.shape[0])

    if include_oracle:
        start = time.perf_counter()
        hyperparameter = json.dumps(scenario.parameter_summary, sort_keys=True)
        try:
            logpdf = scenario.logpdf(test)
            eval_seconds = time.perf_counter() - start
            score, nonfinite_count, status = score_logpdf_values(logpdf)
            score_method(
                rows,
                scenario=scenario.name,
                dimension=scenario.dimension,
                rep=rep,
                n_train=n_train,
                n_test=n_test,
                method="Copula oracle",
                seed=seed,
                hyperparameter=hyperparameter,
                fit_seconds=0.0,
                eval_seconds=eval_seconds,
                score=score,
                status=status,
                converged=True,
                n_iter=0,
                nonfinite_count=nonfinite_count,
            )
        except Exception as exc:
            append_failure_row(
                rows,
                scenario=scenario,
                rep=rep,
                seed=seed,
                n_train=n_train,
                n_test=n_test,
                method="Copula oracle",
                hyperparameter=hyperparameter,
                error=exc,
                eval_seconds=time.perf_counter() - start,
            )

    if include_gaussian:
        start = time.perf_counter()
        hyperparameter = f"reg={args.reg:g}"
        try:
            gaussian = fit_gaussian(train, args.reg)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            logpdf = gaussian.logpdf(test)
            eval_seconds = time.perf_counter() - start
            score, nonfinite_count, status = score_logpdf_values(logpdf)
            score_method(
                rows,
                scenario=scenario.name,
                dimension=scenario.dimension,
                rep=rep,
                n_train=n_train,
                n_test=n_test,
                method="Gaussian",
                seed=seed,
                hyperparameter=hyperparameter,
                fit_seconds=fit_seconds,
                eval_seconds=eval_seconds,
                score=score,
                status=status,
                converged=True,
                n_iter=0,
                nonfinite_count=nonfinite_count,
            )
        except Exception as exc:
            append_failure_row(
                rows,
                scenario=scenario,
                rep=rep,
                seed=seed,
                n_train=n_train,
                n_test=n_test,
                method="Gaussian",
                hyperparameter=hyperparameter,
                error=exc,
                fit_seconds=time.perf_counter() - start,
            )

    for k in gmm_components:
        start = time.perf_counter()
        hyperparameter = f"K={k};reg={args.reg:g};max_iter={args.max_iter};tol={args.tol:g}"
        try:
            gmm = fit_gmm_em(
                train,
                k,
                random_state=seed + 7919 + k,
                reg=args.reg,
                max_iter=args.max_iter,
                tol=args.tol,
            )
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            logpdf = gmm.logpdf(test)
            eval_seconds = time.perf_counter() - start
            score, nonfinite_count, status = score_logpdf_values(logpdf)
            if not gmm.converged and status == "ok":
                status = "warning"
            score_method(
                rows,
                scenario=scenario.name,
                dimension=scenario.dimension,
                rep=rep,
                n_train=n_train,
                n_test=n_test,
                method=f"GMM-{min(k, n_train)}",
                seed=seed,
                hyperparameter=hyperparameter,
                fit_seconds=fit_seconds,
                eval_seconds=eval_seconds,
                score=score,
                status=status,
                converged=gmm.converged,
                n_iter=gmm.n_iter,
                final_objective=gmm.final_objective,
                nonfinite_count=nonfinite_count,
            )
        except Exception as exc:
            append_failure_row(
                rows,
                scenario=scenario,
                rep=rep,
                seed=seed,
                n_train=n_train,
                n_test=n_test,
                method=f"GMM-{min(k, n_train)}",
                hyperparameter=hyperparameter,
                error=exc,
                fit_seconds=time.perf_counter() - start,
            )

    if include_kde and scenario.dimension <= args.kde_max_dim:
        start = time.perf_counter()
        hyperparameter = f"bandwidth=scott;jitter={args.kde_jitter:g};dim_gate={args.kde_max_dim}"
        try:
            kde = fit_kde(train, random_state=seed + 104729, jitter=args.kde_jitter)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            logpdf = kde.logpdf(test)
            eval_seconds = time.perf_counter() - start
            score, nonfinite_count, status = score_logpdf_values(logpdf)
            score_method(
                rows,
                scenario=scenario.name,
                dimension=scenario.dimension,
                rep=rep,
                n_train=n_train,
                n_test=n_test,
                method="KDE",
                seed=seed,
                hyperparameter=hyperparameter,
                fit_seconds=fit_seconds,
                eval_seconds=eval_seconds,
                score=score,
                status=status,
                converged=True,
                n_iter=0,
                nonfinite_count=nonfinite_count,
            )
        except Exception as exc:
            append_failure_row(
                rows,
                scenario=scenario,
                rep=rep,
                seed=seed,
                n_train=n_train,
                n_test=n_test,
                method="KDE",
                hyperparameter=hyperparameter,
                error=exc,
                fit_seconds=time.perf_counter() - start,
            )


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for scenario, method in sorted({(str(r["scenario"]), str(r["method"])) for r in rows}):
        sub = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
        scores = np.asarray([score for r in sub if (score := finite_score(r)) is not None], dtype=float)
        fit = np.asarray([float(r["fit_seconds"]) for r in sub], dtype=float)
        eval_time = np.asarray([float(r["eval_seconds"]) for r in sub], dtype=float)
        total = max(len(sub), 1)
        failure_rate = sum(str(r.get("status")) == "failed" for r in sub) / total
        warning_rate = sum(str(r.get("status")) == "warning" for r in sub) / total
        out.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": scenario,
                "scenario_family": SCENARIO_FAMILY,
                "method": method,
                "mean_heldout_log_score": float(np.mean(scores)) if scores.size else "NA",
                "se_heldout_log_score": float(np.std(scores, ddof=1) / np.sqrt(scores.size)) if scores.size > 1 else 0.0,
                "rank_within_scenario": 0,
                "mean_fit_seconds": float(np.mean(fit)),
                "mean_eval_seconds": float(np.mean(eval_time)),
                "reps": int(len(sub)),
                "failure_rate": failure_rate,
                "warning_rate": warning_rate,
            }
        )
    for scenario in sorted({str(r["scenario"]) for r in out}):
        scenario_rows = [r for r in out if r["scenario"] == scenario]
        rankable = [r for r in scenario_rows if finite_score(r, "mean_heldout_log_score") is not None]
        rankable.sort(key=lambda r: float(r["mean_heldout_log_score"]), reverse=True)
        for rank, row in enumerate(rankable, start=1):
            row["rank_within_scenario"] = rank
    return out


def summarize_runtime(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in summary_rows:
        reps = int(row["reps"])
        mean_fit = float(row["mean_fit_seconds"])
        mean_eval = float(row["mean_eval_seconds"])
        out.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": row["scenario"],
                "scenario_family": SCENARIO_FAMILY,
                "method": row["method"],
                "reps": reps,
                "total_fit_seconds": mean_fit * reps,
                "mean_fit_seconds": mean_fit,
                "total_eval_seconds": mean_eval * reps,
                "mean_eval_seconds": mean_eval,
                "peak_memory_mb": "",
            }
        )
    return out


def validate_rows(rows: list[dict[str, object]], split_rows: list[dict[str, object]], sample_rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Held-out score table is empty.")
    missing = REQUIRED_HELDOUT_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"Held-out score table missing columns: {sorted(missing)}")
    bad_status = [r for r in rows if str(r.get("status")) not in {"ok", "warning", "failed"}]
    if bad_status:
        raise ValueError(f"Found {len(bad_status)} rows with unknown status.")
    bad_ok_scores = [r for r in rows if str(r.get("status")) == "ok" and finite_score(r) is None]
    if bad_ok_scores:
        raise ValueError(f"Found {len(bad_ok_scores)} OK rows without finite held-out scores.")
    split_keys = {(str(r["scenario"]), str(r["rep"]), str(r["sample_id"])) for r in split_rows}
    sample_keys = {(str(r["scenario"]), str(r["rep"]), str(r["sample_id"])) for r in sample_rows}
    if split_keys != sample_keys:
        raise ValueError("Sample rows and split assignments do not have matching keys.")
    return {
        "status": "passed",
        "protocol_version": PROTOCOL_VERSION,
        "scenario_family": SCENARIO_FAMILY,
        "heldout_rows": len(rows),
        "split_assignment_rows": len(split_rows),
        "sample_rows": len(sample_rows),
        "scenario_count": len({r["scenario"] for r in rows}),
        "method_count": len({r["method"] for r in rows}),
        "failed_rows": sum(str(r.get("status")) == "failed" for r in rows),
        "warning_rows": sum(str(r.get("status")) == "warning" for r in rows),
    }


def maybe_plot(out_dir: Path, figure_dir: Path, figure_prefix: str, summary_rows: list[dict[str, object]], scenarios: dict[str, CopulaScenario]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    scenario_order = [name for name in SCENARIO_FACTORIES if any(r["scenario"] == name for r in summary_rows)]
    method_order = ["Copula oracle", "Gaussian", "KDE", "GMM-2", "GMM-4", "GMM-8", "GMM-16"]
    fig, axes = plt.subplots(1, len(scenario_order), figsize=(4.25 * len(scenario_order), 3.7), sharey=False)
    if len(scenario_order) == 1:
        axes = [axes]
    color = {
        "Copula oracle": "#5e8c61",
        "Gaussian": "#8fb9d6",
        "KDE": "#d4a85f",
        "GMM-2": "#b59bcf",
        "GMM-4": "#dc8f70",
        "GMM-8": "#9cbf88",
        "GMM-16": "#c7b36d",
    }
    for ax, scenario_name in zip(axes, scenario_order):
        sub = [r for r in summary_rows if r["scenario"] == scenario_name]
        sub.sort(key=lambda r: method_order.index(str(r["method"])) if str(r["method"]) in method_order else 999)
        x = np.arange(len(sub))
        y = [float(r["mean_heldout_log_score"]) for r in sub]
        yerr = [float(r["se_heldout_log_score"]) for r in sub]
        ax.bar(
            x,
            y,
            yerr=yerr,
            color=[color.get(str(r["method"]), "#b8c0cc") for r in sub],
            edgecolor="white",
            linewidth=0.8,
            error_kw={"elinewidth": 0.85, "ecolor": "#2f3a45", "capsize": 2.5},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["method"]) for r in sub], rotation=35, ha="right")
        ax.set_title(scenarios[scenario_name].label, fontweight="bold", fontsize=10)
        ax.set_ylabel("held-out log score")
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    candidate_paths = [
        out_dir / f"{figure_prefix}.png",
        out_dir / f"{figure_prefix}.pdf",
        figure_dir / f"{figure_prefix}.png",
        figure_dir / f"{figure_prefix}.pdf",
    ]
    paths = []
    for path in candidate_paths:
        if path not in paths:
            paths.append(path)
    for path in paths:
        fig.savefig(path, dpi=220 if path.suffix == ".png" else None)
    plt.close(fig)
    return [str(path) for path in paths]


def write_vine_feasibility(args, vine_status: dict[str, object]) -> None:
    (args.output_dir / "vine_copula_feasibility.json").write_text(
        json.dumps(vine_status, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Vine-Copula Feasibility Check",
        "",
        f"- Decision: `{vine_status['decision']}`",
        f"- Reliable for benchmark rows now: `{vine_status['reliable_for_rows']}`",
        f"- Reason: {vine_status['reason']}",
        f"- Python packages available: `{', '.join(vine_status['python_available']) or 'none'}`",
        f"- R probe status: `{vine_status['r_status'].get('status')}`",
        f"- R packages available: `{', '.join(vine_status['r_available']) or 'none'}`",
        "",
        "## No-Go / Dependency Plan",
        "",
    ]
    lines.extend(f"- {item}" for item in vine_status["no_go_plan"])
    lines.extend(
        [
            "",
            "No vine-copula rows were added by this run. This prevents accidental comparison against an unverified or dependency-fragile vine implementation.",
            "",
        ]
    )
    (args.output_dir / "vine_copula_feasibility.md").write_text("\n".join(lines))


def write_parameters(args, scenarios: dict[str, CopulaScenario], vine_status: dict[str, object], figure_paths: list[str]) -> None:
    params = {
        "script": "scripts/run_large_scale_copula_pilot.py",
        "protocol_version": PROTOCOL_VERSION,
        "scenario_family": SCENARIO_FAMILY,
        "evidence_label": "standard_speed_integration",
        "frozen_profile_complete": False,
        "scenarios": args.scenarios,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "reps": args.reps,
        "seed": args.seed,
        "methods": args.methods or "default",
        "gmm_components": args.gmm_components,
        "kde": args.kde,
        "kde_max_dim": args.kde_max_dim,
        "reg": args.reg,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "vine_feasibility": vine_status,
        "scenario_parameters": {name: scenario.parameter_summary for name, scenario in scenarios.items()},
        "figure_paths": figure_paths,
        "claim_boundary": "Standard-speed external copula-family density benchmark; not a fitted SFD comparison.",
    }
    (args.output_dir / "run_parameters.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")

    lines = [
        "Copula-family standard-speed external-family run",
        f"protocol_version={PROTOCOL_VERSION}",
        f"scenario_family={SCENARIO_FAMILY}",
        "evidence_label=standard_speed_integration",
        "frozen_profile_complete=false",
        f"scenarios={','.join(args.scenarios)}",
        f"n_train={args.n_train}",
        f"n_test={args.n_test}",
        f"reps={args.reps}",
        f"seed={args.seed}",
        f"methods={args.methods or 'default'}",
        f"gmm_components={','.join(map(str, args.gmm_components))}",
        f"kde={args.kde}",
        f"kde_max_dim={args.kde_max_dim}",
        f"reg={args.reg}",
        f"max_iter={args.max_iter}",
        f"tol={args.tol}",
        f"vine_decision={vine_status['decision']}",
        f"vine_reason={vine_status['reason']}",
        f"python_vine_available={','.join(vine_status['python_available'])}",
        f"r_package_status={vine_status['r_status'].get('status')}",
        f"r_available_packages={','.join(vine_status['r_available'])}",
        f"output_dir={args.output_dir}",
        f"figure_dir={args.figure_dir}",
        f"figure_prefix={args.figure_prefix}",
        "claim_boundary=Standard-speed external copula-family density benchmark; not a fitted SFD comparison.",
    ]
    for name, scenario in scenarios.items():
        lines.append(f"scenario.{name}.dimension={scenario.dimension}")
        lines.append(f"scenario.{name}.description={scenario.description}")
        lines.append(f"scenario.{name}.parameters={json.dumps(scenario.parameter_summary, sort_keys=True)}")
    (args.output_dir / "run_parameters.txt").write_text("\n".join(lines) + "\n")


def write_report(
    args,
    scenarios: dict[str, CopulaScenario],
    summary_rows: list[dict[str, object]],
    validation: dict[str, object],
    vine_status: dict[str, object],
    figure_paths: list[str],
) -> None:
    best_by_scenario = []
    for scenario in args.scenarios:
        sub = [r for r in summary_rows if r["scenario"] == scenario]
        sub.sort(key=lambda r: float(r["mean_heldout_log_score"]), reverse=True)
        if sub:
            best_by_scenario.append(sub[0])

    lines = [
        "# Copula-Family Standard-Speed Report",
        "",
        "Evidence label: standard-speed external-family integration benchmark.",
        "",
        "## Claim Boundary",
        "",
        (
            "This run compares generic density estimators on simulated copula-family "
            "targets. It does not estimate an SFD model and should not be used to "
            "claim that SFD outperforms copula or vine-copula methods."
        ),
        "",
        "Copula-based distributions are treated as residual-only SFD special cases, consistent with the notation guide.",
        "",
        "## Run",
        "",
        f"- Command-scale parameters: n_train={args.n_train}, n_test={args.n_test}, reps={args.reps}, seed={args.seed}.",
        f"- Protocol version: `{PROTOCOL_VERSION}`.",
        f"- Scenario family: `{SCENARIO_FAMILY}`.",
        f"- Held-out rows: {validation['heldout_rows']}.",
        f"- Sample rows: {validation['sample_rows']}.",
        f"- Split assignment rows: {validation['split_assignment_rows']}.",
        f"- Warning rows: {validation['warning_rows']}; failed rows: {validation['failed_rows']}.",
        f"- Schema validation: {validation['status']}.",
        "",
        "## Scenarios",
        "",
    ]
    for name in args.scenarios:
        scenario = scenarios[name]
        lines.append(f"- `{name}` ({scenario.dimension}d): {scenario.description}")
    lines.extend(
        [
            "",
            "## Vine-Copula Dependency Gate",
            "",
            f"- Decision: `{vine_status['decision']}`.",
            f"- Reliable for rows now: `{vine_status['reliable_for_rows']}`.",
            f"- Reason: {vine_status['reason']}",
            f"- Python packages detected: `{', '.join(vine_status['python_available']) or 'none'}`.",
            f"- R probe status: `{vine_status['r_status'].get('status')}`.",
            f"- R packages detected: `{', '.join(vine_status['r_available']) or 'none'}`.",
            "- Detailed dependency plan: `vine_copula_feasibility.md`.",
            "",
            "## Summary",
            "",
            "| Scenario | Best pilot method | Mean held-out log score | Mean fit seconds | Mean eval seconds |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in best_by_scenario:
        lines.append(
            f"| `{row['scenario']}` | {row['method']} | "
            f"{float(row['mean_heldout_log_score']):.6g} | "
            f"{float(row['mean_fit_seconds']):.4g} | "
            f"{float(row['mean_eval_seconds']):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `heldout_log_scores.csv`",
            f"- `benchmark_summary.csv`",
            f"- `runtime.csv`",
            f"- `samples.csv`",
            f"- `split_assignments.csv`",
            f"- `run_parameters.txt` and `run_parameters.json`",
            f"- `schema_validation.json`",
            f"- `vine_copula_feasibility.md` and `vine_copula_feasibility.json`",
        ]
    )
    for path in figure_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Remaining Risks",
            "",
            "- The run uses the Department SOP standard-speed repetition count; the full paper-facing protocol remains incomplete until fitted-SFD, support-aware baselines, recovery/calibration, and final extraction audits exist.",
            "- Copula-oracle rows use closed-form density evaluators for the generating laws, but they are diagnostic upper bounds rather than fitted estimators.",
            "- No fitted-SFD rows are present, so this output cannot support fitted-SFD superiority claims.",
            "- Vine-copula benchmarking remains pending until optional R dependencies and a verified density path are available.",
            "- KDE is run only for scenarios with dimension at or below `--kde-max-dim`.",
            "",
        ]
    )
    (args.output_dir / "copula_pilot_report.md").write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output-dir", dest="output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--scenarios",
        type=parse_csv_list,
        default=parse_csv_list("gaussian_copula_lognormal_10d,t_copula_t_margins_10d,clayton_copula_gamma_5d"),
    )
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--methods", type=str)
    parser.add_argument("--gmm-components", type=parse_int_list, default=parse_int_list("2,4,8,16"))
    parser.add_argument("--kde", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kde-max-dim", type=int, default=5)
    parser.add_argument("--kde-jitter", type=float, default=1e-6)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--r-timeout", type=float, default=6.0)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--figure-prefix", type=str, default=DEFAULT_FIGURE_PREFIX)
    return parser.parse_args()


def normalize_paths(args) -> None:
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.figure_dir is None:
        args.figure_dir = args.output_dir
    else:
        args.figure_dir = args.figure_dir if args.figure_dir.is_absolute() else ROOT / args.figure_dir


def main() -> None:
    args = parse_args()
    normalize_paths(args)
    unknown = sorted(set(args.scenarios) - set(SCENARIO_FACTORIES))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")
    if args.n_train <= 0 or args.n_test <= 0 or args.reps <= 0:
        raise ValueError("--n-train, --n-test, and --reps must be positive.")
    if args.kde_max_dim < 1:
        raise ValueError("--kde-max-dim must be at least 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {name: SCENARIO_FACTORIES[name]() for name in args.scenarios}
    vine_status = inspect_vine_dependencies(args.r_timeout)
    write_vine_feasibility(args, vine_status)
    include_gaussian, gmm_components, include_kde, include_oracle = parse_methods(args.methods)
    if args.methods is None:
        gmm_components = args.gmm_components

    heldout_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    sample_fieldnames = ["scenario", "rep", "seed", "sample_id", "split"]
    max_dim = max(scenario.dimension for scenario in scenarios.values())
    sample_fieldnames.extend([f"x{j + 1}" for j in range(max_dim)])

    for rep in range(1, args.reps + 1):
        for scenario_idx, scenario_name in enumerate(args.scenarios):
            scenario = scenarios[scenario_name]
            seed = args.seed + 10000 * rep + 101 * scenario_idx
            rng = np.random.default_rng(seed)
            print(
                f"[copula-pilot] scenario={scenario.name} rep={rep} seed={seed} "
                f"n_train={args.n_train} n_test={args.n_test}",
                flush=True,
            )
            train = scenario.sampler(rng, args.n_train)
            test = scenario.sampler(rng, args.n_test)

            for split, data in [("train", train), ("test", test)]:
                for i, row in enumerate(data):
                    sample_id = f"{scenario.name}_rep{rep:02d}_{split}_{i:06d}"
                    split_rows.append(
                        {
                            "scenario": scenario.name,
                            "rep": rep,
                            "sample_id": sample_id,
                            "split": split,
                            "split_seed": seed,
                            "split_rule": "independent_generated_train_test",
                        }
                    )
                    sample_row = {
                        "scenario": scenario.name,
                        "rep": rep,
                        "seed": seed,
                        "sample_id": sample_id,
                        "split": split,
                    }
                    for j in range(max_dim):
                        sample_row[f"x{j + 1}"] = float(row[j]) if j < scenario.dimension else ""
                    sample_rows.append(sample_row)

            append_baseline_rows(
                heldout_rows,
                train,
                test,
                scenario=scenario,
                rep=rep,
                seed=seed,
                args=args,
                include_gaussian=include_gaussian,
                gmm_components=gmm_components,
                include_kde=include_kde,
                include_oracle=include_oracle,
            )

    summary_rows = summarize(heldout_rows)
    runtime_rows = summarize_runtime(summary_rows)
    validation = validate_rows(heldout_rows, split_rows, sample_rows)
    figure_paths = maybe_plot(args.output_dir, args.figure_dir, args.figure_prefix, summary_rows, scenarios)

    write_csv(
        args.output_dir / "heldout_log_scores.csv",
        heldout_rows,
        SCORE_FIELDNAMES,
    )
    write_csv(
        args.output_dir / "benchmark_summary.csv",
        summary_rows,
        SUMMARY_FIELDNAMES,
    )
    write_csv(
        args.output_dir / "runtime.csv",
        runtime_rows,
        RUNTIME_FIELDNAMES,
    )
    write_csv(args.output_dir / "samples.csv", sample_rows, sample_fieldnames)
    write_csv(
        args.output_dir / "split_assignments.csv",
        split_rows,
        ["scenario", "rep", "sample_id", "split", "split_seed", "split_rule"],
    )
    (args.output_dir / "schema_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    write_parameters(args, scenarios, vine_status, figure_paths)
    write_report(args, scenarios, summary_rows, validation, vine_status, figure_paths)

    print(f"Wrote {args.output_dir / 'heldout_log_scores.csv'}")
    print(f"Wrote {args.output_dir / 'benchmark_summary.csv'}")
    print(f"Wrote {args.output_dir / 'runtime.csv'}")
    print(f"Wrote {args.output_dir / 'copula_pilot_report.md'}")


if __name__ == "__main__":
    main()
