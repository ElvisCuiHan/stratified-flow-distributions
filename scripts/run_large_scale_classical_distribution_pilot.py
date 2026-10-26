#!/usr/bin/env python3
"""Run a pilot benchmark on ordinary full-dimensional distributions.

The output follows the JMLR held-out density benchmark schema while keeping a
separate generated-sample CSV and deterministic split assignment file. These
scenarios are external-family density benchmarks, not evidence for a fitted SFD
method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
from scipy.stats import gaussian_kde, multivariate_normal


PROTOCOL_VERSION = "jmlr_frozen_benchmark_v1"
SCENARIO_FAMILY = "external_classical_full_dimensional"
DEFAULT_OUT = ROOT / "outputs" / "jmlr_frozen_benchmark_v1" / "external_family" / "classical_distribution"
DEFAULT_REPORT = DEFAULT_OUT / "execution_report.md"
DEFAULT_FIGURE_PREFIX = "jmlr_large_scale_classical_heldout_log_score"
FEATURE_COLUMNS = [f"x{idx:02d}" for idx in range(1, 21)]
SCORE_FIELDNAMES = [
    "protocol_version",
    "scenario",
    "scenario_family",
    "dim",
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
REQUIRED_SCORE_COLUMNS = {
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
DEFAULT_SCENARIOS = [
    "mvnormal_correlated_10d",
    "student_t_10d",
    "skew_mixture_10d",
    "gaussian_mixture_20d",
    "banana_2d",
]


def parse_csv_list(text: str) -> list[str]:
    out = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return out


def parse_int_list(text: str) -> list[int]:
    out = [int(piece) for piece in text.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if any(value <= 0 for value in out):
        raise argparse.ArgumentTypeError("All component counts must be positive.")
    return sorted(set(out))


def toeplitz_cov(d: int, rho: float, scale: float = 1.0) -> np.ndarray:
    idx = np.arange(d)
    return scale * rho ** np.abs(idx[:, None] - idx[None, :])


def student_t_logpdf(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, df: float) -> np.ndarray:
    centered = x - mean
    sign, logdet = np.linalg.slogdet(scale)
    if sign <= 0:
        raise ValueError("Student-t scale matrix must be positive definite.")
    solve = np.linalg.solve(scale, centered.T).T
    q = np.sum(centered * solve, axis=1)
    d = mean.size
    norm = gammaln((df + d) / 2.0) - gammaln(df / 2.0) - 0.5 * (d * math.log(df * math.pi) + logdet)
    return norm - 0.5 * (df + d) * np.log1p(q / df)


def mixture_logpdf(x: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    cols = []
    for weight, mean, cov in zip(weights, means, covs):
        cols.append(np.log(max(float(weight), 1e-300)) + multivariate_normal.logpdf(x, mean=mean, cov=cov))
    return logsumexp(np.vstack(cols).T, axis=1)


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    dim: int
    description: str
    sample: Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]
    logpdf: Callable[[np.ndarray], np.ndarray]


def make_mvnormal_correlated_10d() -> Scenario:
    d = 10
    mean = np.linspace(-0.7, 0.7, d)
    cov = toeplitz_cov(d, rho=0.62, scale=1.0)

    def sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        return rng.multivariate_normal(mean, cov, size=n), np.full(n, -1, dtype=int)

    def logpdf(x: np.ndarray) -> np.ndarray:
        return np.asarray(multivariate_normal.logpdf(x, mean=mean, cov=cov), dtype=float)

    return Scenario(
        name="mvnormal_correlated_10d",
        label="Correlated Gaussian 10D",
        dim=d,
        description="Ten-dimensional Gaussian with Toeplitz covariance rho=0.62.",
        sample=sample,
        logpdf=logpdf,
    )


def make_student_t_10d() -> Scenario:
    d = 10
    df = 4.0
    mean = np.zeros(d)
    scale = toeplitz_cov(d, rho=0.48, scale=1.15)
    chol = np.linalg.cholesky(scale)

    def sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        z = rng.standard_normal(size=(n, d)) @ chol.T
        u = rng.chisquare(df, size=n) / df
        return mean + z / np.sqrt(u)[:, None], np.full(n, -1, dtype=int)

    def logpdf(x: np.ndarray) -> np.ndarray:
        return np.asarray(student_t_logpdf(x, mean=mean, scale=scale, df=df), dtype=float)

    return Scenario(
        name="student_t_10d",
        label="Student-t 10D",
        dim=d,
        description="Ten-dimensional Student-t distribution with df=4 and correlated scale.",
        sample=sample,
        logpdf=logpdf,
    )


def make_skew_mixture_10d() -> Scenario:
    d = 10
    weights = np.array([0.70, 0.22, 0.08])
    means = np.vstack(
        [
            np.linspace(-1.1, 0.6, d),
            np.linspace(1.6, -0.8, d),
            np.r_[np.full(5, 2.0), np.full(5, -1.7)],
        ]
    )
    covs = np.stack(
        [
            toeplitz_cov(d, rho=0.35, scale=0.55),
            toeplitz_cov(d, rho=0.58, scale=0.35),
            np.diag(np.linspace(0.18, 0.90, d)),
        ]
    )

    def sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        comp = rng.choice(weights.size, size=n, p=weights)
        out = np.empty((n, d), dtype=float)
        for idx in range(weights.size):
            mask = comp == idx
            if np.any(mask):
                out[mask] = rng.multivariate_normal(means[idx], covs[idx], size=int(mask.sum()))
        return out, comp.astype(int)

    def logpdf(x: np.ndarray) -> np.ndarray:
        return mixture_logpdf(x, weights, means, covs)

    return Scenario(
        name="skew_mixture_10d",
        label="Asymmetric mixture 10D",
        dim=d,
        description="Asymmetric three-component Gaussian mixture with unequal weights and covariances.",
        sample=sample,
        logpdf=logpdf,
    )


def make_gaussian_mixture_20d() -> Scenario:
    d = 20
    weights = np.array([0.35, 0.30, 0.22, 0.13])
    means = np.vstack(
        [
            np.zeros(d),
            np.r_[np.full(10, 1.3), np.full(10, -0.6)],
            np.linspace(-1.8, 1.8, d),
            np.r_[np.full(5, -2.1), np.full(5, 0.8), np.full(10, 1.5)],
        ]
    )
    covs = np.stack(
        [
            toeplitz_cov(d, rho=0.25, scale=0.45),
            toeplitz_cov(d, rho=0.50, scale=0.35),
            np.diag(np.linspace(0.22, 0.75, d)),
            toeplitz_cov(d, rho=0.15, scale=0.60),
        ]
    )

    def sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        comp = rng.choice(weights.size, size=n, p=weights)
        out = np.empty((n, d), dtype=float)
        for idx in range(weights.size):
            mask = comp == idx
            if np.any(mask):
                out[mask] = rng.multivariate_normal(means[idx], covs[idx], size=int(mask.sum()))
        return out, comp.astype(int)

    def logpdf(x: np.ndarray) -> np.ndarray:
        return mixture_logpdf(x, weights, means, covs)

    return Scenario(
        name="gaussian_mixture_20d",
        label="Gaussian mixture 20D",
        dim=d,
        description="Twenty-dimensional four-component Gaussian mixture.",
        sample=sample,
        logpdf=logpdf,
    )


def make_banana_2d() -> Scenario:
    d = 2
    bend = 0.18
    base_mean = np.zeros(d)
    base_cov = np.array([[1.0, 0.45], [0.45, 0.85]])

    def sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        z = rng.multivariate_normal(base_mean, base_cov, size=n)
        x = z.copy()
        x[:, 1] = z[:, 1] + bend * (z[:, 0] ** 2 - base_cov[0, 0])
        return x, np.full(n, -1, dtype=int)

    def logpdf(x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=float).copy()
        z[:, 1] = x[:, 1] - bend * (x[:, 0] ** 2 - base_cov[0, 0])
        return np.asarray(multivariate_normal.logpdf(z, mean=base_mean, cov=base_cov), dtype=float)

    return Scenario(
        name="banana_2d",
        label="Banana 2D",
        dim=d,
        description="Two-dimensional banana transform of a correlated Gaussian; Jacobian determinant is one.",
        sample=sample,
        logpdf=logpdf,
    )


SCENARIOS = {
    scenario.name: scenario
    for scenario in [
        make_mvnormal_correlated_10d(),
        make_student_t_10d(),
        make_skew_mixture_10d(),
        make_gaussian_mixture_20d(),
        make_banana_2d(),
    ]
}


@dataclass
class GaussianFit:
    mean: np.ndarray
    cov: np.ndarray

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov), dtype=float)


@dataclass
class GMMFit:
    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    converged: bool
    n_iter: int
    final_objective: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return mixture_logpdf(x, self.weights, self.means, self.covs)


@dataclass
class KDEFit:
    kde: gaussian_kde

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.kde.logpdf(x.T), dtype=float)


def regularized_cov(x: np.ndarray, reg: float) -> np.ndarray:
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return np.asarray(cov, dtype=float) + reg * np.eye(x.shape[1])


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
    base_cov = regularized_cov(x, reg)
    covs = np.repeat(base_cov[None, :, :], k, axis=0)
    weights = np.full(k, 1.0 / k)
    previous = -np.inf
    converged = False
    objective = -np.inf
    iteration = 0

    for iteration in range(1, max_iter + 1):
        log_cols = []
        for weight, mean, cov in zip(weights, means, covs):
            log_cols.append(np.log(max(float(weight), 1e-300)) + multivariate_normal.logpdf(x, mean=mean, cov=cov))
        log_resp = np.vstack(log_cols).T
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
    finite = np.asarray(logpdf, dtype=float)
    finite = finite[np.isfinite(finite)]
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


def generate_split(
    scenario: Scenario,
    *,
    rep: int,
    seed: int,
    n_train: int,
    n_test: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, train_component = scenario.sample(rng, n_train)
    test, test_component = scenario.sample(rng, n_test)
    return train, test, train_component, test_component


def append_sample_rows(
    sample_rows: list[dict[str, object]],
    split_rows: list[dict[str, object]],
    *,
    scenario: str,
    rep: int,
    seed: int,
    split: str,
    x: np.ndarray,
    components: np.ndarray,
) -> None:
    for local_idx, values in enumerate(x, start=1):
        sample_id = f"{scenario}_rep{rep:02d}_{split}_{local_idx:06d}"
        row: dict[str, object] = {
            "scenario": scenario,
            "rep": rep,
            "sample_id": sample_id,
            "split": split,
            "seed": seed,
            "true_component": int(components[local_idx - 1]),
        }
        for idx, column in enumerate(FEATURE_COLUMNS):
            row[column] = f"{values[idx]:.12g}" if idx < values.size else ""
        sample_rows.append(row)
        split_rows.append(
            {
                "scenario": scenario,
                "rep": rep,
                "sample_id": sample_id,
                "split": split,
                "seed": seed,
                "split_rule": "independent_rng_blocks_train_then_test",
            }
        )


def append_method_row(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    method: str,
    seed: int,
    fit_seconds: float,
    eval_seconds: float,
    score: float | str,
    hyperparameter: str,
    dim: int,
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
            "dim": dim,
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
    scenario: Scenario,
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
    append_method_row(
        rows,
        scenario=scenario.name,
        rep=rep,
        n_train=n_train,
        n_test=n_test,
        method=method,
        seed=seed,
        fit_seconds=fit_seconds,
        eval_seconds=eval_seconds,
        score="NA",
        hyperparameter=hyperparameter,
        dim=scenario.dim,
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
    *,
    scenario: Scenario,
    rep: int,
    seed: int,
    train: np.ndarray,
    test: np.ndarray,
    args,
) -> None:
    n_train = int(train.shape[0])
    n_test = int(test.shape[0])

    if args.oracle:
        start = time.perf_counter()
        try:
            oracle_logpdf = scenario.logpdf(test)
            eval_seconds = time.perf_counter() - start
            score, nonfinite_count, status = score_logpdf_values(oracle_logpdf)
            append_method_row(
                rows,
                scenario=scenario.name,
                rep=rep,
                n_train=n_train,
                n_test=n_test,
                method="Oracle DGP",
                seed=seed,
                fit_seconds=0.0,
                eval_seconds=eval_seconds,
                score=score,
                hyperparameter="generating_density=implemented",
                dim=scenario.dim,
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
                method="Oracle DGP",
                hyperparameter="generating_density=implemented",
                error=exc,
                eval_seconds=time.perf_counter() - start,
            )

    start = time.perf_counter()
    try:
        gaussian = fit_gaussian(train, args.reg)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        logpdf = gaussian.logpdf(test)
        eval_seconds = time.perf_counter() - start
        score, nonfinite_count, status = score_logpdf_values(logpdf)
        append_method_row(
            rows,
            scenario=scenario.name,
            rep=rep,
            n_train=n_train,
            n_test=n_test,
            method="Gaussian",
            seed=seed,
            fit_seconds=fit_seconds,
            eval_seconds=eval_seconds,
            score=score,
            hyperparameter=f"reg={args.reg:g}",
            dim=scenario.dim,
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
            hyperparameter=f"reg={args.reg:g}",
            error=exc,
            fit_seconds=time.perf_counter() - start,
        )

    for k in args.gmm_components:
        hyperparameter = f"K={k};reg={args.reg:g};max_iter={args.max_iter};tol={args.tol:g}"
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
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            logpdf = gmm.logpdf(test)
            eval_seconds = time.perf_counter() - start
            score, nonfinite_count, status = score_logpdf_values(logpdf)
            if not gmm.converged and status == "ok":
                status = "warning"
            append_method_row(
                rows,
                scenario=scenario.name,
                rep=rep,
                n_train=n_train,
                n_test=n_test,
                method=f"GMM-{min(k, n_train)}",
                seed=seed,
                fit_seconds=fit_seconds,
                eval_seconds=eval_seconds,
                score=score,
                hyperparameter=hyperparameter,
                dim=scenario.dim,
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

    if args.kde and (scenario.dim <= args.kde_max_dim or args.force_kde):
        hyperparameter = f"bandwidth=scott;jitter={args.kde_jitter:g};dim_gate={args.kde_max_dim}"
        start = time.perf_counter()
        try:
            kde = fit_kde(train, random_state=seed + 7919, jitter=args.kde_jitter)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            logpdf = kde.logpdf(test)
            eval_seconds = time.perf_counter() - start
            score, nonfinite_count, status = score_logpdf_values(logpdf)
            append_method_row(
                rows,
                scenario=scenario.name,
                rep=rep,
                n_train=n_train,
                n_test=n_test,
                method="KDE",
                seed=seed,
                fit_seconds=fit_seconds,
                eval_seconds=eval_seconds,
                score=score,
                hyperparameter=hyperparameter,
                dim=scenario.dim,
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


def validate_oracle_densities(scenarios: list[Scenario], seed: int) -> list[str]:
    messages = []
    rng = np.random.default_rng(seed + 4242)
    for scenario in scenarios:
        x, _ = scenario.sample(rng, 32)
        values = np.asarray(scenario.logpdf(x), dtype=float)
        if values.shape != (32,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Oracle DGP log density failed validation for {scenario.name}.")
        messages.append(f"{scenario.name}: oracle log density finite on 32 generated samples")
    return messages


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    names = fieldnames or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
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


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(str(row["scenario"]), str(row["method"])) for row in rows})
    out: list[dict[str, object]] = []
    for scenario, method in keys:
        sub = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
        scores = np.asarray([score for row in sub if (score := finite_score(row)) is not None], dtype=float)
        fit = np.asarray([float(row["fit_seconds"]) for row in sub], dtype=float)
        eval_time = np.asarray([float(row["eval_seconds"]) for row in sub], dtype=float)
        total = max(len(sub), 1)
        failure_rate = sum(str(row.get("status")) == "failed" for row in sub) / total
        warning_rate = sum(str(row.get("status")) == "warning" for row in sub) / total
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
    for scenario in sorted({str(row["scenario"]) for row in out}):
        scenario_rows = [row for row in out if row["scenario"] == scenario]
        rankable = [row for row in scenario_rows if finite_score(row, "mean_heldout_log_score") is not None]
        rankable.sort(key=lambda row: float(row["mean_heldout_log_score"]), reverse=True)
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


def count_csv_rows(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def validate_outputs(out_dir: Path) -> list[str]:
    score_path = out_dir / "heldout_log_scores.csv"
    summary_path = out_dir / "benchmark_summary.csv"
    runtime_path = out_dir / "runtime.csv"
    sample_path = out_dir / "classical_distribution_samples.csv"
    split_path = out_dir / "split_assignments.csv"
    for path in [score_path, summary_path, runtime_path, sample_path, split_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    with score_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_SCORE_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{score_path} missing required score columns: {sorted(missing)}")
        score_rows = list(reader)
    if not score_rows:
        raise ValueError(f"{score_path} has no data rows.")
    for row in score_rows:
        score = finite_score(row)
        fit_seconds = float(row["fit_seconds"])
        eval_seconds = float(row["eval_seconds"])
        if str(row["status"]) not in {"ok", "warning", "failed"}:
            raise ValueError(f"Unknown status in {score_path}: {row}")
        if score is None and str(row["status"]) == "ok":
            raise ValueError(f"OK row has non-finite heldout_log_score in {score_path}: {row}")
        if fit_seconds < 0.0 or eval_seconds < 0.0:
            raise ValueError(f"Negative runtime in {score_path}: {row}")

    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f)
        summary_rows = list(reader)
        required = set(SUMMARY_FIELDNAMES)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{summary_path} missing columns: {sorted(missing)}")

    with runtime_path.open(newline="") as f:
        reader = csv.DictReader(f)
        runtime_rows = list(reader)
        required = set(RUNTIME_FIELDNAMES)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{runtime_path} missing columns: {sorted(missing)}")

    with split_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {"scenario", "sample_id", "split"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{split_path} missing split columns: {sorted(missing)}")
        split_rows = list(reader)
    if len(split_rows) != count_csv_rows(sample_path):
        raise ValueError("Sample and split assignment row counts differ.")

    messages = [
        f"heldout_log_scores.csv rows={len(score_rows)}",
        f"benchmark_summary.csv rows={len(summary_rows)}",
        f"runtime.csv rows={len(runtime_rows)}",
        f"classical_distribution_samples.csv rows={count_csv_rows(sample_path)}",
        f"split_assignments.csv rows={len(split_rows)}",
        "schema_validation=passed",
    ]
    (out_dir / "schema_validation.txt").write_text("\n".join(messages) + "\n")
    return messages


def write_parameters(args, scenarios: list[Scenario], oracle_messages: list[str], elapsed_seconds: float) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    relative_scenarios = ",".join(scenario.name for scenario in scenarios)
    lines = [
        "Classical distribution standard-speed external-family run for JMLR SFD benchmark",
        f"protocol_version={PROTOCOL_VERSION}",
        f"scenario_family={SCENARIO_FAMILY}",
        "evidence_label=standard_speed_integration",
        "frozen_profile_complete=false",
        "benchmark_boundary=generic full-dimensional external-family density benchmark; not a fitted SFD comparison",
        f"scenarios={relative_scenarios}",
        f"reps={args.reps}",
        f"seed={args.seed}",
        f"n_train={args.n_train}",
        f"n_test={args.n_test}",
        f"gmm_components={','.join(map(str, args.gmm_components))}",
        f"kde={args.kde}",
        f"kde_max_dim={args.kde_max_dim}",
        f"force_kde={args.force_kde}",
        f"oracle={args.oracle}",
        f"reg={args.reg}",
        f"max_iter={args.max_iter}",
        f"tol={args.tol}",
        f"kde_jitter={args.kde_jitter}",
        f"output_dir={args.output_dir}",
        f"report={args.report}",
        f"figure_dir={args.figure_dir}",
        f"figure_prefix={args.figure_prefix}",
        f"elapsed_seconds={elapsed_seconds:.6f}",
        "sample_csv=classical_distribution_samples.csv",
        "split_assignments=split_assignments.csv",
    ]
    for scenario in scenarios:
        lines.append(f"scenario.{scenario.name}.dim={scenario.dim}")
        lines.append(f"scenario.{scenario.name}.description={scenario.description}")
    for message in oracle_messages:
        lines.append(f"oracle_validation={message}")
    (args.output_dir / "run_parameters.txt").write_text("\n".join(lines) + "\n")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario_family": SCENARIO_FAMILY,
        "evidence_label": "standard_speed_integration",
        "frozen_profile_complete": False,
        "benchmark_boundary": "generic full-dimensional external-family density benchmark; not a fitted SFD comparison",
        "scenarios": [
            {"name": scenario.name, "label": scenario.label, "dim": scenario.dim, "description": scenario.description}
            for scenario in scenarios
        ],
        "reps": args.reps,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "gmm_components": args.gmm_components,
        "kde": args.kde,
        "kde_max_dim": args.kde_max_dim,
        "force_kde": args.force_kde,
        "oracle": args.oracle,
        "reg": args.reg,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "kde_jitter": args.kde_jitter,
        "output_dir": str(args.output_dir.relative_to(ROOT) if args.output_dir.is_relative_to(ROOT) else args.output_dir),
        "report": str(args.report.relative_to(ROOT) if args.report.is_relative_to(ROOT) else args.report),
        "figure_dir": str(args.figure_dir.relative_to(ROOT) if args.figure_dir.is_relative_to(ROOT) else args.figure_dir),
        "figure_prefix": args.figure_prefix,
        "elapsed_seconds": elapsed_seconds,
        "oracle_validation": oracle_messages,
    }
    (args.output_dir / "run_parameters.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def maybe_plot(args, scenarios: list[Scenario], summary_rows: list[dict[str, object]]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    scenario_order = [scenario.name for scenario in scenarios]
    label_map = {scenario.name: scenario.label for scenario in scenarios}
    method_order = ["Oracle DGP", "Gaussian", "KDE", "GMM-2", "GMM-4", "GMM-8", "GMM-16"]
    color = {
        "Oracle DGP": "#3f6b57",
        "Gaussian": "#8ecae6",
        "KDE": "#d9b44a",
        "GMM-2": "#b7a4df",
        "GMM-4": "#79b58c",
        "GMM-8": "#f28e88",
        "GMM-16": "#8f9aa8",
    }
    fig, axes = plt.subplots(1, len(scenario_order), figsize=(3.7 * len(scenario_order), 3.6), sharey=False)
    if len(scenario_order) == 1:
        axes = [axes]
    for ax, scenario_name in zip(axes, scenario_order):
        sub = [row for row in summary_rows if row["scenario"] == scenario_name]
        sub.sort(key=lambda row: method_order.index(str(row["method"])) if str(row["method"]) in method_order else 999)
        x = np.arange(len(sub))
        y = [float(row["mean_heldout_log_score"]) for row in sub]
        yerr = [float(row["se_heldout_log_score"]) for row in sub]
        ax.bar(
            x,
            y,
            yerr=yerr,
            color=[color.get(str(row["method"]), "#c2c7cf") for row in sub],
            edgecolor="white",
            linewidth=0.8,
            error_kw={"elinewidth": 0.8, "ecolor": "#27313a", "capsize": 2.5},
        )
        ax.set_title(label_map.get(scenario_name, scenario_name), fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(row["method"]) for row in sub], rotation=35, ha="right", fontsize=8.0)
        ax.set_ylabel("held-out log score")
        ax.grid(axis="y", alpha=0.24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    candidate_paths = [
        args.output_dir / f"{args.figure_prefix}.png",
        args.output_dir / f"{args.figure_prefix}.pdf",
        args.figure_dir / f"{args.figure_prefix}.png",
        args.figure_dir / f"{args.figure_prefix}.pdf",
    ]
    paths = []
    for path in candidate_paths:
        if path not in paths:
            paths.append(path)
    for path in paths:
        fig.savefig(path, dpi=220 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def write_report(
    args,
    *,
    scenarios: list[Scenario],
    summary_rows: list[dict[str, object]],
    validation_messages: list[str],
    figure_paths: list[Path],
    elapsed_seconds: float,
) -> None:
    score_rows = count_csv_rows(args.output_dir / "heldout_log_scores.csv")
    summary_count = count_csv_rows(args.output_dir / "benchmark_summary.csv")
    runtime_count = count_csv_rows(args.output_dir / "runtime.csv")
    sample_count = count_csv_rows(args.output_dir / "classical_distribution_samples.csv")
    split_count = count_csv_rows(args.output_dir / "split_assignments.csv")
    ranks = [
        f"| {row['scenario']} | {row['method']} | {float(row['mean_heldout_log_score']):.4f} | "
        f"{float(row['se_heldout_log_score']):.4f} | {row['rank_within_scenario']} | "
        f"{float(row['mean_fit_seconds']):.4f} | {float(row['mean_eval_seconds']):.4f} |"
        for row in sorted(summary_rows, key=lambda row: (str(row["scenario"]), int(row["rank_within_scenario"])))
    ]
    scenario_lines = [
        f"- `{scenario.name}` ({scenario.dim}D): {scenario.description}" for scenario in scenarios
    ]
    figure_lines = [f"- `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}`" for path in figure_paths]
    command = (
        "PYTHONPATH=src python3 scripts/run_large_scale_classical_distribution_pilot.py "
        f"--out {args.output_dir.relative_to(ROOT) if args.output_dir.is_relative_to(ROOT) else args.output_dir} "
        f"--n-train {args.n_train} --n-test {args.n_test} --reps {args.reps} "
        f"--seed {args.seed} --gmm-components {','.join(map(str, args.gmm_components))}"
    )
    text = [
        "# Classical Distribution Standard-Speed Report",
        "",
        "Evidence label: standard-speed external-family integration benchmark.",
        "",
        "This run evaluates generic density baselines on ordinary full-dimensional distributions. "
        "It does not fit an SFD estimator and should not be cited as evidence that SFD dominates "
        "classical full-dimensional targets.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Scenarios",
        "",
        *scenario_lines,
        "",
        "## Run Parameters",
        "",
        f"- train/test sizes per scenario and repetition: `{args.n_train}` / `{args.n_test}`",
        f"- repetitions: `{args.reps}`",
        f"- seed: `{args.seed}`",
        f"- GMM component grid: `{','.join(map(str, args.gmm_components))}`",
        f"- KDE gate: dimensions `<= {args.kde_max_dim}` unless force-enabled",
        f"- oracle DGP rows: `{args.oracle}` with finite-density validation",
        f"- elapsed wall time inside script: `{elapsed_seconds:.3f}` seconds",
        "",
        "## Schema And Row Counts",
        "",
        f"- `classical_distribution_samples.csv`: {sample_count} rows",
        f"- `split_assignments.csv`: {split_count} rows",
        f"- `heldout_log_scores.csv`: {score_rows} rows",
        f"- `benchmark_summary.csv`: {summary_count} rows",
        f"- `runtime.csv`: {runtime_count} rows",
        f"- validation: `{'; '.join(validation_messages)}`",
        "",
        "Per-run, summary, and runtime columns follow the frozen JMLR external-family schema, "
        "including protocol version, scenario family, status, convergence, failure, and warning fields.",
        "",
        "## Summary Ranks",
        "",
        "| scenario | method | mean held-out log score | SE | rank | mean fit s | mean eval s |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *ranks,
        "",
        "## Figures",
        "",
        *(figure_lines or ["- Figure generation skipped because matplotlib was unavailable."]),
        "",
        "## Claim Boundary And Remaining Risk",
        "",
        "- These are generic full-dimensional benchmark distributions, not mixed-support SFD stress tests.",
        "- Oracle rows are DGP log-density upper-bound diagnostics, not fitted estimators.",
        "- No fitted-SFD rows are present, so this output cannot support fitted-SFD superiority claims.",
        "- The run uses the Department SOP standard-speed repetition count; the full paper-facing protocol remains incomplete until fitted-SFD, support-aware baselines, recovery/calibration, and final extraction audits exist.",
        "- KDE is included only for low-dimensional scenarios by default.",
        "- The custom GMM-EM implementation is dependency-light and reproducible, but not a tuned state-of-the-art mixture baseline.",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(text))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output-dir", dest="output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--figure-prefix", type=str, default=DEFAULT_FIGURE_PREFIX)
    parser.add_argument("--scenarios", type=parse_csv_list, default=DEFAULT_SCENARIOS)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--gmm-components", type=parse_int_list, default=parse_int_list("2,4,8,16"))
    parser.add_argument("--kde", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kde-max-dim", type=int, default=3)
    parser.add_argument("--force-kde", action="store_true")
    parser.add_argument("--oracle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--kde-jitter", type=float, default=1e-6)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def normalize_paths(args) -> None:
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.report = args.report if args.report.is_absolute() else ROOT / args.report
    if args.figure_dir is None:
        args.figure_dir = args.output_dir
    else:
        args.figure_dir = args.figure_dir if args.figure_dir.is_absolute() else ROOT / args.figure_dir


def main() -> None:
    args = parse_args()
    normalize_paths(args)
    if args.validate_only:
        messages = validate_outputs(args.output_dir)
        print("\n".join(messages))
        return

    unknown = sorted(set(args.scenarios) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")
    if args.n_train <= 0 or args.n_test <= 0 or args.reps <= 0:
        raise ValueError("--n-train, --n-test, and --reps must be positive.")
    if args.kde_max_dim < 1:
        raise ValueError("--kde-max-dim must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [SCENARIOS[name] for name in args.scenarios]
    oracle_messages = validate_oracle_densities(scenarios, args.seed) if args.oracle else []
    start_all = time.perf_counter()
    score_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for rep in range(1, args.reps + 1):
        for idx, scenario in enumerate(scenarios):
            seed = args.seed + 10000 * rep + 101 * idx
            print(
                f"[classical-pilot] scenario={scenario.name} rep={rep} seed={seed} "
                f"n_train={args.n_train} n_test={args.n_test}",
                flush=True,
            )
            train, test, train_component, test_component = generate_split(
                scenario,
                rep=rep,
                seed=seed,
                n_train=args.n_train,
                n_test=args.n_test,
            )
            append_sample_rows(
                sample_rows,
                split_rows,
                scenario=scenario.name,
                rep=rep,
                seed=seed,
                split="train",
                x=train,
                components=train_component,
            )
            append_sample_rows(
                sample_rows,
                split_rows,
                scenario=scenario.name,
                rep=rep,
                seed=seed,
                split="test",
                x=test,
                components=test_component,
            )
            append_baseline_rows(
                score_rows,
                scenario=scenario,
                rep=rep,
                seed=seed,
                train=train,
                test=test,
                args=args,
            )

    summary_rows = summarize(score_rows)
    runtime_rows = summarize_runtime(summary_rows)
    elapsed_seconds = time.perf_counter() - start_all

    write_csv(args.output_dir / "heldout_log_scores.csv", score_rows, SCORE_FIELDNAMES)
    write_csv(args.output_dir / "benchmark_summary.csv", summary_rows, SUMMARY_FIELDNAMES)
    write_csv(args.output_dir / "runtime.csv", runtime_rows, RUNTIME_FIELDNAMES)
    write_csv(
        args.output_dir / "classical_distribution_samples.csv",
        sample_rows,
        fieldnames=["scenario", "rep", "sample_id", "split", "seed", "true_component", *FEATURE_COLUMNS],
    )
    write_csv(
        args.output_dir / "split_assignments.csv",
        split_rows,
        fieldnames=["scenario", "rep", "sample_id", "split", "seed", "split_rule"],
    )
    write_parameters(args, scenarios, oracle_messages, elapsed_seconds)
    validation_messages = validate_outputs(args.output_dir)
    figure_paths = maybe_plot(args, scenarios, summary_rows)
    write_report(
        args,
        scenarios=scenarios,
        summary_rows=summary_rows,
        validation_messages=validation_messages,
        figure_paths=figure_paths,
        elapsed_seconds=elapsed_seconds,
    )
    print(f"Wrote {args.output_dir / 'classical_distribution_samples.csv'}")
    print(f"Wrote {args.output_dir / 'split_assignments.csv'}")
    print(f"Wrote {args.output_dir / 'heldout_log_scores.csv'}")
    print(f"Wrote {args.output_dir / 'benchmark_summary.csv'}")
    print(f"Wrote {args.output_dir / 'runtime.csv'}")
    print(f"Wrote {args.output_dir / 'run_parameters.txt'}")
    print(f"Wrote {args.output_dir / 'run_parameters.json'}")
    print(f"Wrote {args.output_dir / 'schema_validation.txt'}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
