#!/usr/bin/env python3
"""Run support-aware geometric baselines for the JMLR frozen track.

The support-aware baselines are intentionally modest: an MST tube learns a
landmark graph by farthest-point subsampling, while a principal-spline tube
orders samples by the first principal direction and fits a smoothed curve.
Both evaluate full-rank Gaussian tube densities by quadrature. These are
tube-style baselines, not LAND implementations.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

(ROOT / "tmp" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / "tmp" / "cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "cache"))

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from scripts.run_jmlr_density_benchmark import (
    FACTORIES,
    fit_gaussian,
    fit_gmm_em,
    fit_kde,
    write_csv,
)


PROTOCOL_VERSION = "jmlr_frozen_benchmark_v1"
SCENARIO_FAMILY = "builtin_mixed_support"
DEFAULT_OUT = ROOT / "outputs" / "jmlr_frozen_benchmark_v1" / "support_baselines"
DEFAULT_REPORT = DEFAULT_OUT / "department_d_report.md"
DEFAULT_FIGURE_PREFIX = "department_d_support_baselines_heldout_log_score"
REQUIRED_RUN_COLUMNS = {
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
]
SUMMARY_COLUMNS = {
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
}
RUNTIME_COLUMNS = {
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
}
LAND_CITATION = (
    "Arvanitidis, G., Hansen, L. K., & Hauberg, S. (2016). "
    "A locally adaptive normal distribution. Advances in Neural Information "
    "Processing Systems, 4258-4266."
)
LAND_IMPLEMENTATION_URL = "https://github.com/georgiosarvanitidis/geometric_ml"


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
    return out


def as_csv_float(value: float | str) -> float | str:
    if isinstance(value, str):
        return value
    if np.isfinite(value):
        return float(value)
    return "NA"


def short_error(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text.replace("\n", " ")[:500]


def score_from_logpdf(logpdf: np.ndarray) -> tuple[float | str, int, str, str, str]:
    values = np.asarray(logpdf, dtype=float)
    finite = np.isfinite(values)
    nonfinite_count = int(values.size - np.count_nonzero(finite))
    if nonfinite_count:
        return (
            "NA",
            nonfinite_count,
            "warning",
            "NonFiniteLogPdf",
            f"{nonfinite_count} non-finite log-density values returned; score set to NA.",
        )
    return float(np.mean(values)), 0, "ok", "", ""


def make_row(
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    method: str,
    seed: int,
    hyperparameter: str,
    heldout_log_score: float | str = "NA",
    fit_seconds: float = 0.0,
    eval_seconds: float = 0.0,
    status: str = "ok",
    converged: bool | str = True,
    n_iter: int | str = 0,
    error_type: str = "",
    error_message: str = "",
    nonfinite_count: int = 0,
) -> dict[str, object]:
    row = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": scenario,
        "scenario_family": SCENARIO_FAMILY,
        "rep": int(rep),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "method": method,
        "heldout_log_score": as_csv_float(heldout_log_score),
        "fit_seconds": float(fit_seconds),
        "eval_seconds": float(eval_seconds),
        "seed": int(seed),
        "hyperparameter": hyperparameter,
        "status": status,
        "converged": converged,
        "n_iter": n_iter,
        "error_type": error_type,
        "error_message": error_message,
        "nonfinite_count": int(nonfinite_count),
    }
    return {column: row[column] for column in RUN_COLUMNS}


def append_failure_row(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    method: str,
    seed: int,
    hyperparameter: str,
    exc: BaseException,
    fit_seconds: float,
) -> None:
    rows.append(
        make_row(
            scenario=scenario,
            rep=rep,
            n_train=n_train,
            n_test=n_test,
            method=method,
            seed=seed,
            hyperparameter=hyperparameter,
            fit_seconds=fit_seconds,
            status="failed",
            converged=False,
            n_iter=0,
            error_type=exc.__class__.__name__,
            error_message=short_error(exc),
        )
    )


def farthest_point_indices(x: np.ndarray, n_landmarks: int, *, seed: int) -> np.ndarray:
    """Choose deterministic spread-out landmarks from the training sample."""
    n = x.shape[0]
    k = min(max(2, n_landmarks), n)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    selected = [first]
    min_sq = np.sum((x - x[first]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(min_sq))
        selected.append(idx)
        dist_sq = np.sum((x - x[idx]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, dist_sq)
    return np.asarray(selected, dtype=int)


def point_segment_distances(x: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Return distances from each point to its nearest learned graph segment."""
    directions = ends - starts
    lengths_sq = np.sum(directions * directions, axis=1)
    valid = lengths_sq > 0.0
    starts = starts[valid]
    ends = ends[valid]
    directions = directions[valid]
    lengths_sq = lengths_sq[valid]
    if starts.size == 0:
        center = np.mean(x, axis=0, keepdims=True)
        return np.linalg.norm(x - center, axis=1)

    best = np.full(x.shape[0], np.inf)
    for start, direction, length_sq in zip(starts, directions, lengths_sq):
        rel = x - start
        t = np.clip((rel @ direction) / length_sq, 0.0, 1.0)
        projection = start + t[:, None] * direction
        best = np.minimum(best, np.linalg.norm(x - projection, axis=1))
    return best


@dataclass
class GeometricTubeFit:
    starts: np.ndarray
    ends: np.ndarray
    lengths: np.ndarray
    sigma: float
    quad_nodes: np.ndarray
    quad_weights: np.ndarray

    @property
    def n_edges(self) -> int:
        return int(self.lengths.size)

    @property
    def total_length(self) -> float:
        return float(np.sum(self.lengths))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        d = x.shape[1]
        total_length = max(self.total_length, np.finfo(float).tiny)
        log_norm = -0.5 * d * np.log(2.0 * np.pi * self.sigma * self.sigma)
        cols = []
        for start, end, length in zip(self.starts, self.ends, self.lengths):
            points = start[None, :] + self.quad_nodes[:, None] * (end - start)[None, :]
            diff = x[:, None, :] - points[None, :, :]
            sq = np.sum(diff * diff, axis=2)
            log_terms = (
                np.log(max(length, np.finfo(float).tiny) / total_length)
                + np.log(self.quad_weights)[None, :]
                + log_norm
                - 0.5 * sq / (self.sigma * self.sigma)
            )
            cols.append(logsumexp(log_terms, axis=1))
        return logsumexp(np.vstack(cols).T, axis=1)


def fit_geometric_tube(
    x: np.ndarray,
    *,
    n_landmarks: int,
    random_state: int,
    min_sigma: float,
    sigma_scale: float,
    quadrature: int,
) -> GeometricTubeFit:
    landmarks = x[farthest_point_indices(x, n_landmarks, seed=random_state)]
    dist = cdist(landmarks, landmarks)
    mst = minimum_spanning_tree(dist).tocoo()
    starts = landmarks[mst.row]
    ends = landmarks[mst.col]
    lengths = np.asarray(mst.data, dtype=float)
    valid = lengths > 0.0
    starts = starts[valid]
    ends = ends[valid]
    lengths = lengths[valid]
    if starts.size == 0:
        order = np.argsort(x[:, 0])
        starts = x[order[:-1]]
        ends = x[order[1:]]
        lengths = np.linalg.norm(ends - starts, axis=1)
        valid = lengths > 0.0
        starts = starts[valid]
        ends = ends[valid]
        lengths = lengths[valid]
    if starts.size == 0:
        raise ValueError("Could not construct a nondegenerate geometric tube graph.")

    residual = point_segment_distances(x, starts, ends)
    residual_sigma = np.sqrt(np.mean(residual * residual) / max(x.shape[1] - 1, 1))
    nearest = np.partition(cdist(x, x), kth=1, axis=1)[:, 1]
    spacing_sigma = float(np.median(nearest))
    sigma = max(float(sigma_scale * residual_sigma), 0.5 * spacing_sigma, min_sigma)

    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return GeometricTubeFit(
        starts=np.asarray(starts, dtype=float),
        ends=np.asarray(ends, dtype=float),
        lengths=np.asarray(lengths, dtype=float),
        sigma=float(sigma),
        quad_nodes=np.asarray(nodes, dtype=float),
        quad_weights=np.asarray(weights, dtype=float),
    )


def principal_axis_scores(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[0]
    span = float(np.max(scores) - np.min(scores))
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("Could not find a nondegenerate principal ordering.")
    return (scores - np.min(scores)) / span


def spline_support_centers(x: np.ndarray, scores: np.ndarray, n_knots: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    k = min(max(4, n_knots), n)
    order = np.argsort(scores)
    score_sorted = scores[order]
    x_sorted = x[order]
    chunks = np.array_split(np.arange(n), k)
    center_scores = []
    centers = []
    for chunk in chunks:
        if chunk.size == 0:
            continue
        center_scores.append(float(np.mean(score_sorted[chunk])))
        centers.append(np.mean(x_sorted[chunk], axis=0))
    center_scores = np.asarray(center_scores, dtype=float)
    centers = np.asarray(centers, dtype=float)
    unique_scores, unique_idx = np.unique(center_scores, return_index=True)
    centers = centers[unique_idx]
    if unique_scores.size < 4:
        raise ValueError("Principal-spline tube needs at least four unique support centers.")
    return unique_scores, centers


def fit_principal_spline_tube(
    x: np.ndarray,
    *,
    n_knots: int,
    n_curve_points: int,
    smoothing: float,
    min_sigma: float,
    sigma_scale: float,
    quadrature: int,
) -> GeometricTubeFit:
    scores = principal_axis_scores(x)
    center_scores, centers = spline_support_centers(x, scores, n_knots)
    degree = min(3, center_scores.size - 1)
    grid = np.linspace(0.0, 1.0, max(n_curve_points, center_scores.size))
    curve_cols = []
    for dim in range(x.shape[1]):
        scale = float(np.var(centers[:, dim]))
        smooth_budget = max(0.0, smoothing) * center_scores.size * scale
        spline = UnivariateSpline(center_scores, centers[:, dim], k=degree, s=smooth_budget)
        curve_cols.append(spline(grid))
    curve = np.vstack(curve_cols).T
    starts = curve[:-1]
    ends = curve[1:]
    lengths = np.linalg.norm(ends - starts, axis=1)
    valid = lengths > 0.0
    starts = starts[valid]
    ends = ends[valid]
    lengths = lengths[valid]
    if starts.size == 0:
        raise ValueError("Could not construct a nondegenerate principal-spline tube.")

    residual = point_segment_distances(x, starts, ends)
    residual_sigma = np.sqrt(np.mean(residual * residual) / max(x.shape[1] - 1, 1))
    nearest = np.partition(cdist(x, x), kth=1, axis=1)[:, 1]
    spacing_sigma = float(np.median(nearest))
    sigma = max(float(sigma_scale * residual_sigma), 0.5 * spacing_sigma, min_sigma)

    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return GeometricTubeFit(
        starts=np.asarray(starts, dtype=float),
        ends=np.asarray(ends, dtype=float),
        lengths=np.asarray(lengths, dtype=float),
        sigma=float(sigma),
        quad_nodes=np.asarray(nodes, dtype=float),
        quad_weights=np.asarray(weights, dtype=float),
    )


def append_oracle_row(
    rows: list[dict[str, object]],
    dist,
    test: np.ndarray,
    *,
    scenario: str,
    rep: int,
    seed: int,
    args,
) -> None:
    start = time.perf_counter()
    hyperparameter = f"n_grid={args.n_grid}"
    try:
        oracle_logpdf = np.asarray(dist.logpdf(test, n_grid=args.n_grid), dtype=float)
        eval_seconds = time.perf_counter() - start
        score, nonfinite_count, status, error_type, error_message = score_from_logpdf(oracle_logpdf)
        rows.append(
            make_row(
                scenario=scenario,
                rep=rep,
                n_train=args.n_train,
                n_test=args.n_test,
                method="Oracle SFD",
                heldout_log_score=score,
                fit_seconds=0.0,
                eval_seconds=eval_seconds,
                seed=seed,
                hyperparameter=hyperparameter,
                status=status,
                error_type=error_type,
                error_message=error_message,
                nonfinite_count=nonfinite_count,
            )
        )
    except Exception as exc:
        append_failure_row(
            rows,
            scenario=scenario,
            rep=rep,
            n_train=args.n_train,
            n_test=args.n_test,
            method="Oracle SFD",
            seed=seed,
            hyperparameter=hyperparameter,
            exc=exc,
            fit_seconds=0.0,
        )


def append_classical_rows(
    rows: list[dict[str, object]],
    train: np.ndarray,
    test: np.ndarray,
    *,
    scenario: str,
    rep: int,
    seed: int,
    args,
) -> None:
    method_specs = [("Gaussian", f"reg={args.reg:g}")]
    method_specs.extend(
        (f"GMM-{min(k, train.shape[0])}", f"K={k};reg={args.reg:g};max_iter={args.max_iter};tol={args.tol:g}")
        for k in args.gmm_components
    )
    if args.kde:
        method_specs.append(("KDE", f"bandwidth=scott;jitter={args.kde_jitter:g}"))

    for method, hyperparameter in method_specs:
        start = time.perf_counter()
        try:
            if method == "Gaussian":
                fit = fit_gaussian(train, args.reg)
                n_iter: int | str = 0
                converged: bool | str = True
            elif method.startswith("GMM-"):
                k = int(method.split("-", 1)[1])
                fit = fit_gmm_em(
                    train,
                    k,
                    random_state=seed + 17 * k,
                    reg=args.reg,
                    max_iter=args.max_iter,
                    tol=args.tol,
                )
                n_iter = "not_recorded"
                converged = "not_recorded"
            else:
                fit = fit_kde(train, random_state=seed + 7919, jitter=args.kde_jitter)
                n_iter = 0
                converged = True
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            score, nonfinite_count, status, error_type, error_message = score_from_logpdf(fit.logpdf(test))
            eval_seconds = time.perf_counter() - start
            rows.append(
                make_row(
                    scenario=scenario,
                    rep=rep,
                    n_train=int(train.shape[0]),
                    n_test=int(test.shape[0]),
                    method=method,
                    heldout_log_score=score,
                    fit_seconds=fit_seconds,
                    eval_seconds=eval_seconds,
                    seed=seed,
                    hyperparameter=hyperparameter,
                    status=status,
                    converged=converged,
                    n_iter=n_iter,
                    error_type=error_type,
                    error_message=error_message,
                    nonfinite_count=nonfinite_count,
                )
            )
        except Exception as exc:
            append_failure_row(
                rows,
                scenario=scenario,
                rep=rep,
                n_train=int(train.shape[0]),
                n_test=int(test.shape[0]),
                method=method,
                seed=seed,
                hyperparameter=hyperparameter,
                exc=exc,
                fit_seconds=time.perf_counter() - start,
            )


def append_mst_tube_row(
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    train: np.ndarray,
    test: np.ndarray,
    *,
    scenario: str,
    rep: int,
    seed: int,
    args,
) -> None:
    start = time.perf_counter()
    method = "MST tube baseline"
    try:
        fit = fit_geometric_tube(
            train,
            n_landmarks=args.tube_landmarks,
            random_state=seed + 4243,
            min_sigma=args.tube_min_sigma,
            sigma_scale=args.tube_sigma_scale,
            quadrature=args.tube_quadrature,
        )
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        score, nonfinite_count, status, error_type, error_message = score_from_logpdf(fit.logpdf(test))
        eval_seconds = time.perf_counter() - start
        hyperparameter = (
            f"graph=MST;landmarks={args.tube_landmarks};sigma={fit.sigma:.6g};"
            f"sigma_scale={args.tube_sigma_scale:g};quadrature={args.tube_quadrature}"
        )
        rows.append(
            make_row(
                scenario=scenario,
                rep=rep,
                n_train=int(train.shape[0]),
                n_test=int(test.shape[0]),
                method=method,
                heldout_log_score=score,
                fit_seconds=fit_seconds,
                eval_seconds=eval_seconds,
                seed=seed,
                hyperparameter=hyperparameter,
                status=status,
                error_type=error_type,
                error_message=error_message,
                nonfinite_count=nonfinite_count,
            )
        )
        diagnostics.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario_family": SCENARIO_FAMILY,
                "scenario": scenario,
                "rep": rep,
                "seed": seed,
                "method": method,
                "n_edges": fit.n_edges,
                "total_graph_length": fit.total_length,
                "tube_sigma": fit.sigma,
                "tube_landmarks_requested": args.tube_landmarks,
                "tube_quadrature": args.tube_quadrature,
                "status": status,
            }
        )
    except Exception as exc:
        append_failure_row(
            rows,
            scenario=scenario,
            rep=rep,
            n_train=int(train.shape[0]),
            n_test=int(test.shape[0]),
            method=method,
            seed=seed,
            hyperparameter=(
                f"graph=MST;landmarks={args.tube_landmarks};"
                f"sigma_scale={args.tube_sigma_scale:g};quadrature={args.tube_quadrature}"
            ),
            exc=exc,
            fit_seconds=time.perf_counter() - start,
        )
        diagnostics.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario_family": SCENARIO_FAMILY,
                "scenario": scenario,
                "rep": rep,
                "seed": seed,
                "method": method,
                "n_edges": "NA",
                "total_graph_length": "NA",
                "tube_sigma": "NA",
                "tube_landmarks_requested": args.tube_landmarks,
                "tube_quadrature": args.tube_quadrature,
                "status": "failed",
            }
        )


def append_principal_spline_row(
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    train: np.ndarray,
    test: np.ndarray,
    *,
    scenario: str,
    rep: int,
    seed: int,
    args,
) -> None:
    start = time.perf_counter()
    method = "Principal spline tube baseline"
    try:
        fit = fit_principal_spline_tube(
            train,
            n_knots=args.spline_knots,
            n_curve_points=args.spline_curve_points,
            smoothing=args.spline_smoothing,
            min_sigma=args.tube_min_sigma,
            sigma_scale=args.tube_sigma_scale,
            quadrature=args.tube_quadrature,
        )
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        score, nonfinite_count, status, error_type, error_message = score_from_logpdf(fit.logpdf(test))
        eval_seconds = time.perf_counter() - start
        hyperparameter = (
            f"ordering=PCA1;knots={args.spline_knots};curve_points={args.spline_curve_points};"
            f"smoothing={args.spline_smoothing:g};sigma={fit.sigma:.6g};"
            f"sigma_scale={args.tube_sigma_scale:g};quadrature={args.tube_quadrature}"
        )
        rows.append(
            make_row(
                scenario=scenario,
                rep=rep,
                n_train=int(train.shape[0]),
                n_test=int(test.shape[0]),
                method=method,
                heldout_log_score=score,
                fit_seconds=fit_seconds,
                eval_seconds=eval_seconds,
                seed=seed,
                hyperparameter=hyperparameter,
                status=status,
                error_type=error_type,
                error_message=error_message,
                nonfinite_count=nonfinite_count,
            )
        )
        diagnostics.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario_family": SCENARIO_FAMILY,
                "scenario": scenario,
                "rep": rep,
                "seed": seed,
                "method": method,
                "n_edges": fit.n_edges,
                "total_graph_length": fit.total_length,
                "tube_sigma": fit.sigma,
                "tube_landmarks_requested": args.spline_knots,
                "tube_quadrature": args.tube_quadrature,
                "status": status,
            }
        )
    except Exception as exc:
        append_failure_row(
            rows,
            scenario=scenario,
            rep=rep,
            n_train=int(train.shape[0]),
            n_test=int(test.shape[0]),
            method=method,
            seed=seed,
            hyperparameter=(
                f"ordering=PCA1;knots={args.spline_knots};curve_points={args.spline_curve_points};"
                f"smoothing={args.spline_smoothing:g};sigma_scale={args.tube_sigma_scale:g};"
                f"quadrature={args.tube_quadrature}"
            ),
            exc=exc,
            fit_seconds=time.perf_counter() - start,
        )
        diagnostics.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario_family": SCENARIO_FAMILY,
                "scenario": scenario,
                "rep": rep,
                "seed": seed,
                "method": method,
                "n_edges": "NA",
                "total_graph_length": "NA",
                "tube_sigma": "NA",
                "tube_landmarks_requested": args.spline_knots,
                "tube_quadrature": args.tube_quadrature,
                "status": "failed",
            }
        )


def append_land_blocker_row(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    seed: int,
) -> None:
    rows.append(
        make_row(
            scenario=scenario,
            rep=rep,
            n_train=n_train,
            n_test=n_test,
            method="LAND-style local metric baseline",
            seed=seed,
            hyperparameter="implementation_status=blocked;see=land_style_feasibility_note.md",
            status="failed",
            converged=False,
            n_iter=0,
            error_type="ImplementationBlocked",
            error_message="LAND baseline not implemented or audited in this runner.",
        )
    )


def numeric_or_none(value: object) -> float | None:
    if value in {"", "NA", None}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def summarize_frozen(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(r["scenario"], r["method"]) for r in rows})
    out = []
    for scenario, method in keys:
        sub = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
        scores = [numeric_or_none(r["heldout_log_score"]) for r in sub]
        finite_scores = np.asarray([s for s in scores if s is not None], dtype=float)
        fit = np.asarray([float(r["fit_seconds"]) for r in sub], dtype=float)
        eval_time = np.asarray([float(r["eval_seconds"]) for r in sub], dtype=float)
        failure_rate = sum(1 for r in sub if r["status"] == "failed") / len(sub)
        warning_rate = sum(1 for r in sub if r["status"] == "warning") / len(sub)
        if finite_scores.size:
            mean_score: float | str = float(np.mean(finite_scores))
            se_score: float | str = (
                float(np.std(finite_scores, ddof=1) / np.sqrt(finite_scores.size))
                if finite_scores.size > 1
                else 0.0
            )
        else:
            mean_score = "NA"
            se_score = "NA"
        out.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": scenario,
                "scenario_family": SCENARIO_FAMILY,
                "method": method,
                "mean_heldout_log_score": mean_score,
                "se_heldout_log_score": se_score,
                "rank_within_scenario": "NA",
                "mean_fit_seconds": float(np.mean(fit)),
                "mean_eval_seconds": float(np.mean(eval_time)),
                "reps": int(len(sub)),
                "failure_rate": float(failure_rate),
                "warning_rate": float(warning_rate),
            }
        )
    for scenario in sorted({str(r["scenario"]) for r in out}):
        scenario_rows = [
            r
            for r in out
            if r["scenario"] == scenario
            and numeric_or_none(r["mean_heldout_log_score"]) is not None
            and float(r["failure_rate"]) <= 0.10
        ]
        scenario_rows.sort(key=lambda r: float(r["mean_heldout_log_score"]), reverse=True)
        for rank, row in enumerate(scenario_rows, start=1):
            row["rank_within_scenario"] = rank
    return out


def summarize_runtime_frozen(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in summary_rows:
        reps = int(row["reps"])
        mean_fit = float(row["mean_fit_seconds"])
        mean_eval = float(row["mean_eval_seconds"])
        out.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": row["scenario"],
                "scenario_family": row["scenario_family"],
                "method": row["method"],
                "reps": reps,
                "total_fit_seconds": mean_fit * reps,
                "mean_fit_seconds": mean_fit,
                "total_eval_seconds": mean_eval * reps,
                "mean_eval_seconds": mean_eval,
                "peak_memory_mb": "NA",
            }
        )
    return out


def validate_schema(
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
) -> list[str]:
    messages = []
    if not rows:
        raise ValueError("No held-out score rows were produced.")
    missing = REQUIRED_RUN_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"Per-run rows missing required columns: {sorted(missing)}")
    for idx, row in enumerate(rows, start=1):
        for column in REQUIRED_RUN_COLUMNS:
            if column not in row:
                raise ValueError(f"Row {idx} missing {column!r}.")
        if row["status"] not in {"ok", "warning", "failed"}:
            raise ValueError(f"Row {idx} has unsupported status: {row['status']}")
        for column in ("fit_seconds", "eval_seconds"):
            value = float(row[column])
            if not np.isfinite(value):
                raise ValueError(f"Row {idx} has non-finite {column}: {value}")
        score = numeric_or_none(row["heldout_log_score"])
        if row["status"] == "ok" and score is None:
            raise ValueError(f"Successful row {idx} has missing or non-finite heldout_log_score.")
    if summary_rows:
        missing_summary = SUMMARY_COLUMNS - set(summary_rows[0])
        if missing_summary:
            raise ValueError(f"Summary rows missing required columns: {sorted(missing_summary)}")
    if runtime_rows:
        missing_runtime = RUNTIME_COLUMNS - set(runtime_rows[0])
        if missing_runtime:
            raise ValueError(f"Runtime rows missing required columns: {sorted(missing_runtime)}")
    messages.append(f"per_run_rows={len(rows)}")
    messages.append(f"summary_rows={len(summary_rows)}")
    messages.append(f"runtime_rows={len(runtime_rows)}")
    messages.append("frozen_required_columns_present=1")
    messages.append("status_failure_handling_present=1")
    return messages


def maybe_plot(out_dir: Path, figure_dir: Path, figure_prefix: str, summary_rows: list[dict[str, object]]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    scenario_order = ["snowflake", "doraemon", "maze", "ecg", "torus"]
    scenario_order.extend(
        sorted({str(r["scenario"]) for r in summary_rows if str(r["scenario"]) not in scenario_order})
    )
    scenario_labels = {
        "snowflake": "Snowflake",
        "doraemon": "Doraemon-style",
        "maze": "Maze graph",
        "ecg": "ECG-like flow",
        "torus": "Torus knot",
    }
    method_order = [
        "Gaussian",
        "KDE",
        "GMM-2",
        "GMM-4",
        "GMM-8",
        "GMM-16",
        "MST tube baseline",
        "Principal spline tube baseline",
        "LAND-style local metric baseline",
        "Oracle SFD",
    ]
    color = {
        "Gaussian": "#9fd8cb",
        "KDE": "#d7b56d",
        "GMM-2": "#91c4df",
        "GMM-4": "#b7a4df",
        "GMM-8": "#f7c979",
        "GMM-16": "#c7d77f",
        "MST tube baseline": "#6f78b8",
        "Principal spline tube baseline": "#4b8f78",
        "LAND-style local metric baseline": "#a9b0ba",
        "Oracle SFD": "#f28e88",
    }
    scenarios = [s for s in scenario_order if any(str(r["scenario"]) == s for r in summary_rows)]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.35 * len(scenarios), 3.8), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        sub = [r for r in summary_rows if r["scenario"] == scenario]
        sub = [r for r in sub if numeric_or_none(r["mean_heldout_log_score"]) is not None]
        sub.sort(key=lambda r: method_order.index(str(r["method"])) if str(r["method"]) in method_order else 999)
        if not sub:
            ax.set_axis_off()
            ax.set_title(scenario_labels.get(scenario, scenario.replace("_", " ").title()), fontweight="bold")
            continue
        x = np.arange(len(sub))
        y = [float(r["mean_heldout_log_score"]) for r in sub]
        yerr = [float(r["se_heldout_log_score"]) if numeric_or_none(r["se_heldout_log_score"]) is not None else 0.0 for r in sub]
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
        ax.set_title(scenario_labels.get(scenario, scenario.replace("_", " ").title()), fontweight="bold")
        ax.set_ylabel("held-out log score")
        ax.axhline(0.0, color="#53606b", linewidth=0.55, alpha=0.45)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for path in [
        out_dir / f"{figure_prefix}.png",
        out_dir / f"{figure_prefix}.pdf",
        figure_dir / f"{figure_prefix}.png",
        figure_dir / f"{figure_prefix}.pdf",
    ]:
        if path not in paths:
            paths.append(path)
    for path in paths:
        fig.savefig(path, dpi=220 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def write_parameters(path: Path, args) -> None:
    lines = [
        "Department D support-aware baselines for JMLR frozen benchmark",
        f"protocol_version={PROTOCOL_VERSION}",
        f"scenario_family={SCENARIO_FAMILY}",
        "evidence_label=standard_development_run",
        "method_boundary=Tube baselines are not LAND; blocked LAND rows are explicit failure rows.",
        f"command={args.command}",
        f"scenarios={','.join(args.scenarios)}",
        f"reps={args.reps}",
        f"seed={args.seed}",
        "seed_formula=seed + 10000 * rep + 101 * scenario_index",
        f"n_train={args.n_train}",
        f"n_test={args.n_test}",
        f"gmm_components={','.join(map(str, args.gmm_components))}",
        f"kde={args.kde}",
        f"oracle={args.oracle}",
        f"land_blocker_rows={args.land_blocker_rows}",
        f"n_grid={args.n_grid}",
        f"reg={args.reg}",
        f"max_iter={args.max_iter}",
        f"tol={args.tol}",
        f"kde_jitter={args.kde_jitter}",
        f"tube_landmarks={args.tube_landmarks}",
        f"tube_sigma_scale={args.tube_sigma_scale}",
        f"tube_min_sigma={args.tube_min_sigma}",
        f"tube_quadrature={args.tube_quadrature}",
        f"spline_knots={args.spline_knots}",
        f"spline_curve_points={args.spline_curve_points}",
        f"spline_smoothing={args.spline_smoothing}",
        f"land_citation={LAND_CITATION}",
        f"land_candidate_implementation_url={LAND_IMPLEMENTATION_URL}",
        f"output_dir={args.output_dir}",
        f"figure_dir={args.figure_dir}",
        f"figure_prefix={args.figure_prefix}",
        f"report={args.report}",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_parameters_json(path: Path, args) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario_family": SCENARIO_FAMILY,
        "evidence_label": "standard_development_run",
        "command": args.command,
        "scenarios": args.scenarios,
        "reps": args.reps,
        "seed": args.seed,
        "seed_formula": "seed + 10000 * rep + 101 * scenario_index",
        "n_train": args.n_train,
        "n_test": args.n_test,
        "methods": {
            "oracle": args.oracle,
            "gaussian": True,
            "gmm_components": args.gmm_components,
            "kde": args.kde,
            "mst_tube": {
                "landmarks": args.tube_landmarks,
                "sigma_scale": args.tube_sigma_scale,
                "min_sigma": args.tube_min_sigma,
                "quadrature": args.tube_quadrature,
            },
            "principal_spline_tube": {
                "ordering": "PCA1",
                "knots": args.spline_knots,
                "curve_points": args.spline_curve_points,
                "smoothing": args.spline_smoothing,
                "sigma_scale": args.tube_sigma_scale,
                "min_sigma": args.tube_min_sigma,
                "quadrature": args.tube_quadrature,
            },
            "land_style_local_metric_baseline": {
                "status": "blocked",
                "blocker_rows_written": args.land_blocker_rows,
                "citation": LAND_CITATION,
                "candidate_implementation_url": LAND_IMPLEMENTATION_URL,
            },
        },
        "failure_rules": {
            "success_status": "ok",
            "warning_status": "warning",
            "failure_status": "failed",
            "nonfinite_logpdf": "heldout_log_score=NA and nonfinite_count recorded",
            "blocked_land": "status=failed rows when --land-blocker-rows is enabled",
        },
        "outputs": {
            "output_dir": str(args.output_dir),
            "report": str(args.report),
            "figure_dir": str(args.figure_dir),
            "figure_prefix": args.figure_prefix,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_land_note(path: Path, args) -> None:
    text = [
        "# LAND-Style Baseline Feasibility Note",
        "",
        f"Protocol version: `{PROTOCOL_VERSION}`.",
        "",
        "## Citation",
        "",
        LAND_CITATION,
        "",
        "The paper defines a locally adaptive normal distribution by replacing the Euclidean metric "
        "in a Gaussian-style density with a data-adaptive Riemannian metric and fitting parameters "
        "with gradient descent plus Monte Carlo normalization. The corresponding mixture extension "
        "uses an EM-style procedure.",
        "",
        "## Implementation Status",
        "",
        "`scripts/run_geometric_baseline_pilot.py` does not implement LAND. It writes explicit "
        "`LAND-style local metric baseline` rows with `status=failed` and "
        "`error_type=ImplementationBlocked` so the required baseline is visible rather than silently omitted.",
        "",
        "A candidate research-code source was identified but not vendored, audited, or run in this task:",
        "",
        f"- `{LAND_IMPLEMENTATION_URL}`",
        "",
        "## Settings That Must Be Frozen Before A Real LAND Run",
        "",
        "- metric estimator: local covariance/Riemannian metric construction and regularization;",
        "- metric smoothing or kernel bandwidth;",
        "- geodesic or shortest-path solver and tolerances;",
        "- logarithmic/exponential map approximation rules;",
        "- Monte Carlo sample count and random seed for normalization;",
        "- optimizer, learning rate, stopping rule, and maximum iterations;",
        "- mixture component count, initialization, and EM convergence rule if a mixture LAND is used;",
        "- density-evaluation convention and failure handling for non-finite geodesic distances or normalizers.",
        "",
        "## Current Department D Run Settings",
        "",
        f"- blocker rows enabled: `{args.land_blocker_rows}`",
        "- blocker method name: `LAND-style local metric baseline`",
        "- blocker row status: `failed`",
        "- blocker error type: `ImplementationBlocked`",
        "- blocker hyperparameter field: `implementation_status=blocked;see=land_style_feasibility_note.md`",
        f"- scenarios: `{','.join(args.scenarios)}`",
        f"- repetitions: `{args.reps}`",
        f"- train/test sizes: `{args.n_train}` / `{args.n_test}`",
        f"- output directory: `{args.output_dir}`",
        "",
        "Claim boundary: these rows document a missing LAND implementation. They are not LAND results and "
        "must not be used as evidence that SFD outperforms LAND.",
        "",
    ]
    path.write_text("\n".join(text))


def read_csv_rows(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def fmt_cell(value: object, digits: int = 4) -> str:
    numeric = numeric_or_none(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.{digits}f}"


def rank_sort_value(value: object) -> tuple[int, int]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, 10_000)


def write_report(
    path: Path,
    *,
    args,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    validation_messages: list[str],
    figure_paths: list[Path],
    elapsed_seconds: float,
) -> None:
    method_names = sorted({str(row["method"]) for row in rows})
    score_rows = read_csv_rows(args.output_dir / "heldout_log_scores.csv")
    summary_count = read_csv_rows(args.output_dir / "benchmark_summary.csv")
    runtime_count = read_csv_rows(args.output_dir / "runtime.csv")
    diagnostic_count = read_csv_rows(args.output_dir / "tube_baseline_diagnostics.csv")
    ranks = [
        f"| {row['scenario']} | {row['method']} | {fmt_cell(row['mean_heldout_log_score'])} | "
        f"{fmt_cell(row['se_heldout_log_score'])} | {row['rank_within_scenario']} | "
        f"{fmt_cell(row['mean_fit_seconds'])} | {fmt_cell(row['mean_eval_seconds'])} | "
        f"{fmt_cell(row['failure_rate'], 3)} | {fmt_cell(row['warning_rate'], 3)} |"
        for row in sorted(summary_rows, key=lambda r: (str(r["scenario"]), rank_sort_value(r["rank_within_scenario"]), str(r["method"])))
    ]
    sigma_lines = [
        f"| {row['scenario']} | {row['rep']} | {row['method']} | {row['n_edges']} | "
        f"{fmt_cell(row['total_graph_length'])} | {fmt_cell(row['tube_sigma'], 5)} | {row['status']} |"
        for row in diagnostics
    ]
    figure_lines = [f"- `{path.relative_to(ROOT)}`" for path in figure_paths]
    status_counts = {
        status: sum(1 for row in rows if row["status"] == status)
        for status in ("ok", "warning", "failed")
    }
    land_rows = [row for row in rows if row["method"] == "LAND-style local metric baseline"]
    support_methods = ["MST tube baseline", "Principal spline tube baseline", "LAND-style local metric baseline"]
    text = [
        "# Department D Support-Aware Baseline Report",
        "",
        f"Protocol version: `{PROTOCOL_VERSION}`.",
        "",
        "Evidence label: standard development run, not final paper-facing Monte Carlo evidence.",
        "",
        "This run adds two support-aware tube baselines to the built-in JMLR density benchmark: "
        "a conservative landmark MST tube and a principal-spline tube ordered by the first principal "
        "direction. Neither method is a LAND implementation. LAND-style rows are explicit blocked "
        "rows tied to `land_style_feasibility_note.md`.",
        "",
        "## Command",
        "",
        "```bash",
        args.command,
        "```",
        "",
        "## Run Parameters",
        "",
        f"- scenarios: `{','.join(args.scenarios)}`",
        f"- train/test sizes: `{args.n_train}` / `{args.n_test}`",
        f"- repetitions: `{args.reps}`",
        f"- seed: `{args.seed}`",
        f"- methods: `{', '.join(method_names)}`",
        f"- tube hyperparameters: landmarks=`{args.tube_landmarks}`, sigma_scale=`{args.tube_sigma_scale}`, "
        f"min_sigma=`{args.tube_min_sigma}`, quadrature=`{args.tube_quadrature}`",
        f"- principal-spline hyperparameters: knots=`{args.spline_knots}`, curve_points=`{args.spline_curve_points}`, "
        f"smoothing=`{args.spline_smoothing}`",
        f"- LAND citation: {LAND_CITATION}",
        f"- elapsed wall time inside script: `{elapsed_seconds:.3f}` seconds",
        "",
        "## Schema And Row Counts",
        "",
        f"- `heldout_log_scores.csv`: {score_rows} rows",
        f"- `benchmark_summary.csv`: {summary_count} rows",
        f"- `runtime.csv`: {runtime_count} rows",
        f"- `tube_baseline_diagnostics.csv`: {diagnostic_count} rows",
        f"- validation: `{'; '.join(validation_messages)}`",
        f"- status counts: ok=`{status_counts['ok']}`, warning=`{status_counts['warning']}`, failed=`{status_counts['failed']}`",
        f"- LAND blocked rows: `{len(land_rows)}`",
        "",
        "Required per-run columns follow the frozen schema: `protocol_version`, `scenario`, "
        "`scenario_family`, `rep`, `n_train`, `n_test`, `method`, `heldout_log_score`, "
        "`fit_seconds`, `eval_seconds`, `seed`, `hyperparameter`, `status`, `converged`, "
        "`n_iter`, `error_type`, and `error_message`.",
        "",
        "## Summary Ranks",
        "",
        "| scenario | method | mean held-out log score | SE | rank | mean fit s | mean eval s | failure rate | warning rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *ranks,
        "",
        "## Tube Diagnostics",
        "",
        "| scenario | rep | method | graph edges | total graph length | tube sigma | status |",
        "|---|---:|---|---:|---:|---:|---|",
        *sigma_lines,
        "",
        "## Figures",
        "",
        *figure_lines,
        "",
        "## Support-Aware Method Boundary",
        "",
        f"- support-aware methods tracked by Department D: `{', '.join(support_methods)}`",
        "- The MST tube is conservative and can connect nearby but topologically distinct branches.",
        "- The principal-spline tube is a one-dimensional principal-curve approximation and can miss branching topology.",
        "- LAND-style rows are blocked implementation rows, not LAND performance results.",
        "",
        "## Failure Modes And Claim Boundary",
        "",
        "- The isotropic tube bandwidth is estimated from training residuals and is not tuned by validation likelihood.",
        "- This standard development run uses modest repetitions and should not be written as a final paper-facing superiority claim.",
        "- The output provides held-out density rows only; it does not evaluate latent recovery, branch mass, calibration, or scaling.",
        "- Failure and blocker rows remain in the CSVs and summary tables; methods with failure rate above 10 percent are not ranked.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output-dir", dest="output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scenarios", type=parse_csv_list, default=parse_csv_list("snowflake,doraemon,maze,ecg"))
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--gmm-components", type=parse_int_list, default=parse_int_list("2,4,8,16"))
    parser.add_argument("--kde", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oracle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--land-blocker-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kde-jitter", type=float, default=1e-6)
    parser.add_argument("--n-grid", type=int, default=128)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--tube-landmarks", type=int, default=48)
    parser.add_argument("--tube-sigma-scale", type=float, default=1.15)
    parser.add_argument("--tube-min-sigma", type=float, default=1e-3)
    parser.add_argument("--tube-quadrature", type=int, default=12)
    parser.add_argument("--spline-knots", type=int, default=36)
    parser.add_argument("--spline-curve-points", type=int, default=96)
    parser.add_argument("--spline-smoothing", type=float, default=0.03)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-prefix", type=str, default=DEFAULT_FIGURE_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_default = args.report == DEFAULT_REPORT
    figure_default = args.figure_dir == DEFAULT_OUT
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.report = (
        args.output_dir / "department_d_report.md"
        if report_default
        else args.report if args.report.is_absolute() else ROOT / args.report
    )
    args.figure_dir = (
        args.output_dir
        if figure_default
        else args.figure_dir if args.figure_dir.is_absolute() else ROOT / args.figure_dir
    )
    args.command = "PYTHONPATH=src python3 " + " ".join(shlex.quote(part) for part in sys.argv)
    unknown = sorted(set(args.scenarios) - set(FACTORIES))
    if unknown:
        raise ValueError(f"Unknown synthetic scenarios: {unknown}")
    if args.n_train <= 0 or args.n_test <= 0 or args.reps <= 0:
        raise ValueError("--n-train, --n-test, and --reps must be positive.")
    if args.tube_landmarks < 2:
        raise ValueError("--tube-landmarks must be at least 2.")
    if args.tube_quadrature < 2:
        raise ValueError("--tube-quadrature must be at least 2.")
    if args.spline_knots < 4:
        raise ValueError("--spline-knots must be at least 4.")
    if args.spline_curve_points < 2:
        raise ValueError("--spline-curve-points must be at least 2.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.perf_counter()
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for rep in range(1, args.reps + 1):
        for idx, scenario in enumerate(args.scenarios):
            seed = args.seed + 10000 * rep + 101 * idx
            print(f"[geometric-pilot] scenario={scenario} rep={rep} seed={seed}", flush=True)
            dist = FACTORIES[scenario](seed)
            rng = np.random.default_rng(seed + 1009)
            train = np.asarray(dist.rvs(size=args.n_train, random_state=rng), dtype=float)
            test = np.asarray(dist.rvs(size=args.n_test, random_state=rng), dtype=float)
            if args.oracle:
                append_oracle_row(
                    rows,
                    dist,
                    test,
                    scenario=scenario,
                    rep=rep,
                    seed=seed,
                    args=args,
                )
            append_classical_rows(
                rows,
                train,
                test,
                scenario=scenario,
                rep=rep,
                seed=seed,
                args=args,
            )
            append_mst_tube_row(
                rows,
                diagnostics,
                train,
                test,
                scenario=scenario,
                rep=rep,
                seed=seed,
                args=args,
            )
            append_principal_spline_row(
                rows,
                diagnostics,
                train,
                test,
                scenario=scenario,
                rep=rep,
                seed=seed,
                args=args,
            )
            if args.land_blocker_rows:
                append_land_blocker_row(
                    rows,
                    scenario=scenario,
                    rep=rep,
                    n_train=int(train.shape[0]),
                    n_test=int(test.shape[0]),
                    seed=seed,
                )

    summary_rows = summarize_frozen(rows)
    runtime_rows = summarize_runtime_frozen(summary_rows)
    validation_messages = validate_schema(rows, summary_rows, runtime_rows)
    write_csv(args.output_dir / "heldout_log_scores.csv", rows)
    write_csv(args.output_dir / "benchmark_summary.csv", summary_rows)
    write_csv(args.output_dir / "runtime.csv", runtime_rows)
    write_csv(args.output_dir / "tube_baseline_diagnostics.csv", diagnostics)
    write_parameters(args.output_dir / "run_parameters.txt", args)
    write_parameters_json(args.output_dir / "run_parameters.json", args)
    write_land_note(args.output_dir / "land_style_feasibility_note.md", args)
    figure_paths = maybe_plot(args.output_dir, args.figure_dir, args.figure_prefix, summary_rows)
    elapsed_seconds = time.perf_counter() - start_all
    write_report(
        args.report,
        args=args,
        rows=rows,
        summary_rows=summary_rows,
        runtime_rows=runtime_rows,
        diagnostics=diagnostics,
        validation_messages=validation_messages,
        figure_paths=figure_paths,
        elapsed_seconds=elapsed_seconds,
    )
    execution_report = args.output_dir / "execution_report.md"
    if execution_report != args.report:
        write_report(
            execution_report,
            args=args,
            rows=rows,
            summary_rows=summary_rows,
            runtime_rows=runtime_rows,
            diagnostics=diagnostics,
            validation_messages=validation_messages,
            figure_paths=figure_paths,
            elapsed_seconds=elapsed_seconds,
        )
    print(f"Wrote {args.output_dir / 'heldout_log_scores.csv'}")
    print(f"Wrote {args.output_dir / 'benchmark_summary.csv'}")
    print(f"Wrote {args.output_dir / 'runtime.csv'}")
    print(f"Wrote {args.output_dir / 'tube_baseline_diagnostics.csv'}")
    print(f"Wrote {args.output_dir / 'run_parameters.txt'}")
    print(f"Wrote {args.output_dir / 'run_parameters.json'}")
    print(f"Wrote {args.output_dir / 'land_style_feasibility_note.md'}")
    print(f"Wrote {args.report}")
    if execution_report != args.report:
        print(f"Wrote {execution_report}")


if __name__ == "__main__":
    main()
