#!/usr/bin/env python3
"""Estimated-skeleton SFD diagnostics for the JMLR manuscript.

The script intentionally implements a modest fitted SFD: an estimated one-
dimensional skeleton tube plus a full-dimensional Gaussian residual component.
The skeleton can be an MST graph, a principal-spline curve, a diffusion-map
ordered spline, or a simple trajectory-style path extracted from an MST
diameter. The fitted parameters are the tube mass, residual mass, tube width,
and residual Gaussian moments. These diagnostics are not topology recovery
guarantees; they provide an auditable bridge between support-aware baselines
and fitted SFD summaries.
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
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

(ROOT / "tmp" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / "tmp" / "cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "cache"))

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, multivariate_normal

from run_geometric_baseline_pilot import (
    GeometricTubeFit,
    farthest_point_indices,
    fit_geometric_tube,
    fit_principal_spline_tube,
    point_segment_distances,
    spline_support_centers,
)
from run_jmlr_density_benchmark import (
    FACTORIES,
    fit_gaussian,
    fit_gmm_em,
    fit_kde,
    mean_log_score,
    parse_str_list,
)
from sfd_figure_style import SFD_GRID, SFD_INK, SFD_MUTED, set_sfd_matplotlib_style


PROTOCOL_VERSION = "jmlr_estimated_skeleton_sfd_v2"
SYNTHETIC_OUT = ROOT / "outputs" / "jmlr_frozen_benchmark_v1" / "estimated_skeleton_sfd"
PBMC_OUT = ROOT / "outputs" / "jmlr_real_applications" / "qian_stochastic_biology" / "fitted_sfd_demo"
PBMC_EMBEDDING = ROOT / "outputs" / "jmlr_real_applications" / "qian_stochastic_biology" / "pbmc3k_observed_state_embedding.csv"

SFD_METHODS = [
    "Estimated MST-skeleton SFD",
    "Estimated principal-spline SFD",
    "Estimated diffusion-spline SFD",
    "Estimated trajectory-spline SFD",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def command_line() -> str:
    return " ".join(shlex.quote(piece) for piece in sys.argv)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n")


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def se(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def regularized_cov(x: np.ndarray, weights: np.ndarray | None, reg: float) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        mean = np.mean(x, axis=0)
        cov = np.cov(x, rowvar=False)
    else:
        w = np.asarray(weights, dtype=float)
        total = float(np.sum(w))
        if total <= 1e-12:
            mean = np.mean(x, axis=0)
            cov = np.cov(x, rowvar=False)
        else:
            w = w / total
            mean = np.sum(w[:, None] * x, axis=0)
            centered = x - mean
            cov = (centered * w[:, None]).T @ centered
    if np.ndim(cov) == 0:
        cov = np.asarray([[float(cov)]])
    cov = np.asarray(cov, dtype=float) + reg * np.eye(x.shape[1])
    return np.asarray(mean, dtype=float), cov


def score_and_nonfinite(logpdf: np.ndarray) -> tuple[float, int, str]:
    values = np.asarray(logpdf, dtype=float)
    finite = np.isfinite(values)
    nonfinite = int(values.size - np.count_nonzero(finite))
    status = "ok" if nonfinite == 0 else "warning"
    return mean_log_score(values), nonfinite, status


def principal_scores(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    raw = centered @ vt[0]
    span = float(np.max(raw) - np.min(raw))
    if span <= 0 or not np.isfinite(span):
        raise ValueError("Degenerate principal-axis score.")
    return (raw - np.min(raw)) / span


def diffusion_scores(x: np.ndarray, *, seed: int, n_neighbors: int) -> np.ndarray:
    try:
        from sklearn.manifold import SpectralEmbedding

        n_neighbors = min(max(5, n_neighbors), max(5, x.shape[0] - 1))
        emb = SpectralEmbedding(
            n_components=1,
            n_neighbors=n_neighbors,
            affinity="nearest_neighbors",
            random_state=seed,
        ).fit_transform(x)
        raw = emb[:, 0]
    except Exception:
        raw = principal_scores(x)
    span = float(np.max(raw) - np.min(raw))
    if span <= 0 or not np.isfinite(span):
        return principal_scores(x)
    return (raw - np.min(raw)) / span


def mst_diameter_path(landmarks: np.ndarray) -> np.ndarray:
    """Return landmark coordinates along the diameter path of an MST."""
    dist = cdist(landmarks, landmarks)
    mst = minimum_spanning_tree(dist).toarray()
    graph = mst + mst.T
    dmat, predecessors = shortest_path(graph, directed=False, return_predecessors=True)
    finite = np.isfinite(dmat)
    if not np.any(finite):
        raise ValueError("Could not compute a finite MST shortest path.")
    start, end = np.unravel_index(int(np.nanargmax(np.where(finite, dmat, -np.inf))), dmat.shape)
    path = [int(end)]
    current = int(end)
    while current != int(start):
        current = int(predecessors[start, current])
        if current < 0:
            raise ValueError("MST predecessor path is incomplete.")
        path.append(current)
        if len(path) > landmarks.shape[0] + 1:
            raise ValueError("MST predecessor path did not terminate.")
    path.reverse()
    coords = landmarks[np.asarray(path, dtype=int)]
    if coords.shape[0] < 2:
        raise ValueError("Trajectory path needs at least two landmarks.")
    return coords


def projection_scores_to_polyline(x: np.ndarray, curve: np.ndarray) -> np.ndarray:
    starts = curve[:-1]
    ends = curve[1:]
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1)
    valid = lengths > 0
    starts, directions, lengths = starts[valid], directions[valid], lengths[valid]
    if starts.size == 0:
        return principal_scores(x)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    best_dist = np.full(x.shape[0], np.inf)
    best_score = np.zeros(x.shape[0], dtype=float)
    for j, (start, direction, length) in enumerate(zip(starts, directions, lengths)):
        length_sq = float(length * length)
        rel = x - start
        t = np.clip((rel @ direction) / length_sq, 0.0, 1.0)
        projection = start + t[:, None] * direction
        distance = np.linalg.norm(x - projection, axis=1)
        update = distance < best_dist
        best_dist[update] = distance[update]
        best_score[update] = cumulative[j] + t[update] * length
    span = float(np.max(best_score) - np.min(best_score))
    if span <= 0 or not np.isfinite(span):
        return principal_scores(x)
    return (best_score - np.min(best_score)) / span


def trajectory_scores(x: np.ndarray, *, seed: int, n_landmarks: int) -> np.ndarray:
    """Estimate a one-dimensional trajectory score from an MST diameter path."""
    try:
        landmark_idx = farthest_point_indices(x, n_landmarks, seed=seed)
        path = mst_diameter_path(x[landmark_idx])
        return projection_scores_to_polyline(x, path)
    except Exception:
        return principal_scores(x)


def spline_from_scores(
    x: np.ndarray,
    scores: np.ndarray,
    *,
    n_knots: int,
    n_curve_points: int,
    smoothing: float,
    min_sigma: float,
    sigma_scale: float,
    quadrature: int,
) -> GeometricTubeFit:
    center_scores, centers = spline_support_centers(x, scores, n_knots)
    degree = min(3, center_scores.size - 1)
    grid = np.linspace(0.0, 1.0, max(n_curve_points, center_scores.size))
    cols = []
    for dim in range(x.shape[1]):
        scale = float(np.var(centers[:, dim]))
        smooth_budget = max(0.0, smoothing) * center_scores.size * scale
        spline = UnivariateSpline(center_scores, centers[:, dim], k=degree, s=smooth_budget)
        cols.append(spline(grid))
    curve = np.vstack(cols).T
    starts = curve[:-1]
    ends = curve[1:]
    lengths = np.linalg.norm(ends - starts, axis=1)
    valid = lengths > 0
    starts, ends, lengths = starts[valid], ends[valid], lengths[valid]
    if starts.size == 0:
        raise ValueError("Could not construct nondegenerate diffusion-spline tube.")
    residual = point_segment_distances(x, starts, ends)
    residual_sigma = np.sqrt(np.mean(residual * residual) / max(x.shape[1] - 1, 1))
    nearest = np.partition(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2), kth=1, axis=1)[:, 1]
    spacing_sigma = float(np.median(nearest))
    sigma = max(float(sigma_scale * residual_sigma), 0.5 * spacing_sigma, min_sigma)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return GeometricTubeFit(starts, ends, lengths, float(sigma), nodes, weights)


def fit_skeleton(train: np.ndarray, method: str, *, seed: int, args: argparse.Namespace) -> GeometricTubeFit:
    if method == "Estimated MST-skeleton SFD":
        return fit_geometric_tube(
            train,
            n_landmarks=args.tube_landmarks,
            random_state=seed + 17,
            min_sigma=args.min_sigma,
            sigma_scale=args.sigma_scale,
            quadrature=args.quadrature,
        )
    if method == "Estimated principal-spline SFD":
        return fit_principal_spline_tube(
            train,
            n_knots=args.spline_knots,
            n_curve_points=args.spline_curve_points,
            smoothing=args.spline_smoothing,
            min_sigma=args.min_sigma,
            sigma_scale=args.sigma_scale,
            quadrature=args.quadrature,
        )
    if method == "Estimated diffusion-spline SFD":
        scores = diffusion_scores(train, seed=seed + 29, n_neighbors=args.diffusion_neighbors)
        return spline_from_scores(
            train,
            scores,
            n_knots=args.spline_knots,
            n_curve_points=args.spline_curve_points,
            smoothing=args.spline_smoothing,
            min_sigma=args.min_sigma,
            sigma_scale=args.sigma_scale,
            quadrature=args.quadrature,
        )
    if method == "Estimated trajectory-spline SFD":
        scores = trajectory_scores(train, seed=seed + 43, n_landmarks=args.tube_landmarks)
        return spline_from_scores(
            train,
            scores,
            n_knots=args.spline_knots,
            n_curve_points=args.spline_curve_points,
            smoothing=args.spline_smoothing,
            min_sigma=args.min_sigma,
            sigma_scale=args.sigma_scale,
            quadrature=args.quadrature,
        )
    raise ValueError(f"Unknown SFD method {method!r}.")


@dataclass
class FittedSkeletonSFD:
    method: str
    skeleton: GeometricTubeFit
    tube_mass: float
    residual_mass: float
    residual_mean: np.ndarray
    residual_cov: np.ndarray
    iterations: int
    converged: bool
    final_objective: float
    final_max_change: float

    def component_logpdf(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tube = self.skeleton.logpdf(x)
        resid = multivariate_normal.logpdf(x, mean=self.residual_mean, cov=self.residual_cov)
        return np.asarray(tube, dtype=float), np.asarray(resid, dtype=float)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        tube, resid = self.component_logpdf(x)
        return logsumexp(
            np.vstack(
                [
                    np.log(max(self.tube_mass, 1e-300)) + tube,
                    np.log(max(self.residual_mass, 1e-300)) + resid,
                ]
            ).T,
            axis=1,
        )

    def responsibilities(self, x: np.ndarray) -> np.ndarray:
        tube, resid = self.component_logpdf(x)
        logs = np.vstack(
            [
                np.log(max(self.tube_mass, 1e-300)) + tube,
                np.log(max(self.residual_mass, 1e-300)) + resid,
            ]
        ).T
        norm = logsumexp(logs, axis=1, keepdims=True)
        return np.exp(logs - norm)

    def sample(self, n: int, *, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        use_tube = rng.uniform(size=n) < self.tube_mass
        out = np.zeros((n, self.residual_mean.size), dtype=float)
        n_tube = int(np.sum(use_tube))
        if n_tube:
            lengths = np.asarray(self.skeleton.lengths, dtype=float)
            probs = lengths / np.sum(lengths)
            edges = rng.choice(lengths.size, size=n_tube, p=probs)
            t = rng.uniform(size=n_tube)
            starts = self.skeleton.starts[edges]
            ends = self.skeleton.ends[edges]
            centers = starts + t[:, None] * (ends - starts)
            out[use_tube] = centers + rng.normal(scale=self.skeleton.sigma, size=centers.shape)
        n_resid = n - n_tube
        if n_resid:
            out[~use_tube] = rng.multivariate_normal(self.residual_mean, self.residual_cov, size=n_resid)
        return out


def fit_estimated_sfd(train: np.ndarray, method: str, *, seed: int, args: argparse.Namespace) -> FittedSkeletonSFD:
    skeleton = fit_skeleton(train, method, seed=seed, args=args)
    distances = point_segment_distances(train, skeleton.starts, skeleton.ends)
    nearest = np.partition(np.linalg.norm(train[:, None, :] - train[None, :, :], axis=2), kth=1, axis=1)[:, 1]
    floor = max(float(args.min_sigma), 0.5 * float(np.median(nearest)))
    residual_mean, residual_cov = regularized_cov(train, None, args.reg)
    tube_mass = 0.85
    residual_mass = 0.15
    converged = False
    final_objective = float("nan")
    final_change = float("inf")
    for iteration in range(1, args.em_max_iter + 1):
        tube_log = skeleton.logpdf(train)
        resid_log = multivariate_normal.logpdf(train, mean=residual_mean, cov=residual_cov)
        logs = np.vstack(
            [
                np.log(max(tube_mass, args.min_weight)) + tube_log,
                np.log(max(residual_mass, args.min_weight)) + resid_log,
            ]
        ).T
        norm = logsumexp(logs, axis=1)
        resp = np.exp(logs - norm[:, None])
        new_tube_mass = max(float(np.mean(resp[:, 0])), args.min_weight)
        new_resid_mass = max(float(np.mean(resp[:, 1])), args.min_weight)
        total = new_tube_mass + new_resid_mass
        new_tube_mass /= total
        new_resid_mass /= total
        denom = float(np.sum(resp[:, 0]))
        if denom > 1e-12:
            sigma = np.sqrt(float(np.sum(resp[:, 0] * distances * distances)) / (denom * max(train.shape[1] - 1, 1)))
            skeleton.sigma = max(float(args.sigma_scale * sigma), floor)
        residual_mean, residual_cov = regularized_cov(train, resp[:, 1], args.reg)
        final_objective = float(np.mean(norm))
        final_change = max(abs(new_tube_mass - tube_mass), abs(new_resid_mass - residual_mass))
        tube_mass, residual_mass = new_tube_mass, new_resid_mass
        if final_change < args.em_tol:
            converged = True
            break
    return FittedSkeletonSFD(
        method=method,
        skeleton=skeleton,
        tube_mass=tube_mass,
        residual_mass=residual_mass,
        residual_mean=residual_mean,
        residual_cov=residual_cov,
        iterations=iteration,
        converged=converged,
        final_objective=final_objective,
        final_max_change=final_change,
    )


def baseline_rows(train: np.ndarray, test: np.ndarray, *, scenario: str, rep: int, seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [("Gaussian", None), ("KDE", None)] + [(f"GMM-{k}", k) for k in args.gmm_components]
    for method, k in specs:
        start = time.perf_counter()
        try:
            if method == "Gaussian":
                fit = fit_gaussian(train, args.reg)
                hyper = f"reg={args.reg:g}"
            elif method == "KDE":
                fit = fit_kde(train, random_state=seed + 311, jitter=args.kde_jitter)
                hyper = f"bandwidth=scott;jitter={args.kde_jitter:g}"
            else:
                fit = fit_gmm_em(train, int(k), random_state=seed + 17 * int(k), reg=args.reg, max_iter=args.gmm_max_iter, tol=args.gmm_tol)
                hyper = f"K={k};reg={args.reg:g};max_iter={args.gmm_max_iter};tol={args.gmm_tol:g}"
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            score, nonfinite, status = score_and_nonfinite(fit.logpdf(test))
            eval_seconds = time.perf_counter() - start
            rows.append(make_score_row(args, scenario, rep, train.shape[0], test.shape[0], method, score, fit_seconds, eval_seconds, seed, hyper, status, True, "", nonfinite))
        except Exception as exc:
            rows.append(make_score_row(args, scenario, rep, train.shape[0], test.shape[0], method, float("nan"), time.perf_counter() - start, 0.0, seed, "", "failed", False, type(exc).__name__, 0, str(exc)[:500]))
    return rows


def make_score_row(
    args: argparse.Namespace,
    scenario: str,
    rep: int,
    n_train: int,
    n_test: int,
    method: str,
    score: float,
    fit_seconds: float,
    eval_seconds: float,
    seed: int,
    hyperparameter: str,
    status: str,
    converged: Any,
    error_type: str,
    nonfinite_count: int,
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": scenario,
        "scenario_family": args.scenario_family,
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
        "converged": converged,
        "n_iter": "",
        "error_type": error_type,
        "error_message": error_message,
        "nonfinite_count": nonfinite_count,
    }


def summarize_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(r["scenario"], r["method"]) for r in rows})
    for scenario, method in keys:
        sub = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
        scores = np.asarray([as_float(r["heldout_log_score"]) for r in sub], dtype=float)
        fit = np.asarray([as_float(r["fit_seconds"]) for r in sub], dtype=float)
        eval_time = np.asarray([as_float(r["eval_seconds"]) for r in sub], dtype=float)
        failure_rate = sum(r["status"] == "failed" for r in sub) / len(sub)
        warning_rate = sum(r["status"] == "warning" for r in sub) / len(sub)
        out.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": scenario,
                "scenario_family": sub[0]["scenario_family"],
                "method": method,
                "mean_heldout_log_score": finite_mean(scores),
                "se_heldout_log_score": se(scores),
                "rank_within_scenario": "NA",
                "mean_fit_seconds": finite_mean(fit),
                "mean_eval_seconds": finite_mean(eval_time),
                "reps": len(sub),
                "failure_rate": failure_rate,
                "warning_rate": warning_rate,
            }
        )
    for scenario in sorted({r["scenario"] for r in out}):
        sub = [r for r in out if r["scenario"] == scenario and np.isfinite(as_float(r["mean_heldout_log_score"])) and float(r["failure_rate"]) <= 0.1]
        sub.sort(key=lambda r: float(r["mean_heldout_log_score"]), reverse=True)
        for rank, row in enumerate(sub, start=1):
            row["rank_within_scenario"] = rank
    return out


def synthetic_truth_binary(dist, labels: np.ndarray) -> np.ndarray:
    kinds = np.asarray([comp.kind for comp in dist.components], dtype=object)
    return np.where(kinds[labels] == "tube", 1, 0)


def run_synthetic(args: argparse.Namespace) -> None:
    args.scenario_family = "estimated_skeleton_builtin_mixed_support"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    resp_rows: list[dict[str, Any]] = []
    for scen_idx, scenario in enumerate(args.scenarios):
        for rep in range(1, args.reps + 1):
            seed = args.seed + 10000 * rep + 101 * scen_idx
            dist = FACTORIES[scenario](seed)
            rng = np.random.default_rng(seed + 1009)
            train, train_info = dist.rvs(args.n_train, random_state=rng, return_info=True)
            test, test_info = dist.rvs(args.n_test, random_state=rng, return_info=True)
            train = np.asarray(train, dtype=float)
            test = np.asarray(test, dtype=float)
            test_labels = np.asarray(test_info["component_index"], dtype=int)
            true_binary = synthetic_truth_binary(dist, test_labels)
            score_rows.extend(baseline_rows(train, test, scenario=scenario, rep=rep, seed=seed, args=args))
            for method in SFD_METHODS:
                start = time.perf_counter()
                try:
                    fit = fit_estimated_sfd(train, method, seed=seed, args=args)
                    fit_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    logpdf = fit.logpdf(test)
                    resp = fit.responsibilities(test)
                    eval_seconds = time.perf_counter() - start
                    score, nonfinite, status = score_and_nonfinite(logpdf)
                    hyper = (
                        f"edges={fit.skeleton.n_edges};tube_sigma={fit.skeleton.sigma:.6g};"
                        f"tube_mass={fit.tube_mass:.6g};residual_mass={fit.residual_mass:.6g}"
                    )
                    score_rows.append(make_score_row(args, scenario, rep, train.shape[0], test.shape[0], method, score, fit_seconds, eval_seconds, seed, hyper, status, fit.converged, "", nonfinite))
                    pred_binary = (resp[:, 0] >= 0.5).astype(int)
                    fit_rows.append(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "scenario": scenario,
                            "scenario_family": args.scenario_family,
                            "rep": rep,
                            "method": method,
                            "seed": seed,
                            "n_train": train.shape[0],
                            "n_test": test.shape[0],
                            "n_edges": fit.skeleton.n_edges,
                            "total_skeleton_length": fit.skeleton.total_length,
                            "tube_sigma": fit.skeleton.sigma,
                            "tube_mass": fit.tube_mass,
                            "residual_mass": fit.residual_mass,
                            "em_iterations": fit.iterations,
                            "em_converged": fit.converged,
                            "final_objective": fit.final_objective,
                            "final_max_change": fit.final_max_change,
                            "binary_tube_responsibility_accuracy": float(np.mean(pred_binary == true_binary)),
                            "mean_tube_resp_true_tube": float(np.mean(resp[true_binary == 1, 0])) if np.any(true_binary == 1) else float("nan"),
                            "mean_residual_resp_true_nontube": float(np.mean(resp[true_binary == 0, 1])) if np.any(true_binary == 0) else float("nan"),
                            "estimator_scope": "estimated one-dimensional skeleton plus Gaussian residual; not topology recovery validation",
                        }
                    )
                    resp_rows.extend(
                        [
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "scenario": scenario,
                                "scenario_family": args.scenario_family,
                                "rep": rep,
                                "method": method,
                                "seed": seed,
                                "group": "true_tube",
                                "n": int(np.sum(true_binary == 1)),
                                "mean_tube_responsibility": float(np.mean(resp[true_binary == 1, 0])) if np.any(true_binary == 1) else float("nan"),
                                "mean_residual_responsibility": float(np.mean(resp[true_binary == 1, 1])) if np.any(true_binary == 1) else float("nan"),
                            },
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "scenario": scenario,
                                "scenario_family": args.scenario_family,
                                "rep": rep,
                                "method": method,
                                "seed": seed,
                                "group": "true_nontube",
                                "n": int(np.sum(true_binary == 0)),
                                "mean_tube_responsibility": float(np.mean(resp[true_binary == 0, 0])) if np.any(true_binary == 0) else float("nan"),
                                "mean_residual_responsibility": float(np.mean(resp[true_binary == 0, 1])) if np.any(true_binary == 0) else float("nan"),
                            },
                        ]
                    )
                except Exception as exc:
                    score_rows.append(make_score_row(args, scenario, rep, train.shape[0], test.shape[0], method, float("nan"), time.perf_counter() - start, 0.0, seed, "", "failed", False, type(exc).__name__, 0, str(exc)[:500]))
    summary_rows = summarize_scores(score_rows)
    fit_summary = summarize_fit_rows(fit_rows)
    write_csv(out / "heldout_log_scores.csv", score_rows)
    write_csv(out / "benchmark_summary.csv", summary_rows)
    write_csv(out / "estimated_sfd_fit_summary.csv", fit_rows)
    write_csv(out / "estimated_sfd_fit_summary_by_method.csv", fit_summary)
    write_csv(out / "responsibility_summary.csv", resp_rows)
    write_run_parameters(out, args, mode="synthetic")
    plot_synthetic(fig_dir, summary_rows, fit_summary)
    write_report(out / "execution_report.md", args, mode="synthetic", rows=score_rows, extra_rows=fit_rows)


def summarize_fit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(r["scenario"], r["method"]) for r in rows})
    fields = [
        "tube_mass",
        "residual_mass",
        "tube_sigma",
        "n_edges",
        "binary_tube_responsibility_accuracy",
        "mean_tube_resp_true_tube",
        "mean_residual_resp_true_nontube",
        "em_iterations",
    ]
    for scenario, method in keys:
        sub = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
        row = {
            "protocol_version": PROTOCOL_VERSION,
            "scenario": scenario,
            "scenario_family": sub[0]["scenario_family"],
            "method": method,
            "reps": len(sub),
        }
        for field in fields:
            values = np.asarray([as_float(r[field]) for r in sub], dtype=float)
            row[f"mean_{field}"] = finite_mean(values)
            row[f"se_{field}"] = se(values)
        out.append(row)
    return out


def read_pbmc_embedding(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    rows = []
    train = []
    test = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            xy = [float(row["Y1"]), float(row["Y2"])]
            if row["split"] == "train":
                train.append(xy)
            elif row["split"] == "test":
                test.append(xy)
    if not train or not test:
        raise ValueError(f"Need train/test split rows in {path}.")
    return np.asarray(train, dtype=float), np.asarray(test, dtype=float), rows


def run_pbmc3k(args: argparse.Namespace) -> None:
    args.scenario_family = "pbmc3k_observed_state_fitted_sfd_demo"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    train, test, all_rows = read_pbmc_embedding(args.embedding)
    scenario = "pbmc3k_observed_state"
    rep = 1
    seed = args.seed
    score_rows = baseline_rows(train, test, scenario=scenario, rep=rep, seed=seed, args=args)
    fit_rows = []
    responsibility_rows = []
    conditional_rows = []
    conditional_samples = []
    best_fit: FittedSkeletonSFD | None = None
    best_method = ""
    for method in SFD_METHODS:
        start = time.perf_counter()
        try:
            fit = fit_estimated_sfd(train, method, seed=seed, args=args)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            logpdf = fit.logpdf(test)
            resp = fit.responsibilities(test)
            eval_seconds = time.perf_counter() - start
            score, nonfinite, status = score_and_nonfinite(logpdf)
            score_rows.append(make_score_row(args, scenario, rep, train.shape[0], test.shape[0], method, score, fit_seconds, eval_seconds, seed, f"edges={fit.skeleton.n_edges};tube_sigma={fit.skeleton.sigma:.6g};tube_mass={fit.tube_mass:.6g};residual_mass={fit.residual_mass:.6g}", status, fit.converged, "", nonfinite))
            fit_rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "scenario": scenario,
                    "scenario_family": args.scenario_family,
                    "method": method,
                    "seed": seed,
                    "n_train": train.shape[0],
                    "n_test": test.shape[0],
                    "n_edges": fit.skeleton.n_edges,
                    "total_skeleton_length": fit.skeleton.total_length,
                    "tube_sigma": fit.skeleton.sigma,
                    "tube_mass": fit.tube_mass,
                    "residual_mass": fit.residual_mass,
                    "mean_tube_responsibility_test": float(np.mean(resp[:, 0])),
                    "median_tube_responsibility_test": float(np.median(resp[:, 0])),
                    "share_test_tube_resp_ge_0_5": float(np.mean(resp[:, 0] >= 0.5)),
                    "mean_residual_responsibility_test": float(np.mean(resp[:, 1])),
                    "em_iterations": fit.iterations,
                    "em_converged": fit.converged,
                    "final_objective": fit.final_objective,
                    "estimator_scope": "PBMC3k fitted SFD demo on two PCA coordinates; no biological mechanism or transition claim",
                }
            )
            responsibility_rows.extend(
                [
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "scenario": scenario,
                        "method": method,
                        "summary": "mean_tube_responsibility_test",
                        "value": float(np.mean(resp[:, 0])),
                    },
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "scenario": scenario,
                        "method": method,
                        "summary": "share_test_tube_resp_ge_0_5",
                        "value": float(np.mean(resp[:, 0] >= 0.5)),
                    },
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "scenario": scenario,
                        "method": method,
                        "summary": "mean_residual_responsibility_test",
                        "value": float(np.mean(resp[:, 1])),
                    },
                ]
            )
            if best_fit is None or score > max(as_float(r["heldout_log_score"]) for r in score_rows if r["method"] == best_method):
                best_fit = fit
                best_method = method
        except Exception as exc:
            score_rows.append(make_score_row(args, scenario, rep, train.shape[0], test.shape[0], method, float("nan"), time.perf_counter() - start, 0.0, seed, "", "failed", False, type(exc).__name__, 0, str(exc)[:500]))
    if best_fit is not None:
        conditional_rows, conditional_samples = conditional_demo(best_fit, train, method=best_method, seed=seed + 505, n_samples=args.conditional_samples)
    summary_rows = summarize_scores(score_rows)
    write_csv(out / "heldout_log_scores.csv", score_rows)
    write_csv(out / "benchmark_summary.csv", summary_rows)
    write_csv(out / "fitted_sfd_summary.csv", fit_rows)
    write_csv(out / "posterior_responsibility_summary.csv", responsibility_rows)
    if conditional_rows:
        write_csv(out / "conditional_simulation_summary.csv", conditional_rows)
    if conditional_samples:
        write_csv(out / "conditional_simulation_samples.csv", conditional_samples)
    write_run_parameters(out, args, mode="pbmc3k")
    plot_pbmc(fig_dir, train, test, score_rows, fit_rows, conditional_samples, best_method)
    write_report(out / "execution_report.md", args, mode="pbmc3k", rows=score_rows, extra_rows=fit_rows)


def conditional_demo(
    fit: FittedSkeletonSFD,
    train: np.ndarray,
    *,
    method: str,
    seed: int,
    n_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    low, high = np.quantile(train[:, 0], [0.35, 0.65])
    accepted: list[np.ndarray] = []
    rng_seed = seed
    attempts = 0
    while sum(chunk.shape[0] for chunk in accepted) < n_samples and attempts < 50:
        samples = fit.sample(max(1000, 2 * n_samples), seed=rng_seed)
        mask = (samples[:, 0] >= low) & (samples[:, 0] <= high)
        if np.any(mask):
            accepted.append(samples[mask])
        attempts += 1
        rng_seed += 1
    if accepted:
        x = np.vstack(accepted)[:n_samples]
    else:
        x = np.empty((0, train.shape[1]))
    if x.size:
        resp = fit.responsibilities(x)
        summary = [
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": "pbmc3k_observed_state",
                "method": method,
                "condition": "Y1 between training 35th and 65th percentiles",
                "condition_y1_low": float(low),
                "condition_y1_high": float(high),
                "n_conditional_samples": int(x.shape[0]),
                "mean_y1": float(np.mean(x[:, 0])),
                "mean_y2": float(np.mean(x[:, 1])),
                "tube_mass_among_conditional_samples": float(np.mean(resp[:, 0])),
                "residual_mass_among_conditional_samples": float(np.mean(resp[:, 1])),
            }
        ]
        sample_rows = [
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": "pbmc3k_observed_state",
                "method": method,
                "sample_index": i,
                "Y1": float(row[0]),
                "Y2": float(row[1]),
                "posterior_tube_responsibility": float(resp[i, 0]),
                "posterior_residual_responsibility": float(resp[i, 1]),
                "condition": "Y1 between training 35th and 65th percentiles",
            }
            for i, row in enumerate(x)
        ]
        return summary, sample_rows
    return [], []


def write_run_parameters(out: Path, args: argparse.Namespace, *, mode: str) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "created_at_utc": utc_now(),
        "command": command_line(),
        "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "claim_boundary": "Estimated-skeleton SFD diagnostic; no topology-learning theorem, no learned-manifold density claim, no broad superiority claim.",
    }
    write_json(out / "run_parameters.json", payload)
    text = "\n".join(f"{k}: {v}" for k, v in payload.items()) + "\n"
    (out / "run_parameters.txt").write_text(text)


def write_report(path: Path, args: argparse.Namespace, *, mode: str, rows: list[dict[str, Any]], extra_rows: list[dict[str, Any]]) -> None:
    failed = sum(r["status"] == "failed" for r in rows)
    warning = sum(r["status"] == "warning" for r in rows)
    ok = sum(r["status"] == "ok" for r in rows)
    path.write_text(
        "\n".join(
            [
                f"# Estimated-Skeleton SFD {mode} execution report",
                "",
                f"- protocol_version: {PROTOCOL_VERSION}",
                f"- created_at_utc: {utc_now()}",
                f"- command: `{command_line()}`",
                f"- score_rows: {len(rows)}",
                f"- ok_rows: {ok}",
                f"- warning_rows: {warning}",
                f"- failed_rows: {failed}",
                f"- sfd_summary_rows: {len(extra_rows)}",
                "- claim_boundary: estimated skeleton plus residual diagnostics only; not a learned-topology guarantee or broad method ranking.",
                "",
            ]
        )
    )


def plot_synthetic(fig_dir: Path, summary_rows: list[dict[str, Any]], fit_summary: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    set_sfd_matplotlib_style(plt, base_size=9.8, title_size=10.8, label_size=9.6, tick_size=8.4)
    sfd_rows = [r for r in summary_rows if r["method"] in SFD_METHODS]
    scenarios = ["snowflake", "stylized_face", "doraemon", "maze", "ecg", "torus"]
    scenarios = [s for s in scenarios if any(r["scenario"] == s for r in sfd_rows)]
    labels = {"stylized_face": "Stylized face", "doraemon": "Stylized face", "snowflake": "Snowflake", "maze": "Maze", "ecg": "ECG", "torus": "Torus"}
    colors = ["#62B6CB", "#7FD1B9", "#F2C76E", "#B7A6E4"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1))
    ax = axes[0]
    width = min(0.18, 0.8 / max(len(SFD_METHODS), 1))
    x = np.arange(len(scenarios))
    for j, method in enumerate(SFD_METHODS):
        means = []
        ses = []
        for scenario in scenarios:
            match = [r for r in sfd_rows if r["scenario"] == scenario and r["method"] == method]
            means.append(as_float(match[0]["mean_heldout_log_score"]) if match else np.nan)
            ses.append(as_float(match[0]["se_heldout_log_score"]) if match else np.nan)
        offset = (j - (len(SFD_METHODS) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=ses, label=method.replace("Estimated ", ""), color=colors[j], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in scenarios], rotation=20, ha="right")
    ax.set_ylabel("Held-out log score")
    ax.set_title("Estimated-skeleton SFD score")
    ax.grid(axis="y", color=SFD_GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    ax = axes[1]
    for j, method in enumerate(SFD_METHODS):
        means = []
        for scenario in scenarios:
            match = [r for r in fit_summary if r["scenario"] == scenario and r["method"] == method]
            means.append(as_float(match[0]["mean_residual_mass"]) if match else np.nan)
        ax.plot(x, means, marker="o", label=method.replace("Estimated ", ""), color=colors[j], linewidth=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in scenarios], rotation=20, ha="right")
    ax.set_ylabel("Estimated residual mass")
    ax.set_title("Mass assigned outside the skeleton tube")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=SFD_GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"estimated_skeleton_sfd_diagnostic.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_pbmc(
    fig_dir: Path,
    train: np.ndarray,
    test: np.ndarray,
    score_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    conditional_samples: list[dict[str, Any]],
    best_method: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    set_sfd_matplotlib_style(plt, base_size=9.4, title_size=10.4, label_size=9.2, tick_size=8.0)
    train_color = "#6FB6DA"
    test_color = "#F29A7A"
    baseline_bar = "#CDE4F0"
    sfd_bar = "#8BD7BF"
    sfd_bar_dark = "#4FAF9C"
    resp_cmap = LinearSegmentedColormap.from_list(
        "sfd_fresh_responsibility",
        ["#5F93D9", "#55C8D3", "#92DDBE", "#F4D76E", "#F2996D"],
    )
    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    ax = axes[0]
    ax.scatter(train[:, 0], train[:, 1], s=9, alpha=0.34, color=train_color, linewidths=0, label="train")
    ax.scatter(test[:, 0], test[:, 1], s=12, alpha=0.52, color=test_color, linewidths=0, label="test")
    ax.set_title("PBMC3k observed state")
    ax.set_xlabel("Y1")
    ax.set_ylabel("Y2")
    legend = ax.legend(frameon=False, fontsize=7.2)
    for handle in legend.legend_handles:
        handle.set_sizes([22])
    ax = axes[1]
    sub = [r for r in score_rows if r["status"] != "failed"]
    sub.sort(key=lambda r: as_float(r["heldout_log_score"]), reverse=True)
    methods = [r["method"].replace("Estimated ", "").replace(" SFD", "") for r in sub]
    scores = [as_float(r["heldout_log_score"]) for r in sub]
    colors = [
        sfd_bar_dark if r["method"] == best_method else sfd_bar
        if "skeleton" in r["method"] or "spline" in r["method"]
        else baseline_bar
        for r in sub
    ]
    ax.barh(np.arange(len(sub)), scores, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(methods, fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out log score")
    ax.set_title("One-split density score")
    ax.grid(axis="x", color=SFD_GRID, linewidth=0.6)
    ax = axes[2]
    if conditional_samples:
        xy = np.asarray([[float(r["Y1"]), float(r["Y2"])] for r in conditional_samples], dtype=float)
        c = np.asarray([float(r["posterior_tube_responsibility"]) for r in conditional_samples], dtype=float)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=resp_cmap, s=12, alpha=0.72, linewidths=0)
        fig.colorbar(sc, ax=ax, fraction=0.026, pad=0.02, label="tube resp.")
    ax.set_title("Conditional simulation")
    ax.set_xlabel("Y1")
    ax.set_ylabel("Y2")
    title = best_method.replace("Estimated ", "") if best_method else "fitted SFD"
    fig.suptitle(f"PBMC3k fitted-SFD demo ({title})", fontsize=11.2, color=SFD_INK)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"pbmc3k_fitted_sfd_demo.{ext}", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=20260625)
    common.add_argument("--tube-landmarks", type=int, default=48)
    common.add_argument("--spline-knots", type=int, default=36)
    common.add_argument("--spline-curve-points", type=int, default=96)
    common.add_argument("--spline-smoothing", type=float, default=0.001)
    common.add_argument("--diffusion-neighbors", type=int, default=20)
    common.add_argument("--quadrature", type=int, default=12)
    common.add_argument("--sigma-scale", type=float, default=1.0)
    common.add_argument("--min-sigma", type=float, default=1e-3)
    common.add_argument("--reg", type=float, default=1e-5)
    common.add_argument("--min-weight", type=float, default=1e-4)
    common.add_argument("--em-max-iter", type=int, default=80)
    common.add_argument("--em-tol", type=float, default=1e-6)
    common.add_argument("--gmm-components", type=lambda s: [int(x) for x in s.split(",") if x], default=[2, 4, 8])
    common.add_argument("--gmm-max-iter", type=int, default=160)
    common.add_argument("--gmm-tol", type=float, default=1e-6)
    common.add_argument("--kde-jitter", type=float, default=1e-6)

    syn = sub.add_parser("synthetic", parents=[common])
    syn.add_argument("--scenarios", type=parse_str_list, default=["snowflake", "stylized_face", "maze", "ecg"])
    syn.add_argument("--n-train", type=int, default=800)
    syn.add_argument("--n-test", type=int, default=800)
    syn.add_argument("--reps", type=int, default=5)
    syn.add_argument("--out", type=Path, default=SYNTHETIC_OUT)

    pbmc = sub.add_parser("pbmc3k", parents=[common])
    pbmc.add_argument("--embedding", type=Path, default=PBMC_EMBEDDING)
    pbmc.add_argument("--conditional-samples", type=int, default=600)
    pbmc.add_argument("--out", type=Path, default=PBMC_OUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    elif args.mode == "pbmc3k":
        run_pbmc3k(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
