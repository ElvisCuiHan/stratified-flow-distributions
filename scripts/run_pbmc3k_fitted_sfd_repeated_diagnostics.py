#!/usr/bin/env python3
"""Repeated PBMC3k fitted-SFD diagnostics for the JMLR manuscript.

This script reuses the cached two-dimensional PBMC3k observed-state embedding
and the estimated-skeleton SFD fitter.  It records repeated train/test split
summaries for SFD-specific quantities that ordinary ambient density rows do not
define: tube mass, residual mass, posterior tube responsibility, conditional
simulation under PC1 events, and simple responsibility sanity checks.

The marker-population output uses predeclared marker-gene proxies from the
cached 10x count matrix.  It is a coarse sanity check, not an official
cell-type annotation or biological discovery analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
import time
from collections import defaultdict
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
from scipy.special import logsumexp
from scipy.stats import kruskal, multivariate_normal, spearmanr

from run_estimated_skeleton_sfd_diagnostics import (
    PBMC_EMBEDDING,
    SFD_METHODS,
    FittedSkeletonSFD,
    as_float,
    fit_estimated_sfd,
    fit_kde,
    se,
    write_csv,
)
from run_jmlr_density_benchmark import fit_gmm_em
from run_qian_real_stochastic_biology_application import read_10x_tarball


PROTOCOL_VERSION = "jmlr_pbmc3k_fitted_sfd_repeated_v1"
SCENARIO = "pbmc3k_observed_state"
SCENARIO_FAMILY = "pbmc3k_observed_state_fitted_sfd_repeated"
DEFAULT_OUT = (
    ROOT
    / "outputs"
    / "jmlr_real_applications"
    / "qian_stochastic_biology"
    / "fitted_sfd_repeated_splits"
)
DEFAULT_RAW_TARBALL = (
    ROOT
    / "outputs"
    / "jmlr_real_applications"
    / "qian_stochastic_biology"
    / "raw"
    / "pbmc3k_filtered_gene_bc_matrices.tar.gz"
)
CLAIM_BOUNDARY = (
    "Static fitted-SFD diagnostics on two PBMC3k PCA observed-state coordinates; "
    "no cell-type discovery, biological mechanism, transition-rate, probability-current, "
    "entropy-production, topology-learning, or broad method-superiority claim."
)
MARKER_SETS = {
    "T_NK_marker_proxy": ["CD3D", "CD3E", "IL7R", "NKG7", "GNLY"],
    "B_marker_proxy": ["MS4A1", "CD79A", "CD79B"],
    "myeloid_marker_proxy": ["LYZ", "S100A9", "CST3", "CD14", "FCGR3A", "LST1"],
    "dendritic_marker_proxy": ["FCER1A", "HLA-DRA"],
    "platelet_marker_proxy": ["PPBP", "PF4"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def command_line() -> str:
    return " ".join(shlex.quote(piece) for piece in sys.argv)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def read_embedding(path: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    xy: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            xy.append([float(row["Y1"]), float(row["Y2"])])
    if not rows:
        raise ValueError(f"No embedding rows in {path}.")
    return np.asarray(xy, dtype=float), rows


def split_mask(n: int, train_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    n_train = int(round(train_fraction * n))
    mask = np.zeros(n, dtype=bool)
    mask[order[:n_train]] = True
    return mask


def parse_methods(text: str) -> list[str]:
    aliases = {
        "mst": "Estimated MST-skeleton SFD",
        "estimated mst-skeleton sfd": "Estimated MST-skeleton SFD",
        "principal": "Estimated principal-spline SFD",
        "principal-spline": "Estimated principal-spline SFD",
        "diffusion": "Estimated diffusion-spline SFD",
        "diffusion-spline": "Estimated diffusion-spline SFD",
        "trajectory": "Estimated trajectory-spline SFD",
        "trajectory-spline": "Estimated trajectory-spline SFD",
    }
    out: list[str] = []
    for piece in text.split(","):
        key = piece.strip().lower()
        if not key:
            continue
        method = aliases.get(key, piece.strip())
        if method not in SFD_METHODS:
            raise ValueError(f"Unknown SFD method {piece!r}.")
        if method not in out:
            out.append(method)
    if not out:
        raise ValueError("At least one SFD method is required.")
    return out


def score_row(
    *,
    method: str,
    split_rep: int,
    split_seed: int,
    n_train: int,
    n_test: int,
    heldout_log_score: float,
    fit_seconds: float,
    eval_seconds: float,
    status: str,
    converged: bool,
    n_iter: int,
    hyperparameter: str,
    error_type: str = "",
    error_message: str = "",
    nonfinite_count: int = 0,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": SCENARIO,
        "scenario_family": SCENARIO_FAMILY,
        "split_rep": split_rep,
        "split_seed": split_seed,
        "n_train": n_train,
        "n_test": n_test,
        "method": method,
        "heldout_log_score": heldout_log_score,
        "fit_seconds": fit_seconds,
        "eval_seconds": eval_seconds,
        "status": status,
        "converged": converged,
        "n_iter": n_iter,
        "hyperparameter": hyperparameter,
        "error_type": error_type,
        "error_message": error_message,
        "nonfinite_count": nonfinite_count,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def summarize_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        sub = [row for row in rows if row["method"] == method]
        scores = [as_float(row["heldout_log_score"]) for row in sub]
        fit_seconds = [as_float(row["fit_seconds"]) for row in sub]
        eval_seconds = [as_float(row["eval_seconds"]) for row in sub]
        out.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": SCENARIO,
                "scenario_family": SCENARIO_FAMILY,
                "method": method,
                "mean_heldout_log_score": finite_mean(scores),
                "se_heldout_log_score": se(np.asarray(scores, dtype=float)),
                "mean_fit_seconds": finite_mean(fit_seconds),
                "mean_eval_seconds": finite_mean(eval_seconds),
                "split_reps": len(sub),
                "failure_rate": float(np.mean([row["status"] == "failed" for row in sub])),
                "warning_rate": float(np.mean([row["status"] == "warning" for row in sub])),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    out.sort(key=lambda row: row["mean_heldout_log_score"], reverse=True)
    for rank, row in enumerate(out, start=1):
        row["rank_within_sfd_rows"] = rank
    return out


def summarize_fit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "tube_mass",
        "residual_mass",
        "mean_tube_responsibility_test",
        "mean_residual_responsibility_test",
        "share_test_tube_resp_ge_0_5",
        "tube_sigma",
        "n_edges",
        "em_iterations",
    ]
    out: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        sub = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "scenario": SCENARIO,
            "scenario_family": SCENARIO_FAMILY,
            "method": method,
            "split_reps": len(sub),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for field in fields:
            values = np.asarray([as_float(row[field]) for row in sub], dtype=float)
            item[f"mean_{field}"] = finite_mean(values.tolist())
            item[f"se_{field}"] = se(values)
        out.append(item)
    return out


def fit_row(
    fit: FittedSkeletonSFD,
    resp: np.ndarray,
    *,
    split_rep: int,
    split_seed: int,
    n_train: int,
    n_test: int,
    score: float,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": SCENARIO,
        "scenario_family": SCENARIO_FAMILY,
        "split_rep": split_rep,
        "split_seed": split_seed,
        "method": fit.method,
        "n_train": n_train,
        "n_test": n_test,
        "heldout_log_score": score,
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
        "final_max_change": fit.final_max_change,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def gmm_responsibilities(fit: Any, x: np.ndarray) -> np.ndarray:
    cols = []
    for weight, mean, cov in zip(fit.weights, fit.means, fit.covs):
        cols.append(np.log(max(float(weight), 1e-300)) + multivariate_normal.logpdf(x, mean=mean, cov=cov))
    logs = np.vstack(cols).T
    norm = logsumexp(logs, axis=1, keepdims=True)
    return np.exp(logs - norm)


def rank_quantile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    vals = np.asarray(values, dtype=float)
    return np.searchsorted(ref, vals, side="right") / max(ref.size, 1)


def density_proxy_comparison(
    *,
    train: np.ndarray,
    test: np.ndarray,
    tube_resp: np.ndarray,
    split_seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gmm = fit_gmm_em(
        train,
        8,
        random_state=split_seed + 8 * 17,
        reg=args.reg,
        max_iter=args.gmm_max_iter,
        tol=args.gmm_tol,
    )
    gmm_resp = gmm_responsibilities(gmm, test)
    gmm_entropy = -np.sum(gmm_resp * np.log(np.maximum(gmm_resp, 1e-300)), axis=1) / np.log(gmm_resp.shape[1])
    kde = fit_kde(train, random_state=split_seed + 311, jitter=args.kde_jitter)
    kde_train_log = kde.logpdf(train)
    kde_test_log = kde.logpdf(test)
    kde_quant = rank_quantile(kde_train_log, kde_test_log)
    residual_resp = 1.0 - tube_resp

    comparisons = [
        ("tube_resp_vs_gmm8_entropy", tube_resp, gmm_entropy),
        ("tube_resp_vs_kde_density_quantile", tube_resp, kde_quant),
        ("residual_resp_vs_gmm8_entropy", residual_resp, gmm_entropy),
        ("residual_resp_vs_kde_density_quantile", residual_resp, kde_quant),
    ]
    summary_rows: list[dict[str, Any]] = []
    for name, left, right in comparisons:
        corr = spearmanr(left, right, nan_policy="omit")
        summary_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": SCENARIO,
                "split_seed": split_seed,
                "comparison": name,
                "spearman_rho": float(corr.statistic),
                "p_value": float(corr.pvalue),
                "n_test": int(test.shape[0]),
                "interpretation": (
                    "Low or incomplete rank correlation supports treating SFD posterior responsibility "
                    "as a model-based stratum summary rather than a monotone re-labeling of ambient density height."
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    detail_rows = []
    for i in range(test.shape[0]):
        detail_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": SCENARIO,
                "split_seed": split_seed,
                "test_index": i,
                "tube_responsibility": float(tube_resp[i]),
                "residual_responsibility": float(residual_resp[i]),
                "gmm8_posterior_entropy": float(gmm_entropy[i]),
                "kde_log_density": float(kde_test_log[i]),
                "kde_density_quantile_vs_train": float(kde_quant[i]),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return summary_rows, detail_rows


def marker_proxy_labels(
    embedding_rows: list[dict[str, str]],
    *,
    raw_tarball: Path,
    target_sum: float,
    min_score: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    tenx = read_10x_tarball(raw_tarball)
    barcode_to_idx = {barcode: i for i, barcode in enumerate(tenx.barcodes)}
    selected = []
    for row in embedding_rows:
        barcode = row["cell_barcode"]
        if barcode not in barcode_to_idx:
            raise ValueError(f"Could not find barcode {barcode!r} in cached 10x matrix.")
        selected.append(barcode_to_idx[barcode])
    counts = tenx.matrix[np.asarray(selected, dtype=int)].tocsr()
    library_size = np.asarray(counts.sum(axis=1)).ravel()
    symbol_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, symbol in enumerate(tenx.gene_symbols):
        symbol_to_indices[symbol.upper()].append(idx)

    group_scores: dict[str, np.ndarray] = {}
    available_markers: dict[str, list[str]] = {}
    for group, markers in MARKER_SETS.items():
        indices = []
        present = []
        for marker in markers:
            matches = symbol_to_indices.get(marker.upper(), [])
            if matches:
                indices.extend(matches)
                present.append(marker)
        available_markers[group] = present
        if indices:
            values = counts[:, indices].toarray()
            normalized = values * (target_sum / np.maximum(library_size, 1.0))[:, None]
            score = np.mean(np.log1p(normalized), axis=1)
        else:
            score = np.zeros(counts.shape[0], dtype=float)
        group_scores[group] = score

    labels: dict[str, dict[str, Any]] = {}
    groups = list(MARKER_SETS)
    score_matrix = np.vstack([group_scores[group] for group in groups]).T
    best_idx = np.argmax(score_matrix, axis=1)
    best_scores = score_matrix[np.arange(score_matrix.shape[0]), best_idx]
    for i, row in enumerate(embedding_rows):
        label = groups[int(best_idx[i])] if best_scores[i] >= min_score else "unassigned_marker_proxy"
        labels[row["sample_id"]] = {
            "marker_proxy": label,
            "marker_proxy_score": float(best_scores[i]),
            **{f"score_{group}": float(group_scores[group][i]) for group in groups},
        }
    return labels, available_markers


def marker_association_summary(
    *,
    embedding_rows: list[dict[str, str]],
    test_indices: np.ndarray,
    tube_resp: np.ndarray,
    marker_labels: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_values: dict[str, list[float]] = defaultdict(list)
    group_resid: dict[str, list[float]] = defaultdict(list)
    group_y1: dict[str, list[float]] = defaultdict(list)
    group_y2: dict[str, list[float]] = defaultdict(list)
    for local_i, row_index in enumerate(test_indices):
        row = embedding_rows[int(row_index)]
        label = str(marker_labels[row["sample_id"]]["marker_proxy"])
        group_values[label].append(float(tube_resp[local_i]))
        group_resid[label].append(float(1.0 - tube_resp[local_i]))
        group_y1[label].append(float(row["Y1"]))
        group_y2[label].append(float(row["Y2"]))

    rows = []
    for label in sorted(group_values):
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": SCENARIO,
                "annotation_source": "predeclared marker-gene proxy from cached 10x counts; not an official cell-type label",
                "marker_proxy": label,
                "n_test": len(group_values[label]),
                "mean_tube_responsibility": finite_mean(group_values[label]),
                "se_tube_responsibility": se(np.asarray(group_values[label], dtype=float)),
                "mean_residual_responsibility": finite_mean(group_resid[label]),
                "mean_y1": finite_mean(group_y1[label]),
                "mean_y2": finite_mean(group_y2[label]),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    eligible = [np.asarray(values, dtype=float) for values in group_values.values() if len(values) >= 5]
    test_rows = []
    if len(eligible) >= 2:
        stat = kruskal(*eligible)
        test_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": SCENARIO,
                "test": "kruskal_wallis_tube_responsibility_by_marker_proxy",
                "statistic": float(stat.statistic),
                "p_value": float(stat.pvalue),
                "groups_with_n_ge_5": len(eligible),
                "interpretation": (
                    "Association check between SFD posterior tube responsibility and predeclared "
                    "coarse marker proxies; this is a sanity check, not biological validation."
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows, test_rows


def conditional_event_summary(
    fit: FittedSkeletonSFD,
    train: np.ndarray,
    *,
    split_seed: int,
    n_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = [
        ("low PC1", 0.10, 0.30),
        ("middle PC1", 0.35, 0.65),
        ("high PC1", 0.70, 0.90),
    ]
    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for event_index, (label, q_low, q_high) in enumerate(events):
        low, high = np.quantile(train[:, 0], [q_low, q_high])
        accepted: list[np.ndarray] = []
        rng_seed = split_seed + 505 + 1000 * event_index
        attempts = 0
        while sum(chunk.shape[0] for chunk in accepted) < n_samples and attempts < 80:
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
        if x.size == 0:
            continue
        resp = fit.responsibilities(x)
        summary_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scenario": SCENARIO,
                "method": fit.method,
                "condition": label,
                "condition_y1_low": float(low),
                "condition_y1_high": float(high),
                "condition_quantile_low": q_low,
                "condition_quantile_high": q_high,
                "n_conditional_samples": int(x.shape[0]),
                "mean_y1": float(np.mean(x[:, 0])),
                "mean_y2": float(np.mean(x[:, 1])),
                "mean_tube_responsibility": float(np.mean(resp[:, 0])),
                "mean_residual_responsibility": float(np.mean(resp[:, 1])),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        for i, row in enumerate(x):
            sample_rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "scenario": SCENARIO,
                    "method": fit.method,
                    "condition": label,
                    "sample_index": i,
                    "Y1": float(row[0]),
                    "Y2": float(row[1]),
                    "posterior_tube_responsibility": float(resp[i, 0]),
                    "posterior_residual_responsibility": float(resp[i, 1]),
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return summary_rows, sample_rows


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# PBMC3k Repeated Fitted-SFD Diagnostics",
        "",
        f"- status: `completed`",
        f"- protocol_version: `{PROTOCOL_VERSION}`",
        f"- command: `{command_line()}`",
        f"- created_at_utc: `{payload['created_at_utc']}`",
        f"- split_reps: `{payload['split_reps']}`",
        f"- methods: `{', '.join(payload['methods'])}`",
        f"- output_dir: `{payload['output_dir']}`",
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Notes",
        "",
        "- Repeated rows estimate fitted-SFD quantities from the cached two-coordinate PBMC3k embedding.",
        "- Marker-proxy rows use predeclared PBMC marker genes from the cached 10x count matrix; they are not official cell-type annotations.",
        "- GMM-entropy and KDE-quantile comparisons are diagnostic checks that SFD responsibility is not merely ambient density height.",
        "",
        "## Output Files",
        "",
    ]
    for label, value in payload["files"].items():
        lines.append(f"- `{label}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", type=Path, default=PBMC_EMBEDDING)
    parser.add_argument("--raw-tarball", type=Path, default=DEFAULT_RAW_TARBALL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--split-reps", type=int, default=25)
    parser.add_argument("--split-seed-step", type=int, default=10000)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--methods", default="mst")
    parser.add_argument("--primary-method", default="Estimated MST-skeleton SFD")
    parser.add_argument("--conditional-samples", type=int, default=600)
    parser.add_argument("--target-sum", type=float, default=10000.0)
    parser.add_argument("--min-marker-score", type=float, default=0.20)
    parser.add_argument("--tube-landmarks", type=int, default=48)
    parser.add_argument("--spline-knots", type=int, default=36)
    parser.add_argument("--spline-curve-points", type=int, default=96)
    parser.add_argument("--spline-smoothing", type=float, default=0.001)
    parser.add_argument("--diffusion-neighbors", type=int, default=20)
    parser.add_argument("--quadrature", type=int, default=12)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--min-sigma", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--min-weight", type=float, default=1e-4)
    parser.add_argument("--em-max-iter", type=int, default=80)
    parser.add_argument("--em-tol", type=float, default=1e-6)
    parser.add_argument("--gmm-max-iter", type=int, default=160)
    parser.add_argument("--gmm-tol", type=float, default=1e-6)
    parser.add_argument("--kde-jitter", type=float, default=1e-6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.embedding.is_absolute():
        args.embedding = ROOT / args.embedding
    if not args.raw_tarball.is_absolute():
        args.raw_tarball = ROOT / args.raw_tarball
    if not args.out.is_absolute():
        args.out = ROOT / args.out
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")
    if args.split_reps < 1:
        raise ValueError("--split-reps must be positive.")
    methods = parse_methods(args.methods)
    primary = args.primary_method
    if primary.lower() == "mst":
        primary = "Estimated MST-skeleton SFD"
    if primary not in methods:
        primary = methods[0]
    args.scenario_family = SCENARIO_FAMILY
    args.out.mkdir(parents=True, exist_ok=True)

    xy, embedding_rows = read_embedding(args.embedding)
    marker_labels: dict[str, dict[str, Any]] = {}
    available_markers: dict[str, list[str]] = {}
    if args.raw_tarball.exists():
        marker_labels, available_markers = marker_proxy_labels(
            embedding_rows,
            raw_tarball=args.raw_tarball,
            target_sum=args.target_sum,
            min_score=args.min_marker_score,
        )

    score_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    conditional_sample_rows: list[dict[str, Any]] = []
    density_proxy_summary: list[dict[str, Any]] = []
    density_proxy_detail: list[dict[str, Any]] = []
    marker_group_rows: list[dict[str, Any]] = []
    marker_test_rows: list[dict[str, Any]] = []

    primary_first_fit: FittedSkeletonSFD | None = None
    primary_first_train: np.ndarray | None = None
    primary_first_test: np.ndarray | None = None
    primary_first_test_indices: np.ndarray | None = None
    primary_first_resp: np.ndarray | None = None
    primary_first_seed = args.seed + args.split_seed_step

    for split_rep in range(1, args.split_reps + 1):
        split_seed = args.seed + args.split_seed_step * split_rep
        train_mask = split_mask(xy.shape[0], args.train_fraction, split_seed)
        train = xy[train_mask]
        test = xy[~train_mask]
        test_indices = np.flatnonzero(~train_mask)
        for method in methods:
            started = time.perf_counter()
            try:
                fit = fit_estimated_sfd(train, method, seed=split_seed, args=args)
                fit_seconds = time.perf_counter() - started
                eval_started = time.perf_counter()
                logpdf = fit.logpdf(test)
                resp = fit.responsibilities(test)
                eval_seconds = time.perf_counter() - eval_started
                finite = np.isfinite(logpdf)
                nonfinite = int(logpdf.size - int(np.sum(finite)))
                status = "ok" if nonfinite == 0 else "warning"
                score = float(np.mean(logpdf[finite])) if np.any(finite) else float("nan")
                score_rows.append(
                    score_row(
                        method=method,
                        split_rep=split_rep,
                        split_seed=split_seed,
                        n_train=train.shape[0],
                        n_test=test.shape[0],
                        heldout_log_score=score,
                        fit_seconds=fit_seconds,
                        eval_seconds=eval_seconds,
                        status=status,
                        converged=fit.converged,
                        n_iter=fit.iterations,
                        hyperparameter=(
                            f"edges={fit.skeleton.n_edges};tube_sigma={fit.skeleton.sigma:.6g};"
                            f"tube_mass={fit.tube_mass:.6g};residual_mass={fit.residual_mass:.6g}"
                        ),
                        nonfinite_count=nonfinite,
                    )
                )
                fit_rows.append(
                    fit_row(
                        fit,
                        resp,
                        split_rep=split_rep,
                        split_seed=split_seed,
                        n_train=train.shape[0],
                        n_test=test.shape[0],
                        score=score,
                    )
                )
                if split_rep == 1 and method == primary:
                    primary_first_fit = fit
                    primary_first_train = train
                    primary_first_test = test
                    primary_first_test_indices = test_indices
                    primary_first_resp = resp
                    primary_first_seed = split_seed
            except Exception as exc:
                score_rows.append(
                    score_row(
                        method=method,
                        split_rep=split_rep,
                        split_seed=split_seed,
                        n_train=train.shape[0],
                        n_test=test.shape[0],
                        heldout_log_score=float("nan"),
                        fit_seconds=time.perf_counter() - started,
                        eval_seconds=0.0,
                        status="failed",
                        converged=False,
                        n_iter=0,
                        hyperparameter="",
                        error_type=type(exc).__name__,
                        error_message=" ".join(str(exc).split())[:500],
                    )
                )

    if primary_first_fit is not None and primary_first_train is not None:
        conditional_rows, conditional_sample_rows = conditional_event_summary(
            primary_first_fit,
            primary_first_train,
            split_seed=primary_first_seed,
            n_samples=args.conditional_samples,
        )
    if primary_first_train is not None and primary_first_test is not None and primary_first_resp is not None:
        density_proxy_summary, density_proxy_detail = density_proxy_comparison(
            train=primary_first_train,
            test=primary_first_test,
            tube_resp=primary_first_resp[:, 0],
            split_seed=primary_first_seed,
            args=args,
        )
    if (
        marker_labels
        and primary_first_test_indices is not None
        and primary_first_resp is not None
    ):
        marker_group_rows, marker_test_rows = marker_association_summary(
            embedding_rows=embedding_rows,
            test_indices=primary_first_test_indices,
            tube_resp=primary_first_resp[:, 0],
            marker_labels=marker_labels,
        )

    score_summary = summarize_score_rows(score_rows)
    fit_summary = summarize_fit_rows(fit_rows)
    files = {
        "heldout_log_scores": args.out / "heldout_log_scores.csv",
        "benchmark_summary": args.out / "benchmark_summary.csv",
        "fitted_sfd_summary": args.out / "fitted_sfd_summary.csv",
        "fitted_sfd_summary_by_method": args.out / "fitted_sfd_summary_by_method.csv",
        "conditional_simulation_summary": args.out / "conditional_simulation_summary.csv",
        "conditional_simulation_samples": args.out / "conditional_simulation_samples.csv",
        "density_proxy_comparison": args.out / "density_proxy_comparison.csv",
        "density_proxy_cell_diagnostics": args.out / "density_proxy_cell_diagnostics.csv",
        "marker_proxy_responsibility_summary": args.out / "marker_proxy_responsibility_summary.csv",
        "marker_proxy_association_tests": args.out / "marker_proxy_association_tests.csv",
        "run_parameters": args.out / "run_parameters.json",
        "execution_report": args.out / "execution_report.md",
    }
    write_csv(files["heldout_log_scores"], score_rows)
    write_csv(files["benchmark_summary"], score_summary)
    write_csv(files["fitted_sfd_summary"], fit_rows)
    write_csv(files["fitted_sfd_summary_by_method"], fit_summary)
    if conditional_rows:
        write_csv(files["conditional_simulation_summary"], conditional_rows)
    if conditional_sample_rows:
        write_csv(files["conditional_simulation_samples"], conditional_sample_rows)
    if density_proxy_summary:
        write_csv(files["density_proxy_comparison"], density_proxy_summary)
    if density_proxy_detail:
        write_csv(files["density_proxy_cell_diagnostics"], density_proxy_detail)
    if marker_group_rows:
        write_csv(files["marker_proxy_responsibility_summary"], marker_group_rows)
    if marker_test_rows:
        write_csv(files["marker_proxy_association_tests"], marker_test_rows)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": utc_now(),
        "command": command_line(),
        "embedding": str(args.embedding),
        "raw_tarball": str(args.raw_tarball),
        "output_dir": str(args.out),
        "split_reps": args.split_reps,
        "split_seed_formula": f"seed + {args.split_seed_step} * split_rep, split_rep=1..{args.split_reps}",
        "train_fraction": args.train_fraction,
        "methods": methods,
        "primary_method": primary,
        "conditional_events": ["low PC1", "middle PC1", "high PC1"],
        "available_markers": available_markers,
        "claim_boundary": CLAIM_BOUNDARY,
        "files": {label: str(path.relative_to(ROOT)) for label, path in files.items() if path.exists()},
    }
    write_json(files["run_parameters"], payload)
    write_report(files["execution_report"], payload)


if __name__ == "__main__":
    main()
