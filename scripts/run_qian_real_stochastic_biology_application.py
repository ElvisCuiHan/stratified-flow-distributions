#!/usr/bin/env python3
"""Run a restrained real-data stochastic-biology SFD application probe.

The selected source is the 10x Genomics PBMC3k public single-cell RNA-seq
dataset. The workflow treats the first two PCA coordinates of normalized
single-cell expression as an observed state-space embedding and runs the
existing external-CSV held-out density benchmark on that embedding.

This is a static marginal density exercise only. It does not estimate
probability flux, entropy production, thermodynamic force, transition rates, or
reaction propensities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tmp" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / "tmp" / "cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "cache"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse
from scipy.io import mmread
from sklearn.decomposition import PCA

from run_jmlr_density_benchmark import (
    RUN_COLUMNS as BENCHMARK_RUN_COLUMNS,
    RUNTIME_COLUMNS as BENCHMARK_RUNTIME_COLUMNS,
    SUMMARY_COLUMNS as BENCHMARK_SUMMARY_COLUMNS,
    append_baseline_rows,
    numeric_or_none,
    parse_methods,
    summarize,
    summarize_runtime,
)
from sfd_figure_style import SFD_GRID, SFD_INK, SFD_MUTED, panel_label, panel_title, set_sfd_matplotlib_style


OUT_DIR = ROOT / "outputs" / "jmlr_real_applications" / "qian_stochastic_biology"
SCENARIO = "pbmc3k_10x_observed_state"
REPEATED_PROTOCOL_VERSION = "jmlr_qian_pbmc3k_repeated_split_v1"
REPEATED_SCENARIO_FAMILY = "real_stochastic_biology_repeated_split"
SOURCE_PAGE_URL = "https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0"
DOWNLOAD_URL = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
CELLXGENE_DOWNLOAD_DOC_URL = "https://cellxgene.cziscience.com/docs/03__Download%20Published%20Data"
GEO_PROBE_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3940254"
TRAIN_FRACTION = 0.80
CLAIM_BOUNDARY = (
    "Static observed marginal density in a two-dimensional PBMC3k expression embedding only; "
    "no probability-current, entropy-production, thermodynamic-force, transition-rate, "
    "reaction-rate, causal-transition-path, or biological-validation claim."
)
QIAN_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "qian_pbmc3k_density_soft",
    [
        (0.00, "#f8fbfb"),
        (0.18, "#e8f2f1"),
        (0.38, "#c8e4df"),
        (0.58, "#86c8bd"),
        (0.76, "#3d9e9e"),
        (0.90, "#2f6f8f"),
        (1.00, "#233b63"),
    ],
    N=256,
)
QIAN_TRAIN_COLOR = "#5f94ad"
QIAN_TEST_COLOR = "#d59b89"
QIAN_SCORE_COLORS = ["#6f9db8", "#88b99a", "#dcbf70", "#c99080", "#a995c9", "#b8c0cc"]
QIAN_WARNING_ZERO = "#d8e1e8"
QIAN_WARNING_LOW = "#d5b978"
QIAN_WARNING_HIGH = "#cf7f6c"


@dataclass(frozen=True)
class TenXMatrix:
    matrix: sparse.csr_matrix
    gene_ids: list[str]
    gene_symbols: list[str]
    barcodes: list[str]
    matrix_member: str
    genes_member: str
    barcodes_member: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def shell_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(piece)) for piece in argv)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def probe_url(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "FlowDistribution-Department-J/1.0"})
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "url": url,
        "method": "HEAD",
        "status": "not_attempted",
        "http_status": None,
        "elapsed_seconds": None,
        "content_length": None,
        "content_type": "",
        "last_modified": "",
        "etag": "",
        "error_type": "",
        "error_message": "",
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            headers = response.headers
            payload.update(
                {
                    "status": "ok",
                    "http_status": int(response.status),
                    "elapsed_seconds": time.perf_counter() - started,
                    "content_length": int(headers.get("Content-Length", "0") or 0),
                    "content_type": headers.get("Content-Type", ""),
                    "last_modified": headers.get("Last-Modified", ""),
                    "etag": headers.get("ETag", ""),
                }
            )
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error_message": " ".join(str(exc).split())[:500],
            }
        )
    return payload


def download_with_sha256(url: str, path: Path, *, timeout: float, force: bool) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "url": url,
            "path": str(path.relative_to(ROOT)),
            "status": "reused_existing",
            "bytes": path.stat().st_size,
            "sha256": digest,
            "elapsed_seconds": 0.0,
            "error_type": "",
            "error_message": "",
        }

    tmp_path = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FlowDistribution-Department-J/1.0"})
    digest = hashlib.sha256()
    total = 0
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
                total += len(chunk)
        tmp_path.replace(path)
        return {
            "url": url,
            "path": str(path.relative_to(ROOT)),
            "status": "downloaded",
            "bytes": total,
            "sha256": digest.hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": "",
            "error_message": "",
        }
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "url": url,
            "path": str(path.relative_to(ROOT)),
            "status": "failed",
            "bytes": total,
            "sha256": digest.hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error_message": " ".join(str(exc).split())[:500],
        }


def find_member(names: list[str], suffixes: tuple[str, ...]) -> str:
    matches = [name for name in names if name.endswith(suffixes)]
    if not matches:
        raise ValueError(f"Could not find any tar member ending in {suffixes}.")
    matches.sort(key=lambda item: (len(item), item))
    return matches[0]


def read_tsv_member(tar: tarfile.TarFile, member_name: str) -> list[list[str]]:
    member = tar.getmember(member_name)
    handle = tar.extractfile(member)
    if handle is None:
        raise ValueError(f"Could not open tar member {member_name}.")
    with handle:
        rows = []
        for raw_line in handle:
            line = raw_line.decode("utf-8").strip()
            if line:
                rows.append(line.split("\t"))
    return rows


def read_10x_tarball(path: Path) -> TenXMatrix:
    with tarfile.open(path, "r:gz") as tar:
        names = [member.name for member in tar.getmembers() if member.isfile()]
        matrix_member = find_member(names, ("matrix.mtx", "matrix.mtx.gz"))
        genes_member = find_member(names, ("genes.tsv", "features.tsv", "genes.tsv.gz", "features.tsv.gz"))
        barcodes_member = find_member(names, ("barcodes.tsv", "barcodes.tsv.gz"))

        matrix_handle = tar.extractfile(tar.getmember(matrix_member))
        if matrix_handle is None:
            raise ValueError(f"Could not open tar member {matrix_member}.")
        with matrix_handle:
            raw_matrix = mmread(matrix_handle).tocsc()

        gene_rows = read_tsv_member(tar, genes_member)
        barcode_rows = read_tsv_member(tar, barcodes_member)

    gene_ids = [row[0] for row in gene_rows]
    gene_symbols = [row[1] if len(row) > 1 else row[0] for row in gene_rows]
    barcodes = [row[0] for row in barcode_rows]

    if raw_matrix.shape == (len(gene_ids), len(barcodes)):
        cell_by_gene = raw_matrix.T.tocsr()
    elif raw_matrix.shape == (len(barcodes), len(gene_ids)):
        cell_by_gene = raw_matrix.tocsr()
    else:
        raise ValueError(
            "Matrix dimensions do not match gene/barcode counts: "
            f"matrix={raw_matrix.shape}, genes={len(gene_ids)}, barcodes={len(barcodes)}."
        )

    return TenXMatrix(
        matrix=cell_by_gene.astype(np.float64).tocsr(),
        gene_ids=gene_ids,
        gene_symbols=gene_symbols,
        barcodes=barcodes,
        matrix_member=matrix_member,
        genes_member=genes_member,
        barcodes_member=barcodes_member,
    )


def select_cells(
    x: sparse.csr_matrix,
    barcodes: list[str],
    *,
    max_cells: int,
    seed: int,
) -> tuple[sparse.csr_matrix, list[str], np.ndarray]:
    n_cells = x.shape[0]
    all_idx = np.arange(n_cells)
    if max_cells <= 0 or max_cells >= n_cells:
        return x, barcodes, all_idx
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(all_idx, size=max_cells, replace=False))
    return x[selected], [barcodes[i] for i in selected], selected


def make_embedding(args: argparse.Namespace, tenx: TenXMatrix) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    x = tenx.matrix
    library_size_all = np.asarray(x.sum(axis=1)).ravel()
    valid_cells = library_size_all > 0
    x = x[valid_cells]
    valid_barcodes = [barcode for barcode, keep in zip(tenx.barcodes, valid_cells) if keep]
    valid_original_index = np.flatnonzero(valid_cells)
    x, selected_barcodes, selected_local_index = select_cells(
        x,
        valid_barcodes,
        max_cells=args.max_cells,
        seed=args.seed + 13,
    )
    selected_original_index = valid_original_index[selected_local_index]

    library_size = np.asarray(x.sum(axis=1)).ravel()
    n_genes_by_counts = np.asarray((x > 0).sum(axis=1)).ravel()
    mt_mask = np.asarray([symbol.upper().startswith("MT-") for symbol in tenx.gene_symbols], dtype=bool)
    mt_counts = np.asarray(x[:, mt_mask].sum(axis=1)).ravel() if bool(np.any(mt_mask)) else np.zeros(x.shape[0])
    pct_mito = np.divide(mt_counts, library_size, out=np.zeros_like(mt_counts), where=library_size > 0) * 100.0

    gene_detection = np.asarray((x > 0).sum(axis=0)).ravel()
    keep_gene = gene_detection >= args.min_cells_per_gene
    if int(np.sum(keep_gene)) < 2:
        raise ValueError(
            f"Only {int(np.sum(keep_gene))} genes passed min_cells_per_gene={args.min_cells_per_gene}."
        )
    x_gene = x[:, keep_gene].tocsr()
    kept_gene_indices = np.flatnonzero(keep_gene)
    scale = np.divide(args.target_sum, library_size, out=np.zeros_like(library_size), where=library_size > 0)
    x_norm = x_gene.multiply(scale[:, None]).tocsr()
    x_norm.data = np.log1p(x_norm.data)

    gene_means = np.asarray(x_norm.mean(axis=0)).ravel()
    gene_second = np.asarray(x_norm.power(2).mean(axis=0)).ravel()
    gene_variance = np.maximum(gene_second - gene_means**2, 0.0)
    n_select = min(args.n_variable_genes, x_norm.shape[1])
    top_local = np.argpartition(gene_variance, -n_select)[-n_select:]
    top_local = top_local[np.argsort(gene_variance[top_local])[::-1]]
    top_global = kept_gene_indices[top_local]

    dense = x_norm[:, top_local].toarray()
    dense_mean = dense.mean(axis=0)
    dense_sd = dense.std(axis=0, ddof=1)
    dense_sd[dense_sd < 1e-8] = 1.0
    dense = (dense - dense_mean) / dense_sd

    pca = PCA(n_components=2, svd_solver="randomized", random_state=args.seed)
    embedding = pca.fit_transform(dense)

    metadata = {
        "source_matrix_shape_cells_by_genes": [int(tenx.matrix.shape[0]), int(tenx.matrix.shape[1])],
        "valid_nonzero_cells": int(np.sum(valid_cells)),
        "selected_cells": int(x.shape[0]),
        "selected_original_indices_min": int(np.min(selected_original_index)),
        "selected_original_indices_max": int(np.max(selected_original_index)),
        "genes_total": int(tenx.matrix.shape[1]),
        "genes_passing_min_cells": int(x_norm.shape[1]),
        "n_variable_genes": int(n_select),
        "min_cells_per_gene": int(args.min_cells_per_gene),
        "target_sum": float(args.target_sum),
        "pca_explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "pca_singular_values": [float(value) for value in pca.singular_values_],
        "library_size_min": float(np.min(library_size)),
        "library_size_median": float(np.median(library_size)),
        "library_size_max": float(np.max(library_size)),
        "n_genes_by_counts_min": int(np.min(n_genes_by_counts)),
        "n_genes_by_counts_median": float(np.median(n_genes_by_counts)),
        "n_genes_by_counts_max": int(np.max(n_genes_by_counts)),
        "pct_mito_median": float(np.median(pct_mito)),
        "pct_mito_max": float(np.max(pct_mito)),
        "selected_barcodes": selected_barcodes,
        "selected_original_indices": selected_original_index.tolist(),
        "library_size": library_size.tolist(),
        "n_genes_by_counts": n_genes_by_counts.tolist(),
        "pct_mito": pct_mito.tolist(),
    }

    variable_gene_rows = []
    for rank, local_idx in enumerate(top_local, start=1):
        global_idx = int(kept_gene_indices[int(local_idx)])
        variable_gene_rows.append(
            {
                "rank": rank,
                "gene_id": tenx.gene_ids[global_idx],
                "gene_symbol": tenx.gene_symbols[global_idx],
                "detected_cells": int(gene_detection[global_idx]),
                "log_normalized_mean": float(gene_means[int(local_idx)]),
                "log_normalized_variance": float(gene_variance[int(local_idx)]),
            }
        )
    return embedding, metadata, variable_gene_rows


def write_embedding_tables(
    out_dir: Path,
    args: argparse.Namespace,
    embedding: np.ndarray,
    metadata: dict[str, Any],
    *,
    split_seed: int | None = None,
) -> dict[str, Path]:
    seed = args.seed if split_seed is None else split_seed
    rng = np.random.default_rng(seed)
    n = embedding.shape[0]
    order = np.arange(n)
    rng.shuffle(order)
    n_train = int(round(args.train_fraction * n))
    train_positions = set(int(i) for i in order[:n_train])

    embedding_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for i in range(n):
        split = "train" if i in train_positions else "test"
        sample_id = f"pbmc3k_cell_{i:04d}"
        embedding_rows.append(
            {
                "scenario": SCENARIO,
                "sample_id": sample_id,
                "cell_barcode": metadata["selected_barcodes"][i],
                "Y1": float(embedding[i, 0]),
                "Y2": float(embedding[i, 1]),
                "library_size": float(metadata["library_size"][i]),
                "n_genes_by_counts": int(metadata["n_genes_by_counts"][i]),
                "pct_mito": float(metadata["pct_mito"][i]),
                "split": split,
                "state_space": "log1p_cp10k_variable_gene_pca2",
                "stratum": "observed_snapshot_unlabeled",
            }
        )
        split_rows.append(
            {
                "scenario": SCENARIO,
                "sample_id": sample_id,
                "split": split,
                "split_seed": seed,
                "train_fraction": args.train_fraction,
                "split_rule": "deterministic_shuffle_over_selected_cells",
            }
        )

    feature_summary_rows = [
        {
            "feature": "Y1",
            "source": "PCA component 1 of selected log-normalized expression genes",
            "mean": float(np.mean(embedding[:, 0])),
            "sd": float(np.std(embedding[:, 0], ddof=1)),
            "min": float(np.min(embedding[:, 0])),
            "max": float(np.max(embedding[:, 0])),
            "pca_explained_variance_ratio": metadata["pca_explained_variance_ratio"][0],
        },
        {
            "feature": "Y2",
            "source": "PCA component 2 of selected log-normalized expression genes",
            "mean": float(np.mean(embedding[:, 1])),
            "sd": float(np.std(embedding[:, 1], ddof=1)),
            "min": float(np.min(embedding[:, 1])),
            "max": float(np.max(embedding[:, 1])),
            "pca_explained_variance_ratio": metadata["pca_explained_variance_ratio"][1],
        },
    ]
    qc_rows = [
        {
            "scenario": SCENARIO,
            "n_cells": n,
            "n_train": n_train,
            "n_test": n - n_train,
            "genes_total": metadata["genes_total"],
            "genes_passing_min_cells": metadata["genes_passing_min_cells"],
            "n_variable_genes": metadata["n_variable_genes"],
            "library_size_min": metadata["library_size_min"],
            "library_size_median": metadata["library_size_median"],
            "library_size_max": metadata["library_size_max"],
            "n_genes_by_counts_min": metadata["n_genes_by_counts_min"],
            "n_genes_by_counts_median": metadata["n_genes_by_counts_median"],
            "n_genes_by_counts_max": metadata["n_genes_by_counts_max"],
            "pct_mito_median": metadata["pct_mito_median"],
            "pct_mito_max": metadata["pct_mito_max"],
        }
    ]

    paths = {
        "embedding_csv": out_dir / "pbmc3k_observed_state_embedding.csv",
        "split_csv": out_dir / "train_test_split_assignments.csv",
        "feature_summary_csv": out_dir / "embedding_feature_summary.csv",
        "qc_summary_csv": out_dir / "dataset_qc_summary.csv",
    }
    write_csv(paths["embedding_csv"], embedding_rows)
    write_csv(paths["split_csv"], split_rows)
    write_csv(paths["feature_summary_csv"], feature_summary_rows)
    write_csv(paths["qc_summary_csv"], qc_rows)
    return paths


def draw_embedding_figure(
    figure_dir: Path,
    embedding: np.ndarray,
    metadata: dict[str, Any],
    split_csv: Path,
) -> dict[str, str]:
    set_sfd_matplotlib_style(plt, base_size=8.6, title_size=10.0, label_size=8.8, tick_size=7.6)
    split_by_sample: list[str] = []
    with split_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            split_by_sample.append(row["split"])
    split_array = np.asarray(split_by_sample)

    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "pbmc3k_observed_state_embedding.png"
    pdf_path = figure_dir / "pbmc3k_observed_state_embedding.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    hb = axes[0].hexbin(
        embedding[:, 0],
        embedding[:, 1],
        gridsize=42,
        mincnt=1,
        cmap=QIAN_DENSITY_CMAP,
        linewidths=0,
    )
    panel_title(axes[0], "Observed state density")
    axes[0].set_xlabel("PCA state coordinate Y1")
    axes[0].set_ylabel("PCA state coordinate Y2")
    cb = fig.colorbar(hb, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label("cell count")

    for split, color, marker, alpha in [
        ("train", QIAN_TRAIN_COLOR, "o", 0.40),
        ("test", QIAN_TEST_COLOR, "^", 0.62),
    ]:
        mask = split_array == split
        axes[1].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=7,
            c=color,
            marker=marker,
            alpha=alpha,
            linewidths=0,
            label=split,
        )
    panel_title(axes[1], "Deterministic train/test split")
    axes[1].set_xlabel("PCA state coordinate Y1")
    axes[1].set_ylabel("PCA state coordinate Y2")
    axes[1].legend(frameon=False, markerscale=1.8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=SFD_GRID, alpha=0.62, linewidth=0.65)
        ax.tick_params(colors=SFD_INK)

    fig.suptitle(
        "10x PBMC3k static single-cell expression embedding",
        fontsize=11.0,
        fontweight="bold",
        color=SFD_INK,
    )
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "embedding_png": str(png_path.relative_to(ROOT)),
        "embedding_pdf": str(pdf_path.relative_to(ROOT)),
        "caption": (
            "Diagnostic state-space view of the 10x PBMC3k snapshot data after "
            "log-normalization, variable-gene selection, and PCA. The left panel "
            "shows observed occupancy in the two-dimensional embedding; the right "
            "panel shows the deterministic split used for held-out density scoring. "
            "The figure supports only a static marginal density framing."
        ),
        "pca_explained_variance_ratio": metadata["pca_explained_variance_ratio"],
    }


def run_density_benchmark(
    args: argparse.Namespace,
    paths: dict[str, Path],
    out_dir: Path,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    benchmark_dir = out_dir / "density_benchmark"
    figure_dir = out_dir / "figures"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_jmlr_density_benchmark.py"),
        "--external-csv",
        str(paths["embedding_csv"]),
        "--split-assignments",
        str(paths["split_csv"]),
        "--features",
        "Y1,Y2",
        "--truth-columns",
        "library_size,n_genes_by_counts,pct_mito",
        "--scenarios",
        SCENARIO,
        "--scenario-family",
        "real_stochastic_biology_snapshot",
        "--out",
        str(benchmark_dir),
        "--figure-dir",
        str(figure_dir),
        "--figure-prefix",
        "qian_pbmc3k_heldout_log_score",
        "--methods",
        args.methods,
        "--seed",
        str(args.seed),
        "--n-train",
        str(n_train),
        "--n-test",
        str(n_test),
        "--max-iter",
        str(args.max_iter),
        "--tol",
        str(args.tol),
        "--reg",
        str(args.reg),
        "--kde-jitter",
        str(args.kde_jitter),
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT / "src") if not existing else f"{ROOT / 'src'}{os.pathsep}{existing}"
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    payload = {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "elapsed_seconds": time.perf_counter() - started,
        "command": shell_command(command),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_dir": str(benchmark_dir.relative_to(ROOT)),
        "expected_files": {
            "heldout_log_scores": str((benchmark_dir / "heldout_log_scores.csv").relative_to(ROOT)),
            "benchmark_summary": str((benchmark_dir / "benchmark_summary.csv").relative_to(ROOT)),
            "runtime": str((benchmark_dir / "runtime.csv").relative_to(ROOT)),
            "execution_report": str((benchmark_dir / "execution_report.md").relative_to(ROOT)),
        },
    }
    if result.returncode != 0:
        raise RuntimeError(f"Density benchmark failed with code {result.returncode}: {result.stderr}")
    return payload


def read_benchmark_summary(benchmark_summary_csv: Path) -> list[dict[str, str]]:
    if not benchmark_summary_csv.exists():
        return []
    with benchmark_summary_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def split_rule_for_seed(args: argparse.Namespace, split_seed: int) -> str:
    return f"deterministic_shuffle_over_selected_cells; train_fraction={args.train_fraction:g}; split_seed={split_seed}"


def split_mask(n: int, train_fraction: float, split_seed: int) -> np.ndarray:
    rng = np.random.default_rng(split_seed)
    order = np.arange(n)
    rng.shuffle(order)
    n_train = int(round(train_fraction * n))
    mask = np.zeros(n, dtype=bool)
    mask[order[:n_train]] = True
    return mask


def split_seed_for_rep(args: argparse.Namespace, split_rep: int) -> int:
    return int(args.seed + args.split_seed_step * split_rep)


def repeated_split_benchmark(
    args: argparse.Namespace,
    embedding: np.ndarray,
    metadata: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any] | None:
    if args.split_reps <= 1:
        return None

    include_gaussian, gmm_components, include_kde, include_oracle = parse_methods(args.methods)
    if include_oracle:
        raise ValueError("Oracle SFD is not available for the real PBMC3k application.")

    benchmark_dir = out_dir / "density_benchmark_repeated_splits"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    bench_args = SimpleNamespace(
        protocol_version=REPEATED_PROTOCOL_VERSION,
        scenario_family=REPEATED_SCENARIO_FAMILY,
        external_csv=out_dir / "pbmc3k_observed_state_embedding.csv",
        reg=args.reg,
        max_iter=args.max_iter,
        tol=args.tol,
        kde_jitter=args.kde_jitter,
    )

    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    n = int(embedding.shape[0])
    sample_ids = [f"pbmc3k_cell_{i:04d}" for i in range(n)]
    for split_rep in range(1, args.split_reps + 1):
        split_seed = split_seed_for_rep(args, split_rep)
        train_mask = split_mask(n, args.train_fraction, split_seed)
        train = embedding[train_mask]
        test = embedding[~train_mask]
        seed = split_seed
        before = len(rows)
        append_baseline_rows(
            rows,
            train,
            test,
            scenario=SCENARIO,
            rep=split_rep,
            seed=seed,
            args=bench_args,
            include_gaussian=include_gaussian,
            gmm_components=gmm_components,
            include_kde=include_kde,
        )
        for row in rows[before:]:
            row.update(
                {
                    "split_rep": split_rep,
                    "split_seed": split_seed,
                    "split_rule": split_rule_for_seed(args, split_seed),
                    "feature_set": "Y1,Y2",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
        for i, sample_id in enumerate(sample_ids):
            split_rows.append(
                {
                    "scenario": SCENARIO,
                    "sample_id": sample_id,
                    "split_rep": split_rep,
                    "split_seed": split_seed,
                    "split": "train" if bool(train_mask[i]) else "test",
                    "train_fraction": args.train_fraction,
                    "split_rule": split_rule_for_seed(args, split_seed),
                }
            )

    summary_rows = summarize(rows)
    for row in summary_rows:
        row.update(
            {
                "split_reps": args.split_reps,
                "split_seed_formula": f"seed + {args.split_seed_step} * split_rep, split_rep=1..{args.split_reps}",
                "feature_set": "Y1,Y2",
                "claim_boundary": CLAIM_BOUNDARY,
                "paper_use_boundary": (
                    "Repeated-split static density diagnostic; usable for supplement evidence with warning "
                    "rates shown explicitly, not for biological or dynamic-process claims."
                ),
            }
        )
    runtime_rows = summarize_runtime(summary_rows)
    for row in runtime_rows:
        row.update({"split_reps": args.split_reps, "claim_boundary": CLAIM_BOUNDARY})

    diagnostic_rows = convergence_diagnostic_rows(rows, summary_rows)

    write_csv(
        benchmark_dir / "heldout_log_scores.csv",
        rows,
        [*BENCHMARK_RUN_COLUMNS, "split_rep", "split_seed", "split_rule", "feature_set", "claim_boundary"],
    )
    write_csv(
        benchmark_dir / "benchmark_summary.csv",
        summary_rows,
        [
            *BENCHMARK_SUMMARY_COLUMNS,
            "split_reps",
            "split_seed_formula",
            "feature_set",
            "claim_boundary",
            "paper_use_boundary",
        ],
    )
    write_csv(
        benchmark_dir / "runtime.csv",
        runtime_rows,
        [*BENCHMARK_RUNTIME_COLUMNS, "split_reps", "claim_boundary"],
    )
    write_csv(
        benchmark_dir / "split_assignments_repeated.csv",
        split_rows,
        ["scenario", "sample_id", "split_rep", "split_seed", "split", "train_fraction", "split_rule"],
    )
    write_csv(
        benchmark_dir / "convergence_warning_diagnostics.csv",
        diagnostic_rows,
        [
            "method",
            "rows",
            "ok_rows",
            "warning_rows",
            "failed_rows",
            "warning_rate",
            "failure_rate",
            "nonconverged_rate",
            "mean_n_iter",
            "max_n_iter",
            "claim_boundary",
        ],
    )

    figure_paths = draw_repeated_split_figure(
        figure_dir=figure_dir,
        benchmark_dir=benchmark_dir,
        embedding=embedding,
        rows=rows,
        summary_rows=summary_rows,
        diagnostic_rows=diagnostic_rows,
        metadata=metadata,
    )
    write_repeated_split_report(
        benchmark_dir / "execution_report.md",
        args,
        rows,
        summary_rows,
        diagnostic_rows,
        figure_paths,
        metadata,
    )

    return {
        "status": "ok",
        "output_dir": str(benchmark_dir.relative_to(ROOT)),
        "split_reps": args.split_reps,
        "rows": len(rows),
        "summary_rows": len(summary_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "figure_paths": [str(path.relative_to(ROOT)) for path in figure_paths],
        "summary_path": str((benchmark_dir / "benchmark_summary.csv").relative_to(ROOT)),
        "diagnostics_path": str((benchmark_dir / "convergence_warning_diagnostics.csv").relative_to(ROOT)),
        "report_path": str((benchmark_dir / "execution_report.md").relative_to(ROOT)),
    }


def convergence_diagnostic_rows(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_by_method = {str(row["method"]): row for row in summary_rows}
    out: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        subset = [row for row in rows if str(row["method"]) == method]
        statuses = [str(row.get("status", "")) for row in subset]
        n_iter = [numeric_or_none(row.get("n_iter")) for row in subset]
        n_iter_values = [float(value) for value in n_iter if value is not None]
        nonconverged = [str(row.get("converged", "")).lower() == "false" or row.get("converged") is False for row in subset]
        summary = summary_by_method.get(method, {})
        out.append(
            {
                "method": method,
                "rows": len(subset),
                "ok_rows": sum(1 for status in statuses if status == "ok"),
                "warning_rows": sum(1 for status in statuses if status == "warning"),
                "failed_rows": sum(1 for status in statuses if status == "failed"),
                "warning_rate": summary.get("warning_rate", ""),
                "failure_rate": summary.get("failure_rate", ""),
                "nonconverged_rate": float(np.mean(nonconverged)) if nonconverged else 0.0,
                "mean_n_iter": float(np.mean(n_iter_values)) if n_iter_values else 0.0,
                "max_n_iter": int(max(n_iter_values)) if n_iter_values else 0,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return out


def draw_repeated_split_figure(
    *,
    figure_dir: Path,
    benchmark_dir: Path,
    embedding: np.ndarray,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[Path]:
    set_sfd_matplotlib_style(plt)
    methods = [str(row["method"]) for row in summary_rows if numeric_or_none(row.get("mean_heldout_log_score")) is not None]
    scores = np.asarray([float(row["mean_heldout_log_score"]) for row in summary_rows if row["method"] in methods], dtype=float)
    ses = np.asarray([float(row.get("se_heldout_log_score") or 0.0) for row in summary_rows if row["method"] in methods], dtype=float)
    warning_by_method = {str(row["method"]): float(row["warning_rate"] or 0.0) for row in diagnostic_rows}
    warning_rates = np.asarray([warning_by_method.get(method, 0.0) for method in methods], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)
    hb = axes[0].hexbin(
        embedding[:, 0],
        embedding[:, 1],
        gridsize=44,
        mincnt=1,
        cmap=QIAN_DENSITY_CMAP,
        linewidths=0,
    )
    panel_title(axes[0], "A. Embedding occupancy")
    axes[0].set_xlabel("PCA state coordinate Y1")
    axes[0].set_ylabel("PCA state coordinate Y2")
    cb = fig.colorbar(hb, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_label("cell count")

    x = np.arange(len(methods))
    axes[1].errorbar(
        x,
        scores,
        yerr=ses,
        fmt="none",
        ecolor=SFD_MUTED,
        elinewidth=1.1,
        capsize=3.0,
        zorder=2,
    )
    axes[1].scatter(
        x,
        scores,
        s=54,
        color=[QIAN_SCORE_COLORS[i % len(QIAN_SCORE_COLORS)] for i in x],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for i, value in enumerate(scores):
        axes[1].text(i, value + 0.045, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5, color=SFD_MUTED)
    panel_title(axes[1], "B. Held-out log score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=32, ha="right")
    axes[1].set_ylabel("mean log score")
    score_pad = max(0.08, 0.08 * float(np.ptp(scores)))
    axes[1].set_ylim(float(np.min(scores) - score_pad), float(np.max(scores) + 2.2 * score_pad))

    warning_colors = [
        QIAN_WARNING_ZERO if rate == 0 else QIAN_WARNING_LOW if rate < 0.5 else QIAN_WARNING_HIGH
        for rate in warning_rates
    ]
    axes[2].bar(x, warning_rates, color=warning_colors, edgecolor="white", linewidth=0.8)
    panel_title(axes[2], "C. Warning diagnostics")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=32, ha="right")
    axes[2].set_ylabel("warning rate")
    axes[2].set_ylim(0.0, 1.05)

    for ax in axes:
        ax.grid(axis="y", color=SFD_GRID, linewidth=0.7, alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=SFD_INK)
    axes[0].grid(color=SFD_GRID, linewidth=0.7, alpha=0.65)
    axes[0].text(
        0.02,
        0.02,
        f"{metadata['selected_cells']} cells; static PCA embedding",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=SFD_MUTED,
    )

    png = figure_dir / "qian_pbmc3k_repeated_split_diagnostics.png"
    pdf = figure_dir / "qian_pbmc3k_repeated_split_diagnostics.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    caption = (
        "Qian/PBMC3k repeated-split static density diagnostic. Panel A shows observed occupancy "
        "in the two-dimensional log-normalized expression PCA embedding. Panel B reports held-out "
        "log scores across deterministic split seeds with standard errors. Panel C reports method "
        "warning rates, including GMM nonconvergence warnings. The figure supports only a static "
        "marginal density and convergence-diagnostic claim."
    )
    (benchmark_dir / "figure_caption.md").write_text(caption + "\n", encoding="utf-8")
    return [png, pdf]


def write_repeated_split_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    figure_paths: list[Path],
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Qian/PBMC3k Repeated-Split Density Evidence",
        "",
        f"- status: `completed`",
        f"- protocol_version: `{REPEATED_PROTOCOL_VERSION}`",
        f"- scenario: `{SCENARIO}`",
        f"- source_dataset: `10x Genomics PBMC3k`",
        f"- selected_cells: `{metadata['selected_cells']}`",
        f"- split_reps: `{args.split_reps}`",
        f"- split_seed_formula: `seed + {args.split_seed_step} * split_rep`, split_rep `1..{args.split_reps}`",
        f"- train_fraction: `{args.train_fraction}`",
        f"- density_methods: `{args.methods}`",
        f"- command: `{shell_command([sys.executable, *sys.argv])}`",
        "",
        "## Held-Out Density Summary",
        "",
        "| method | mean held-out log score | SE | rank | warning rate | failure rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {score:.6g} | {se:.6g} | {rank} | {warning:.3g} | {failure:.3g} |".format(
                method=row["method"],
                score=float(row["mean_heldout_log_score"]),
                se=float(row.get("se_heldout_log_score") or 0.0),
                rank=row["rank_within_scenario"],
                warning=float(row.get("warning_rate") or 0.0),
                failure=float(row.get("failure_rate") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Warning And Convergence Diagnostics",
            "",
            "| method | rows | warnings | failures | nonconverged rate | max iterations |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostic_rows:
        lines.append(
            "| {method} | {rows} | {warning_rows} | {failed_rows} | {nonconverged_rate:.3g} | {max_n_iter} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            *(f"- `{path.relative_to(ROOT)}`" for path in figure_paths),
            "",
            "## Claim Boundary",
            "",
            CLAIM_BOUNDARY,
            "",
            "## Paper Use",
            "",
            "This is supplement-suitable static density evidence if described as a repeated-split "
            "diagnostic. It should not be written as biological validation or as evidence for "
            "transition rates, probability fluxes, entropy production, or a dynamic stochastic "
            "biology model. If a high-scoring GMM row has nonzero warning rate, the warning rate "
            "must accompany the score.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_parameters(
    path_txt: Path,
    path_json: Path,
    args: argparse.Namespace,
    output_paths: dict[str, str],
    source_probe: dict[str, Any],
    download_status: dict[str, Any],
    tenx: TenXMatrix,
    embedding_metadata: dict[str, Any],
    density_status: dict[str, Any] | None,
    repeated_density_status: dict[str, Any] | None,
) -> None:
    payload = {
        "script": str((ROOT / "scripts" / "run_qian_real_stochastic_biology_application.py").relative_to(ROOT)),
        "command": shell_command([sys.executable, *sys.argv]),
        "started_at_utc": args.started_at_utc,
        "finished_at_utc": utc_now(),
        "scenario": SCENARIO,
        "source_page_url": SOURCE_PAGE_URL,
        "download_url": args.download_url,
        "cellxgene_download_doc_url": CELLXGENE_DOWNLOAD_DOC_URL,
        "geo_probe_url": GEO_PROBE_URL,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "n_variable_genes": args.n_variable_genes,
        "min_cells_per_gene": args.min_cells_per_gene,
        "target_sum": args.target_sum,
        "max_cells": args.max_cells,
        "density_methods": args.methods,
        "split_reps": args.split_reps,
        "split_seed_step": args.split_seed_step,
        "offline_cached_data": args.offline_cached_data,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_probe": source_probe,
        "download_status": download_status,
        "tar_members": {
            "matrix": tenx.matrix_member,
            "genes": tenx.genes_member,
            "barcodes": tenx.barcodes_member,
        },
        "embedding_metadata": {
            key: value
            for key, value in embedding_metadata.items()
            if key not in {"selected_barcodes", "selected_original_indices", "library_size", "n_genes_by_counts", "pct_mito"}
        },
        "density_status": density_status,
        "repeated_density_status": repeated_density_status,
        "output_paths": output_paths,
    }
    write_json(path_json, payload)

    lines = [
        "Qian-inspired real stochastic-biology application probe",
        f"scenario={SCENARIO}",
        f"script=scripts/run_qian_real_stochastic_biology_application.py",
        f"command={payload['command']}",
        f"started_at_utc={args.started_at_utc}",
        f"finished_at_utc={payload['finished_at_utc']}",
        f"source_page_url={SOURCE_PAGE_URL}",
        f"download_url={args.download_url}",
        f"download_status={download_status.get('status', '')}",
        f"download_bytes={download_status.get('bytes', '')}",
        f"download_sha256={download_status.get('sha256', '')}",
        f"matrix_shape_cells_by_genes={tenx.matrix.shape[0]}x{tenx.matrix.shape[1]}",
        f"selected_cells={embedding_metadata['selected_cells']}",
        f"genes_passing_min_cells={embedding_metadata['genes_passing_min_cells']}",
        f"n_variable_genes={embedding_metadata['n_variable_genes']}",
        f"train_fraction={args.train_fraction}",
        f"seed={args.seed}",
        f"density_methods={args.methods}",
        f"split_reps={args.split_reps}",
        f"split_seed_step={args.split_seed_step}",
        f"offline_cached_data={args.offline_cached_data}",
        f"repeated_density_status={(repeated_density_status or {'status': 'skipped'})['status']}",
        f"claim_boundary={CLAIM_BOUNDARY}",
    ]
    for label, value in output_paths.items():
        lines.append(f"output.{label}={value}")
    path_txt.write_text("\n".join(lines) + "\n")


def write_report(
    path: Path,
    args: argparse.Namespace,
    output_paths: dict[str, str],
    source_probe: dict[str, Any],
    download_status: dict[str, Any],
    tenx: TenXMatrix,
    embedding_metadata: dict[str, Any],
    density_status: dict[str, Any] | None,
    repeated_density_status: dict[str, Any] | None,
    benchmark_summary: list[dict[str, str]],
    figure_payload: dict[str, Any],
) -> None:
    n_train = int(round(args.train_fraction * embedding_metadata["selected_cells"]))
    n_test = int(embedding_metadata["selected_cells"] - n_train)
    status = density_status["status"] if density_status else "skipped"
    lines = [
        "# Qian-Inspired Real Stochastic-Biology Application Probe",
        "",
        "## Decision",
        "",
        "Selected source: 10x Genomics PBMC3k single-cell RNA-seq, treated as a static snapshot of observed cell-state expression.",
        "The source is suitable for a restrained SFD framing as an observed marginal state-space density after explicit preprocessing.",
        "",
        "## Source URLs",
        "",
        f"- 10x dataset page: {SOURCE_PAGE_URL}",
        f"- 10x Matrix Market download: {args.download_url}",
        f"- CZ CELLxGENE download documentation checked as an alternate route: {CELLXGENE_DOWNLOAD_DOC_URL}",
        f"- GEO sample page checked as an alternate route: {GEO_PROBE_URL}",
        "",
        "## Probe And Parse Status",
        "",
        f"- download probe status: `{source_probe.get('status')}`, HTTP `{source_probe.get('http_status')}`",
        f"- download status: `{download_status.get('status')}`",
        f"- downloaded bytes: `{download_status.get('bytes')}`",
        f"- sha256: `{download_status.get('sha256')}`",
        f"- parsed Matrix Market member: `{tenx.matrix_member}`",
        f"- parsed genes member: `{tenx.genes_member}`",
        f"- parsed barcodes member: `{tenx.barcodes_member}`",
        f"- matrix shape after transpose: `{tenx.matrix.shape[0]}` cells by `{tenx.matrix.shape[1]}` genes",
        f"- selected cells: `{embedding_metadata['selected_cells']}`",
        f"- train/test rows: `{n_train}` train, `{n_test}` test",
        f"- genes passing detection filter: `{embedding_metadata['genes_passing_min_cells']}`",
        f"- variable genes used for PCA: `{embedding_metadata['n_variable_genes']}`",
        "",
        "## Workflow",
        "",
        "- Parsed the official 10x Matrix Market tarball with SciPy sparse readers.",
        "- Kept nonzero-library cells, log-normalized counts per 10,000 molecules, selected high-variance genes after a minimum cell-detection filter, and embedded cells into two PCA state coordinates `Y1,Y2`.",
        "- Wrote a deterministic train/test split and ran the existing external-CSV density benchmark on `Y1,Y2`.",
        "- The embedding figure is a diagnostic of observed occupancy, not a biological discovery claim.",
        "",
        "## Density Benchmark Status",
        "",
        f"- status: `{status}`",
    ]
    if density_status:
        lines.extend(
            [
                f"- command: `{density_status['command']}`",
                f"- elapsed seconds: `{density_status['elapsed_seconds']:.3f}`",
                "",
                "| method | mean held-out log score | rank | warning rate | failure rate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in benchmark_summary:
            lines.append(
                "| {method} | {score} | {rank} | {warning} | {failure} |".format(
                    method=row.get("method", ""),
                    score=row.get("mean_heldout_log_score", ""),
                    rank=row.get("rank_within_scenario", ""),
                    warning=row.get("warning_rate", ""),
                    failure=row.get("failure_rate", ""),
                )
            )
    if repeated_density_status:
        lines.extend(
            [
                "",
                "## Repeated-Split Supplement Evidence",
                "",
                f"- status: `{repeated_density_status['status']}`",
                f"- split_reps: `{repeated_density_status['split_reps']}`",
                f"- output_dir: `{repeated_density_status['output_dir']}`",
                f"- summary: `{repeated_density_status['summary_path']}`",
                f"- diagnostics: `{repeated_density_status['diagnostics_path']}`",
                f"- report: `{repeated_density_status['report_path']}`",
                "",
                "The repeated-split output is the preferred supplement-facing evidence package for PBMC3k. "
                "Use its uncertainty and warning-rate columns together; do not cite a high-scoring "
                "nonconverged GMM row as a clean superiority result.",
            ]
        )

    lines.extend(
        [
            "",
            "## Figure Caption",
            "",
            figure_payload["caption"],
            "",
            "## Output Files",
            "",
        ]
    )
    for label, value in output_paths.items():
        lines.append(f"- `{label}`: `{value}`")

    lines.extend(
        [
            "",
            "## Alternate Source Admission",
            "",
            "CZ CELLxGENE was not selected for this smoke run because its official download route is h5ad/AnnData or Seurat RDS, which is better handled with dedicated ecosystem dependencies and dataset-specific permanent links.",
            "The GEO example confirms that NIH GEO can expose processed single-cell TSV/CSV supplementary files, but that route is more accession- and sample-specific for a first reproducible application smoke run.",
            "",
            "## Claim Boundary And Risks",
            "",
            "- This run models a static observed marginal distribution in a two-dimensional expression embedding.",
            "- It does not estimate probability currents, entropy production, thermodynamic force, transition rates, reaction propensities, or causal transition paths.",
            "- PCA coordinates are a preprocessing choice; any paper-facing biological interpretation would need a frozen preprocessing protocol, cell-type annotations or validation, and sensitivity checks.",
            "- The density benchmark is a real-data smoke run with one deterministic split, not a replicated paper-facing comparison.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output-dir", dest="output_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--download-url", default=DOWNLOAD_URL)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--n-variable-genes", type=int, default=1000)
    parser.add_argument("--min-cells-per-gene", type=int, default=10)
    parser.add_argument("--target-sum", type=float, default=10000.0)
    parser.add_argument("--max-cells", type=int, default=0, help="Optional deterministic cell cap; 0 uses all cells.")
    parser.add_argument("--methods", default="gaussian,kde,gmm2,gmm4,gmm8")
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--kde-jitter", type=float, default=1e-6)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--offline-cached-data",
        action="store_true",
        help="Use the cached 10x tarball only; do not probe or retrieve external data.",
    )
    parser.add_argument(
        "--split-reps",
        type=int,
        default=1,
        help="Number of deterministic train/test split seeds for supplement-facing repeated evidence.",
    )
    parser.add_argument(
        "--split-seed-step",
        type=int,
        default=10000,
        help="Repeated split seed formula is seed + split_seed_step * split_rep.",
    )
    parser.add_argument("--skip-density-benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")
    if args.n_variable_genes < 2:
        raise ValueError("--n-variable-genes must be at least 2.")
    if args.min_cells_per_gene < 1:
        raise ValueError("--min-cells-per-gene must be positive.")
    if args.split_reps < 1:
        raise ValueError("--split-reps must be at least 1.")
    if args.split_seed_step <= 0:
        raise ValueError("--split-seed-step must be positive.")
    if args.offline_cached_data and args.force_download:
        raise ValueError("--offline-cached-data cannot be combined with --force-download.")

    args.started_at_utc = utc_now()
    out_dir = args.output_dir
    raw_dir = out_dir / "raw"
    figure_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    tarball_path = raw_dir / "pbmc3k_filtered_gene_bc_matrices.tar.gz"
    if args.offline_cached_data:
        source_probe = {
            "url": args.download_url,
            "method": "HEAD",
            "status": "skipped_offline_cached_data",
            "http_status": None,
            "elapsed_seconds": 0.0,
            "content_length": None,
            "content_type": "",
            "last_modified": "",
            "etag": "",
            "error_type": "",
            "error_message": "",
        }
        if not tarball_path.exists():
            raise RuntimeError(
                f"Cached PBMC3k tarball is absent at {tarball_path}; rerun without --offline-cached-data only after confirmation."
            )
        download_status = download_with_sha256(
            args.download_url,
            tarball_path,
            timeout=args.timeout,
            force=False,
        )
    else:
        source_probe = probe_url(args.download_url, timeout=args.timeout)
        download_status = download_with_sha256(
            args.download_url,
            tarball_path,
            timeout=args.timeout,
            force=args.force_download,
        )
    if download_status["status"] == "failed":
        raise RuntimeError(f"Download failed: {download_status['error_message']}")

    tenx = read_10x_tarball(tarball_path)
    embedding, embedding_metadata, variable_gene_rows = make_embedding(args, tenx)
    table_paths = write_embedding_tables(out_dir, args, embedding, embedding_metadata)
    variable_genes_csv = out_dir / "variable_genes_selected_for_pca.csv"
    write_csv(variable_genes_csv, variable_gene_rows)

    source_probe_json = out_dir / "source_probe_status.json"
    write_json(source_probe_json, {"probe": source_probe, "download": download_status})
    embedding_metadata_json = out_dir / "embedding_preprocessing_metadata.json"
    write_json(
        embedding_metadata_json,
        {
            key: value
            for key, value in embedding_metadata.items()
            if key not in {"selected_barcodes", "selected_original_indices", "library_size", "n_genes_by_counts", "pct_mito"}
        },
    )
    figure_payload = draw_embedding_figure(figure_dir, embedding, embedding_metadata, table_paths["split_csv"])

    n_train = int(round(args.train_fraction * embedding.shape[0]))
    n_test = int(embedding.shape[0] - n_train)
    density_status = None
    repeated_density_status = None
    if not args.skip_density_benchmark:
        density_status = run_density_benchmark(args, table_paths, out_dir, n_train, n_test)
        repeated_density_status = repeated_split_benchmark(args, embedding, embedding_metadata, out_dir)

    benchmark_summary = read_benchmark_summary(out_dir / "density_benchmark" / "benchmark_summary.csv")

    output_paths = {
        "raw_tarball": str(tarball_path.relative_to(ROOT)),
        "source_probe_status_json": str(source_probe_json.relative_to(ROOT)),
        "embedding_csv": str(table_paths["embedding_csv"].relative_to(ROOT)),
        "split_assignments_csv": str(table_paths["split_csv"].relative_to(ROOT)),
        "feature_summary_csv": str(table_paths["feature_summary_csv"].relative_to(ROOT)),
        "qc_summary_csv": str(table_paths["qc_summary_csv"].relative_to(ROOT)),
        "variable_genes_csv": str(variable_genes_csv.relative_to(ROOT)),
        "embedding_metadata_json": str(embedding_metadata_json.relative_to(ROOT)),
        "embedding_png": figure_payload["embedding_png"],
        "embedding_pdf": figure_payload["embedding_pdf"],
        "density_benchmark_dir": str((out_dir / "density_benchmark").relative_to(ROOT)),
        "report_md": str((out_dir / "application_execution_report.md").relative_to(ROOT)),
        "run_parameters_txt": str((out_dir / "run_parameters.txt").relative_to(ROOT)),
        "run_parameters_json": str((out_dir / "run_parameters.json").relative_to(ROOT)),
    }
    if repeated_density_status:
        output_paths.update(
            {
                "repeated_density_benchmark_dir": repeated_density_status["output_dir"],
                "repeated_density_summary": repeated_density_status["summary_path"],
                "repeated_density_diagnostics": repeated_density_status["diagnostics_path"],
                "repeated_density_report": repeated_density_status["report_path"],
            }
        )

    write_parameters(
        out_dir / "run_parameters.txt",
        out_dir / "run_parameters.json",
        args,
        output_paths,
        source_probe,
        download_status,
        tenx,
        embedding_metadata,
        density_status,
        repeated_density_status,
    )
    write_report(
        out_dir / "application_execution_report.md",
        args,
        output_paths,
        source_probe,
        download_status,
        tenx,
        embedding_metadata,
        density_status,
        repeated_density_status,
        benchmark_summary,
        figure_payload,
    )

    print(f"source_probe_status={source_probe.get('status')} http_status={source_probe.get('http_status')}")
    print(f"download_status={download_status.get('status')} bytes={download_status.get('bytes')}")
    print(f"matrix_shape_cells_by_genes={tenx.matrix.shape[0]}x{tenx.matrix.shape[1]}")
    print(f"selected_cells={embedding_metadata['selected_cells']} n_train={n_train} n_test={n_test}")
    print(f"genes_passing_min_cells={embedding_metadata['genes_passing_min_cells']}")
    print(f"n_variable_genes={embedding_metadata['n_variable_genes']}")
    print(f"density_benchmark_status={(density_status or {'status': 'skipped'})['status']}")
    print(f"repeated_density_status={(repeated_density_status or {'status': 'skipped'})['status']}")
    print(f"Wrote report: {output_paths['report_md']}")


if __name__ == "__main__":
    main()
