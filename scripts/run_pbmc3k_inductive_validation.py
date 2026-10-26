#!/usr/bin/env python3
"""Train-only PBMC representation and feature-held-out marker prediction."""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))

import numpy as np
from scipy import sparse
from scipy.special import logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_estimated_skeleton_sfd_diagnostics import fit_estimated_sfd
from scripts.run_pbmc3k_fitted_sfd_repeated_diagnostics import DEFAULT_RAW_TARBALL, MARKER_SETS
from scripts.run_qian_real_stochastic_biology_application import read_10x_tarball
from scripts.sfd_validation_common import (
    GMM_OPTIONS, begin_run, dump_json, finish_run, fit_reference, logpdf,
    sha, summary, write_csv,
)

MARKERS = {gene for genes in MARKER_SETS.values() for gene in genes}
DENSITY_METHODS = ("MST SFD", "Gaussian", "GMM-2", "GMM-4", "GMM-8", "KDE")
READOUTS = ("PCs", "PCs + density", "PCs + density + entropy", "PCs + density + SFD",
            "PCs + density + GMM memberships")
SFD_OPTIONS = dict(tube_landmarks=24, quadrature=12, min_sigma=1e-3, sigma_scale=1.0,
                   reg=1e-5, min_weight=1e-4, em_max_iter=200, em_tol=1e-6)


def predictor_counts(counts, symbols):
    keep = np.array([symbol.upper() not in MARKERS for symbol in symbols])
    result = sparse.csr_matrix(counts[:, keep], dtype=float)
    library = np.asarray(result.sum(axis=1)).ravel()
    result = result.multiply((10000 / np.maximum(library, 1))[:, None]).tocsr()
    result.data = np.log1p(result.data)
    return result, np.flatnonzero(keep), library


@dataclass
class Representation:
    genes: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    pca: PCA

    def transform(self, normalized):
        x = normalized[:, self.genes].toarray()
        return self.pca.transform(np.clip((x - self.mean) / self.sd, -10, 10))


def fit_representation(normalized, train_ids, n_features=1000, seed=0):
    train = normalized[train_ids]
    detected = np.asarray((train > 0).sum(axis=0)).ravel()
    mean = np.asarray(train.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(train.power(2).mean(axis=0)).ravel() - mean**2, 0)
    eligible = np.flatnonzero((detected >= 3) & (variance > 1e-12))
    if len(eligible) < 2:
        raise ValueError("Fewer than two training-eligible predictor genes")
    order = np.lexsort((eligible, -variance[eligible]))
    selected = eligible[order[:n_features]]
    x = train[:, selected].toarray()
    center, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    sd[sd < 1e-8] = 1
    x = np.clip((x - center) / sd, -10, 10)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed).fit(x)
    return Representation(selected, center, sd, pca)


def marker_targets(counts, symbols, library):
    group_scores = {}
    available = {}
    for group, genes in MARKER_SETS.items():
        indices = [i for i, symbol in enumerate(symbols) if symbol.upper() in genes]
        available[group] = [symbols[i] for i in indices]
        if not indices:
            raise ValueError(f"No cached markers for {group}")
        values = counts[:, indices].toarray() * (10000 / np.maximum(library, 1))[:, None]
        group_scores[group] = np.log1p(values).mean(axis=1)
    b = group_scores["B_marker_proxy"]
    others = np.max([v for k, v in group_scores.items() if k != "B_marker_proxy"], axis=0)
    return ((b >= 0.5) & (b > others)).astype(int), available


def logistic_predict(train_x, train_y, test_x, seed):
    fit_ids, valid_ids = train_test_split(np.arange(len(train_y)), test_size=0.25,
                                        stratify=train_y, random_state=seed)
    rows = []
    for c in (0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, max_iter=2000, tol=1e-8, random_state=seed))
        model.fit(train_x[fit_ids], train_y[fit_ids])
        value = log_loss(train_y[valid_ids], model.predict_proba(train_x[valid_ids]), labels=[0, 1])
        rows.append(dict(C=c, validation_log_loss=float(value), selected=False))
    best = min(range(len(rows)), key=lambda i: rows[i]["validation_log_loss"])
    rows[best]["selected"] = True
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=rows[best]["C"], max_iter=2000, tol=1e-8, random_state=seed))
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1], rows


def run(args):
    config = dict(vars(args), runner="scripts/run_pbmc3k_inductive_validation.py",
                  protocol="pbmc_inductive_marker_holdout_v2", raw_sha256=sha(args.raw),
                  sfd_options=SFD_OPTIONS, gmm_options=GMM_OPTIONS,
                  marker_sets=MARKER_SETS, marker_threshold=0.5,
                  evidence_stage="full_protocol" if (args.reps, args.n_features) == (25, 1000)
                  else "smoke_or_custom",
                  scope="cell splits; feature-held-out internal proxy, not external biological validation")
    begin_run(args.out, config)
    data = read_10x_tarball(args.raw)
    normalized, predictor_indices, library = predictor_counts(data.matrix, data.gene_symbols)
    valid = np.flatnonzero(library > 0)
    counts = data.matrix[valid]
    normalized, library = normalized[valid], library[valid]
    labels, available = marker_targets(counts, data.gene_symbols, library)
    dump_json(args.out / "marker_availability.json", available)
    scores, tasks, starts, split_rows, prediction_rows, gene_rows = [], [], [], [], [], []
    for rep in range(args.reps):
        seed = 20261020 + 10000 * rep
        ids = np.random.default_rng(seed).permutation(len(valid))
        cut = int(0.8 * len(valid))
        train_ids, test_ids = np.sort(ids[:cut]), np.sort(ids[cut:])
        representation = fit_representation(normalized, train_ids, args.n_features, seed)
        xy = representation.transform(normalized)
        train, test = xy[train_ids], xy[test_ids]
        raw_genes = predictor_indices[representation.genes]
        assert not any(data.gene_symbols[i].upper() in MARKERS for i in raw_genes)
        np.savez_compressed(args.out / f"preprocessing_rep_{rep:02d}.npz",
            raw_gene_indices=raw_genes, mean=representation.mean, sd=representation.sd,
            pca_components=representation.pca.components_, pca_mean=representation.pca.mean_,
            train_original_indices=valid[train_ids], test_original_indices=valid[test_ids])
        for rank, index in enumerate(raw_genes):
            gene_rows.append(dict(rep=rep, rank=rank, gene_id=data.gene_ids[index],
                                  gene_symbol=data.gene_symbols[index], raw_index=int(index)))
        for split, indices in (("train", train_ids), ("test", test_ids)):
            for index in indices:
                split_rows.append(dict(rep=rep, split=split, seed=seed,
                    barcode=data.barcodes[valid[index]], raw_index=int(valid[index]),
                    pc1=float(xy[index, 0]), pc2=float(xy[index, 1]),
                    marker_proxy=int(labels[index])))
        fits = {}
        model_parameters = {}
        for mi, method in enumerate(DENSITY_METHODS):
            start = time.perf_counter()
            if method == "MST SFD":
                fit = fit_estimated_sfd(train, "Estimated MST-skeleton SFD", seed=seed,
                                        args=argparse.Namespace(**SFD_OPTIONS))
                detail = [dict(start=0, selected=True, converged=fit.converged,
                    n_iter=fit.iterations, selection_score=float(np.mean(fit.logpdf(train))),
                    selection_set="training", warnings="mass-change heuristic stopping rule",
                    fit_seconds=time.perf_counter() - start)]
            else:
                fit, detail = fit_reference(method, train, seed + 100 + 10 * mi)
            fit_seconds = time.perf_counter() - start
            fits[method] = fit
            if method == "MST SFD":
                model_parameters[method] = dict(tube_mass=fit.tube_mass,
                    residual_mass=fit.residual_mass, residual_mean=fit.residual_mean,
                    residual_cov=fit.residual_cov, starts=fit.skeleton.starts,
                    ends=fit.skeleton.ends, lengths=fit.skeleton.lengths,
                    sigma=fit.skeleton.sigma)
            elif method.startswith("GMM"):
                model_parameters[method] = dict(weights=fit.weights_, means=fit.means_,
                                                covariances=fit.covariances_)
            elif method == "Gaussian":
                model_parameters[method] = dict(mean=fit.mean, covariance=fit.cov)
            else:
                model_parameters[method] = dict(scott_multiplier=fit.factor,
                                                covariance=fit.kde.covariance)
            selected = next((row for row in detail if row["selected"]), {})
            for row in detail:
                starts.append(dict(rep=rep, method=method, **row))
            evaluation_start = time.perf_counter()
            score = float(np.mean(logpdf(fit, test)))
            eval_seconds = time.perf_counter() - evaluation_start
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite inductive test score")
            scores.append(dict(rep=rep, method=method, protocol_version=config["protocol"],
                scenario="pbmc3k_inductive_pca2", scenario_family="marker_excluded_train_only_pca",
                seed=seed, n_train=len(train_ids), n_test=len(test_ids),
                heldout_log_score=score, fit_seconds=fit_seconds, eval_seconds=eval_seconds,
                status="ok" if selected.get("converged", True) else "warning",
                error_message=selected.get("warnings", ""),
                converged=selected.get("converged", True)))
        dump_json(args.out / f"density_parameters_rep_{rep:02d}.json", model_parameters)
        features = []
        for x in (train, test):
            kde = logpdf(fits["KDE"], x)[:, None]
            gmm_resp = fits["GMM-8"].predict_proba(x)
            entropy = -np.sum(gmm_resp * np.log(np.maximum(gmm_resp, 1e-15)), axis=1)[:, None]
            tube = fits["MST SFD"].responsibilities(x)[:, 0]
            tube_odds = logit(np.clip(tube, 1e-6, 1 - 1e-6))[:, None]
            features.append([x, np.c_[x, kde], np.c_[x, kde, entropy],
                             np.c_[x, kde, tube_odds], np.c_[x, kde, gmm_resp[:, :7]]])
        rep_tasks = []
        for index, readout in enumerate(READOUTS):
            prediction, detail = logistic_predict(features[0][index], labels[train_ids],
                                                   features[1][index], seed + 500)
            for row in detail:
                starts.append(dict(rep=rep, method=readout, **row))
            row = dict(rep=rep, readout=readout, n_test=len(test_ids),
                test_positives=int(labels[test_ids].sum()),
                log_loss=float(log_loss(labels[test_ids], prediction, labels=[0, 1])),
                auroc=float(roc_auc_score(labels[test_ids], prediction)),
                balanced_accuracy=float(balanced_accuracy_score(labels[test_ids], prediction >= 0.5)))
            rep_tasks.append(row)
            for idx, probability in zip(test_ids, prediction):
                prediction_rows.append(dict(rep=rep, readout=readout,
                    barcode=data.barcodes[valid[idx]], marker_proxy=int(labels[idx]),
                    predicted_probability=float(probability)))
        for row in rep_tasks:
            row["log_loss_delta_vs_density"] = row["log_loss"] - rep_tasks[1]["log_loss"]
        tasks.extend(rep_tasks)
        print(f"rep={rep} train={len(train_ids)} test={len(test_ids)} complete", flush=True)
    for name, rows in (("scores", scores), ("marker_prediction", tasks),
                       ("splits", split_rows), ("genes", gene_rows), ("predictions", prediction_rows)):
        write_csv(args.out / (name + ".csv"), rows)
    write_csv(args.out / "fit_and_selection.csv", starts,
              fieldnames=sorted(set().union(*(row.keys() for row in starts))))
    write_csv(args.out / "density_summary.csv", summary(scores, ["method"], ["heldout_log_score"]))
    write_csv(args.out / "marker_summary.csv", summary(tasks, ["readout"],
        ["log_loss", "auroc", "balanced_accuracy", "log_loss_delta_vs_density"]))
    finish_run(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_TARBALL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--n-features", type=int, default=1000)
    args = parser.parse_args()
    if args.reps < 1 or args.n_features < 2:
        parser.error("Need positive reps and at least two features")
    run(args)
