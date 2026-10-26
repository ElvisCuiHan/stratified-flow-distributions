"""Auditable train-only reference fits for the validation experiments."""

import hashlib
import json
import os
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import gaussian_kde
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from scripts.run_jmlr_density_benchmark import fit_gaussian, write_csv

ROOT = Path(__file__).resolve().parents[1]
GMM_OPTIONS = dict(covariance_type="full", reg_covar=1e-5, max_iter=2000,
                   tol=1e-5, n_init=1, init_params="kmeans")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x:
        x.item() if isinstance(x, np.generic) else x.tolist()
        if isinstance(x, np.ndarray) else str(x)) + "\n")


def begin_run(out, config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / "docs/46_validation_protocol_20261018.md",
               ROOT / "docs/48_validation_controls_addendum_20261018.md",
               ROOT / "scripts/sfd_validation_common.py",
               ROOT / "scripts/run_sfd_mechanism_experiment.py",
               ROOT / "scripts/run_estimated_skeleton_sfd_diagnostics.py",
               ROOT / config["runner"]]
    config["source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    config["runtime"] = dict(python=platform.python_version(), numpy=np.__version__,
                             scipy=scipy.__version__, sklearn=sklearn.__version__)
    config["command"] = sys.argv
    config["thread_env"] = {k: os.environ.get(k) for k in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    for path in sources:
        target = out / "source_snapshots" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    dump_json(out / "run_parameters.json", config)


def finish_run(out):
    out = Path(out)
    dump_json(out / "manifest.json", {"files": {
        str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "manifest.json"}})


def logpdf(fit, x):
    if hasattr(fit, "score_samples"):
        return fit.score_samples(x)
    return fit.logpdf(x)


class SelectedKDE:
    def __init__(self, x, factor):
        self.factor = factor
        self.kde = gaussian_kde(x.T, bw_method=lambda k: factor * k.scotts_factor())

    def logpdf(self, x):
        return self.kde.logpdf(x.T)


def fit_reference(method, train, seed):
    if method == "Gaussian":
        return fit_gaussian(train, reg=1e-5), []
    starts = []
    if method == "KDE":
        indices = np.random.default_rng(seed).permutation(len(train))
        cut = int(0.8 * len(train))
        inner, valid = train[indices[:cut]], train[indices[cut:]]
        for factor in (0.5, 1.0, 2.0):
            start = time.perf_counter()
            fit = SelectedKDE(inner, factor)
            starts.append({"start": factor, "selected": False, "converged": True,
                           "n_iter": 0, "selection_score": float(np.mean(fit.logpdf(valid))),
                           "selection_set": "inner_validation", "warnings": "",
                           "fit_seconds": time.perf_counter() - start})
        best = max(range(len(starts)), key=lambda i: starts[i]["selection_score"])
        starts[best]["selected"] = True
        return SelectedKDE(train, starts[best]["start"]), starts
    k = int(method.split("-")[1])
    fits = []
    for index in range(3):
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            fit = GaussianMixture(n_components=k, random_state=seed + index,
                                  **GMM_OPTIONS).fit(train)
        fits.append(fit)
        starts.append({"start": index, "selected": False, "converged": bool(fit.converged_),
                       "n_iter": int(fit.n_iter_), "selection_score": float(fit.score(train)),
                       "selection_set": "training", "warnings": "; ".join(str(w.message) for w in caught),
                       "fit_seconds": time.perf_counter() - start})
    best = max(range(len(starts)), key=lambda i: starts[i]["selection_score"])
    starts[best]["selected"] = True
    return fits[best], starts


def summary(rows, keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, members in sorted(groups.items()):
        row = dict(zip(keys, key))
        row["reps"] = len(members)
        for metric in metrics:
            values = np.asarray([r[metric] for r in members], dtype=float)
            row["mean_" + metric] = float(values.mean())
            row["se_" + metric] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        result.append(row)
    return result
