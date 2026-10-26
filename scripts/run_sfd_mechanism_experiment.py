#!/usr/bin/env python3
"""Separate mass, longitudinal occupancy, and width on two fixed SFD arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, ndtr, softmax
from scipy.stats import norm, truncnorm

from scripts.run_jmlr_density_benchmark import (
    fit_gaussian, fit_gmm_em, fit_kde, summarize, write_csv,
)

PROTOCOL = "sfd_mechanism_v1"
GROUPS = ("reference", "mass", "position", "width")
METHODS = (
    "Oracle", "SFD joint", "SFD fixed kernels", "SFD oracle kernels",
    "Gaussian", "GMM-4", "GMM-8", "KDE",
)
SFD_METHODS = METHODS[1:4]
TAU = 0.16
PARALLEL_SD = 0.025
RESIDUAL_MEAN = np.array([0.4, 0.0])
RESIDUAL_SD = 0.55
ANGLES = np.array([np.pi / 6, -np.pi / 6])
TANGENTS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
NORMALS = np.column_stack((-np.sin(ANGLES), np.cos(ANGLES)))
BOUNDS = [(-5.0, 5.0)] * 2 + [(0.1, 0.9)] * 2 + [(np.log(0.025), np.log(0.3))] * 2
OPT_OPTIONS = {"maxiter": 300, "ftol": 1e-10, "gtol": 1e-6, "maxls": 40}
LOG_2PI = np.log(2 * np.pi)


def log_interval(a, b):
    """Log normal interval probability, using the nearer tail."""
    a, b = np.broadcast_arrays(a, b)
    if np.any(a >= b):
        raise ValueError("Normal interval must have increasing endpoints")
    hi = np.where(a > 0, log_ndtr(-a), log_ndtr(b))
    lo = np.where(a > 0, log_ndtr(-b), log_ndtr(a))
    return hi + np.log(-np.expm1(lo - hi))


def coordinate_mean(m):
    a, b = -np.asarray(m) / TAU, (1 - np.asarray(m)) / TAU
    z = np.exp(log_interval(a, b))
    return np.asarray(m) + TAU * (norm.pdf(a) - norm.pdf(b)) / z


def longitudinal(u, m):
    total_var = TAU**2 + PARALLEL_SD**2
    post_sd = np.sqrt(TAU**2 * PARALLEL_SD**2 / total_var)
    post_mean = (PARALLEL_SD**2 * m + TAU**2 * u) / total_var
    a, b = -post_mean / post_sd, (1 - post_mean) / post_sd
    log_z = log_interval(a, b)
    value = norm.logpdf(u, m, np.sqrt(total_var)) + log_z
    value -= log_interval(-m / TAU, (1 - m) / TAU)
    post_expectation = post_mean + post_sd * (
        np.exp(-a**2 / 2 - LOG_2PI / 2 - log_z)
        - np.exp(-b**2 / 2 - LOG_2PI / 2 - log_z)
    )
    score_m = (post_expectation - coordinate_mean(m)) / TAU**2
    return value, score_m


@dataclass
class ArmLaw:
    weights: np.ndarray
    locations: np.ndarray
    widths: np.ndarray

    def components(self, x):
        x = np.asarray(x)
        u, v = x @ TANGENTS.T, x @ NORMALS.T
        along, score_m = longitudinal(u, self.locations)
        log_branches = along + norm.logpdf(v, 0, self.widths)
        residual = -np.sum((x - RESIDUAL_MEAN)**2, axis=1) / (2 * RESIDUAL_SD**2)
        residual -= LOG_2PI + 2 * np.log(RESIDUAL_SD)
        return np.column_stack((log_branches, residual)), score_m, v

    def logpdf(self, x):
        cols = self.components(x)[0] + np.log(self.weights)
        return logsumexp(cols, axis=1)

    def responsibilities(self, x):
        cols = self.components(x)[0] + np.log(self.weights)
        return np.exp(cols - logsumexp(cols, axis=1)[:, None])

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        labels = rng.choice(3, size=n, p=self.weights)
        x = rng.normal(RESIDUAL_MEAN, RESIDUAL_SD, size=(n, 2))
        for branch in range(2):
            mask = labels == branch
            m = self.locations[branch]
            s = truncnorm.rvs(-m / TAU, (1 - m) / TAU, loc=m, scale=TAU,
                              size=mask.sum(), random_state=rng)
            u = s + rng.normal(0, PARALLEL_SD, size=mask.sum())
            v = rng.normal(0, self.widths[branch], size=mask.sum())
            x[mask] = u[:, None] * TANGENTS[branch] + v[:, None] * NORMALS[branch]
        return x, labels

    def positive_y(self):
        nodes, weights = np.polynomial.legendre.leggauss(80)
        s, weights = (nodes + 1) / 2, weights / 2
        result = self.weights[2] * 0.5
        for branch in range(2):
            m = self.locations[branch]
            h = truncnorm.pdf(s, -m / TAU, (1 - m) / TAU, loc=m, scale=TAU)
            sd_y = np.sqrt((TANGENTS[branch, 1] * PARALLEL_SD)**2
                           + (NORMALS[branch, 1] * self.widths[branch])**2)
            result += self.weights[branch] * np.sum(
                weights * h * ndtr(s * TANGENTS[branch, 1] / sd_y)
            )
        return float(result)


def truth(group):
    if group not in GROUPS:
        raise ValueError(group)
    return ArmLaw(
        np.array([0.6, 0.2, 0.2] if group == "mass" else [0.4, 0.4, 0.2]),
        np.array([0.7, 0.5] if group == "position" else [0.5, 0.5]),
        np.array([0.14, 0.07] if group == "width" else [0.07, 0.07]),
    )


def unpack(parameters):
    return ArmLaw(softmax(np.r_[parameters[:2], 0.0]),
                  np.asarray(parameters[2:4]), np.exp(parameters[4:6]))


def objective(parameters, x):
    law = unpack(parameters)
    components, score_m, v = law.components(x)
    cols = components + np.log(law.weights)
    total = logsumexp(cols, axis=1)
    resp = np.exp(cols - total[:, None])
    gradient = np.r_[
        np.mean(resp[:, :2], axis=0) - law.weights[:2],
        np.mean(resp[:, :2] * score_m, axis=0),
        np.mean(resp[:, :2] * (v**2 / law.widths**2 - 1), axis=0),
    ]
    return -float(np.mean(total)), -gradient


def fit_sfd(x, *, fixed_kernels=None):
    free = np.arange(6) if fixed_kernels is None else np.arange(2)
    starts = []
    for locations, widths in [([0.5, 0.5], [0.08, 0.08]),
                              ([0.35, 0.65], [0.12, 0.12]),
                              ([0.65, 0.35], [0.12, 0.12])]:
        if fixed_kernels is not None:
            locations, widths = fixed_kernels.locations, fixed_kernels.widths
        base = np.r_[np.log([2.0, 2.0]), locations, np.log(widths)]

        def fun(values):
            p = base.copy()
            p[free] = values
            value, gradient = objective(p, x)
            return value, gradient[free]

        start = time.perf_counter()
        result = minimize(fun, base[free], jac=True, method="L-BFGS-B",
                          bounds=[BOUNDS[i] for i in free], options=OPT_OPTIONS)
        p = base.copy()
        p[free] = result.x
        starts.append((result, p, time.perf_counter() - start))
    finite = [(i, item) for i, item in enumerate(starts) if np.isfinite(item[0].fun)]
    if not finite:
        raise FloatingPointError("All SFD starts returned nonfinite objectives")
    selected, (result, parameters, _) = min(finite, key=lambda item: item[1][0].fun)
    diagnostics = []
    for i, (fit, p, seconds) in enumerate(starts):
        diagnostics.append({"start": i, "selected": i == selected,
                            "converged": bool(fit.success), "n_iter": int(fit.nit),
                            "train_mean_log_score": -float(fit.fun),
                            "gradient_max": float(np.max(np.abs(fit.jac))),
                            "fit_seconds": seconds, "message": str(fit.message),
                            "boundary_contact": any(
                                abs(p[j] - BOUNDS[j][0]) < 1e-5
                                or abs(p[j] - BOUNDS[j][1]) < 1e-5 for j in free)})
    return unpack(parameters), diagnostics


def probability_positive_y(fit):
    if isinstance(fit, ArmLaw):
        return fit.positive_y()
    if hasattr(fit, "kde"):
        return float(np.mean(ndtr(fit.kde.dataset[1] / np.sqrt(fit.kde.covariance[1, 1]))))
    if hasattr(fit, "weights"):
        return float(np.sum(fit.weights * ndtr(fit.means[:, 1] / np.sqrt(fit.covs[:, 1, 1]))))
    return float(ndtr(fit.mean[1] / np.sqrt(fit.cov[1, 1])))


def fit_method(method, train, law, seed):
    if method == "Oracle":
        return law, []
    if method in SFD_METHODS:
        fixed = None if method == "SFD joint" else truth("reference")
        if method == "SFD oracle kernels":
            fixed = law
        return fit_sfd(train, fixed_kernels=fixed)
    if method == "Gaussian":
        return fit_gaussian(train, reg=1e-5), []
    if method == "KDE":
        return fit_kde(train, random_state=seed, jitter=1e-6), []
    k = int(method.split("-")[1])
    starts = [fit_gmm_em(train, k, random_state=seed + i, reg=1e-5,
                         max_iter=200, tol=1e-6) for i in range(2)]
    scores = [float(np.mean(fit.logpdf(train))) for fit in starts]
    selected = int(np.argmax(scores))
    diagnostics = [{"start": i, "selected": i == selected,
                    "converged": fit.converged, "n_iter": fit.n_iter,
                    "train_mean_log_score": scores[i], "gradient_max": "",
                    "fit_seconds": "", "message": "", "boundary_contact": ""}
                   for i, fit in enumerate(starts)]
    return starts[selected], diagnostics


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def density_plot_summary(scores, method):
    return {row["scenario"]: row for row in summarize(
        [row for row in scores if row["method"] == method]
    )}


def recovery_row(fit, law, test, labels):
    resp, oracle_resp = fit.responsibilities(test), law.responsibilities(test)
    return {
        "upper_mass": float(fit.weights[0]), "true_upper_mass": float(law.weights[0]),
        "upper_position": float(coordinate_mean(fit.locations[0])),
        "true_upper_position": float(coordinate_mean(law.locations[0])),
        "upper_width": float(fit.widths[0]), "true_upper_width": float(law.widths[0]),
        "mass_rmse": float(np.sqrt(np.mean((fit.weights - law.weights)**2))),
        "responsibility_rmse": float(np.sqrt(np.mean((resp - oracle_resp)**2))),
        "brier_score": float(np.mean(np.sum((resp - np.eye(3)[labels])**2, axis=1))),
        "stratum_accuracy": float(np.mean(resp.argmax(axis=1) == labels)),
    }


def contrast_rows(recovery):
    indexed = {(r["scenario"], r["method"], r["rep"]): r for r in recovery}
    rows = []
    for key, row in indexed.items():
        group, method, rep = key
        if group == "reference" or ("reference", method, rep) not in indexed:
            continue
        baseline = indexed["reference", method, rep]
        for metric in ("upper_mass", "upper_position", "upper_width"):
            true_delta = row["true_" + metric] - baseline["true_" + metric]
            estimate = row[metric] - baseline[metric]
            rows.append({"scenario": group, "method": method, "rep": rep,
                         "metric": metric, "true_contrast": true_delta,
                         "estimated_contrast": estimate, "error": estimate - true_delta})
    return rows


def summarize_contrasts(rows):
    result = []
    for group in GROUPS[1:]:
        for method in SFD_METHODS:
            for metric in ("upper_mass", "upper_position", "upper_width"):
                sub = [r for r in rows if (r["scenario"], r["method"], r["metric"])
                       == (group, method, metric)]
                if not sub:
                    continue
                mean, se = mean_se([r["estimated_contrast"] for r in sub])
                errors = np.array([r["error"] for r in sub])
                result.append({"scenario": group, "method": method, "metric": metric,
                               "true_contrast": sub[0]["true_contrast"], "mean_contrast": mean,
                               "se_contrast": se, "bias": float(errors.mean()),
                               "rmse": float(np.sqrt(np.mean(errors**2))), "reps": len(sub)})
    return result


def export_tables(out, summary, contrasts):
    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Change & Fit & $\Delta\pi_+$ & $\Delta\mathbb E(S\mid Z=+)$ & $\Delta\sigma_+$ \\",
             r"\midrule"]
    names = {"SFD joint": "Joint", "SFD fixed kernels": "Fixed kernels",
             "SFD oracle kernels": "Oracle kernels"}
    for group in GROUPS[1:]:
        for method in SFD_METHODS:
            cells = [group.capitalize(), names[method]]
            for metric in ("upper_mass", "upper_position", "upper_width"):
                row = next(r for r in contrasts if (r["scenario"], r["method"], r["metric"])
                           == (group, method, metric))
                cells.append(f"{row['mean_contrast']:.3f} ({row['se_contrast']:.3f})")
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "mechanism_contrasts.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{longtable}{@{}llrr@{}}",
             r"\caption{Fixed-geometry test log scores over " + str(summary[0]["reps"])
             + r" replicates. Entries are means (Monte Carlo SE); warning rates report selected fits reaching the iteration limit.}\label{tab:oa-mechanism-density}\\",
             r"\toprule",
             r"Group & Fit & Mean test log score (SE) & Warning rate \\", r"\midrule",
             r"\endhead"]
    for group in GROUPS:
        for method in METHODS:
            row = next(r for r in summary if (r["scenario"], r["method"]) == (group, method))
            score, se = row["mean_heldout_log_score"], row["se_heldout_log_score"]
            cell = f"{score:.3f} ({se:.3f})" if score is not None else "Failed"
            lines.append(f"{group.capitalize()} & {method} & {cell} & {row['warning_rate']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    (out / "mechanism_density.tex").write_text("\n".join(lines) + "\n")


def plot_results(out, contrasts, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.sfd_figure_style import panel_title, set_sfd_matplotlib_style
    set_sfd_matplotlib_style(plt)
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.875))
    grid_x, grid_y = np.meshgrid(np.linspace(-0.15, 1.05, 160), np.linspace(-0.65, 0.65, 150))
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    reference = truth("reference").logpdf(grid).reshape(grid_x.shape)
    labels = ["Mass", "Position", "Width"]
    for col, group in enumerate(GROUPS[1:]):
        ax = axes[0, col]
        density = truth(group).logpdf(grid).reshape(grid_x.shape)
        ax.contour(grid_x, grid_y, reference, levels=[-1, 0, 1], colors="#a1a7aa",
                   linewidths=0.8, linestyles="dashed")
        ax.contour(grid_x, grid_y, density, levels=[-1, 0, 1], colors="#147d79",
                   linewidths=1.1, linestyles="solid")
        for tangent in TANGENTS:
            ax.plot([0, tangent[0]], [0, tangent[1]], color="#353a40", lw=0.6, alpha=0.55)
        ax.set(xlim=(-0.15, 1.05), ylim=(-0.65, 0.65), xlabel=r"$x_1$", ylabel=r"$x_2$")
        ax.set_aspect("equal")
        panel_title(ax, f"{chr(97 + col)}  {labels[col]} change")
    metrics = ("upper_mass", "upper_position", "upper_width")
    titles = (r"Branch mass $\pi_+$", "Mean position", r"Tube width $\sigma_+$")
    for col, metric in enumerate(metrics):
        ax = axes[1, col]
        rows = [next(r for r in contrasts if r["scenario"] == g and r["method"] == "SFD joint"
                     and r["metric"] == metric) for g in GROUPS[1:]]
        x = np.arange(3)
        ax.axhline(0, color="#b9bfc3", lw=0.7)
        ax.scatter(x - 0.06, [r["true_contrast"] for r in rows], marker="x", s=35,
                   color="#333333", label="True contrast", zorder=4)
        ax.errorbar(x + 0.06, [r["mean_contrast"] for r in rows],
                    yerr=[1.96 * r["se_contrast"] for r in rows], fmt="o", markersize=4,
                    color="#147d79", capsize=3, label="Joint fit: mean and MC interval")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Change from reference")
        panel_title(ax, f"{chr(100 + col)}  " + titles[col])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].plot([], [], "--", color="#a1a7aa", label="Reference law")
    axes[0, 0].plot([], [], "-", color="#147d79", label="Changed law")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, fontsize=7)
    handles, legend_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"mechanism_separation.{ext}", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.16, 1, 0.84))
    colors = ["#333333", "#147d79", "#c05b45", "#638eb0", "#8c7a9b", "#d69d42", "#679f57", "#b2618d"]
    for j, method in enumerate(METHODS):
        offset = (j - 3.5) * 0.08
        summary = density_plot_summary(scores, method)
        axes[0].errorbar(np.arange(4) + offset,
                         [summary[g]["mean_heldout_log_score"] for g in GROUPS],
                         yerr=[1.96 * summary[g]["se_heldout_log_score"] for g in GROUPS],
                         fmt="o", ms=3, capsize=1.5, color=colors[j], label=method)
        events = [[r["positive_y_error"] for r in scores
                   if r["scenario"] == g and r["method"] == method and r["status"] != "failed"]
                  for g in GROUPS]
        means_ses = [mean_se(values) for values in events]
        axes[1].errorbar(np.arange(4) + offset, [r[0] for r in means_ses],
                         yerr=[1.96 * r[1] for r in means_ses], fmt="o", ms=3,
                         capsize=1.5, color=colors[j])
    for ax in axes:
        ax.set_xticks(np.arange(4), ["Reference", "Mass", "Position", "Width"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test mean log density")
    axes[1].set_ylabel(r"Error in $\mathbb{P}(X_2>0)$")
    axes[1].axhline(0, color="#b9bfc3", lw=0.7)
    panel_title(axes[0], "a  Density prediction")
    panel_title(axes[1], "b  Observable event probability")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", frameon=False, fontsize=7, ncol=4)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"mechanism_prediction.{ext}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/sfd_mechanism_v1")
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        def read_rows(name, numeric):
            with (args.out / name).open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                for key in numeric:
                    row[key] = float(row[key]) if row[key] else None
            return rows

        contrast_data = read_rows("contrast_summary.csv", ["true_contrast", "mean_contrast", "se_contrast"])
        score_data = read_rows("heldout_log_scores.csv", ["heldout_log_score", "fit_seconds",
                              "eval_seconds", "positive_y_error"])
        input_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in args.out.glob("*.csv")}
        plot_results(args.out, contrast_data, score_data)
        assert input_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in args.out.glob("*.csv")}
        (args.out / "plot_export.json").write_text(json.dumps({
            "operation": "plot-only", "numerical_inputs_unchanged": input_hashes,
            "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }, indent=2) + "\n")
        return
    if args.reps < 1 or min(args.n_train, args.n_test) < 20:
        parser.error("Need positive replicates and at least 20 rows per split")
    args.out.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / "docs/44_mechanism_experiment_protocol_20261018.md"
    config = {"protocol_version": PROTOCOL, "reps": args.reps, "n_train": args.n_train,
              "n_test": args.n_test, "seed": args.seed, "groups": GROUPS, "methods": METHODS,
              "optimizer": OPT_OPTIONS, "bounds": BOUNDS, "tau": TAU,
              "parallel_sd": PARALLEL_SD, "angles_radians": ANGLES.tolist(),
              "residual_mean": RESIDUAL_MEAN.tolist(), "residual_sd": RESIDUAL_SD,
              "command": sys.argv, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "thread_env": {key: os.getenv(key) for key in
                             ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    (args.out / "run_parameters.json").write_text(json.dumps(config, indent=2) + "\n")
    scores, recovery, starts, hashes, parameters = [], [], [], [], []
    for rep in range(args.reps):
        for g, group in enumerate(GROUPS):
            seed = args.seed + 100000 * rep + 1000 * g
            law = truth(group)
            train, _ = law.sample(args.n_train, seed + 1)
            test, labels = law.sample(args.n_test, seed + 2)
            hashes.append({"scenario": group, "rep": rep, "seed": seed,
                           "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
                           "test_sha256": hashlib.sha256(test.tobytes()).hexdigest()})
            for method_index, method in enumerate(METHODS):
                start = time.perf_counter()
                row = {"protocol_version": PROTOCOL, "scenario": group,
                       "scenario_family": "fixed_geometry_mechanism", "rep": rep,
                       "n_train": args.n_train, "n_test": args.n_test, "method": method,
                       "seed": seed, "heldout_log_score": None, "fit_seconds": 0.0,
                       "eval_seconds": 0.0, "status": "failed", "converged": False,
                       "n_iter": 0, "error_message": "", "positive_y_probability": None,
                       "true_positive_y_probability": law.positive_y(), "positive_y_error": None}
                try:
                    fit, diagnostics = fit_method(method, train, law, seed + 100 + 10 * method_index)
                    row["fit_seconds"] = time.perf_counter() - start
                    start = time.perf_counter()
                    values = fit.logpdf(test)
                    if not np.all(np.isfinite(values)):
                        raise FloatingPointError("Nonfinite test log density")
                    probability = probability_positive_y(fit)
                    row.update(heldout_log_score=float(values.mean()),
                               positive_y_probability=probability,
                               positive_y_error=probability - row["true_positive_y_probability"])
                    row["eval_seconds"] = time.perf_counter() - start
                    selected = next((d for d in diagnostics if d["selected"]), {})
                    converged = selected.get("converged", True)
                    row.update(status="ok" if converged else "warning", converged=converged,
                               n_iter=selected.get("n_iter", 0))
                    starts.extend(dict(scenario=group, rep=rep, method=method, **d) for d in diagnostics)
                    if method in SFD_METHODS:
                        recovery.append(dict(scenario=group, rep=rep, method=method,
                                             **recovery_row(fit, law, test, labels)))
                        for component in range(3):
                            parameters.append({"scenario": group, "rep": rep, "method": method,
                                               "component": component, "weight": fit.weights[component],
                                               "location": fit.locations[component] if component < 2 else "",
                                               "width": fit.widths[component] if component < 2 else ""})
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    row["error_message"] = f"{type(error).__name__}: {error}"
                scores.append(row)
            print(f"rep={rep + 1}/{args.reps} group={group}", flush=True)
    summary = summarize(scores)
    contrasts = contrast_rows(recovery)
    contrast_summary = summarize_contrasts(contrasts)
    for name, rows in [("heldout_log_scores.csv", scores), ("density_summary.csv", summary),
                       ("recovery.csv", recovery), ("optimizer_starts.csv", starts),
                       ("sample_hashes.csv", hashes), ("fitted_parameters.csv", parameters),
                       ("contrasts.csv", contrasts), ("contrast_summary.csv", contrast_summary)]:
        if rows:
            write_csv(args.out / name, rows)
    export_tables(args.out, summary, contrast_summary)
    plot_results(args.out, contrast_summary, scores)
    manifest = {"files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.out.iterdir()) if p.is_file()},
                "runs": len(scores), "failed_runs": sum(r["status"] == "failed" for r in scores),
                "warning_runs": sum(r["status"] == "warning" for r in scores)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), **{k: v for k, v in manifest.items() if k != "files"}}))


if __name__ == "__main__":
    main()
