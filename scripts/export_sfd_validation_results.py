#!/usr/bin/env python3
"""Export all prespecified validation comparisons without refitting models."""

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from scripts.sfd_figure_style import panel_title, set_sfd_matplotlib_style
from scripts.sfd_validation_common import dump_json, sha, summary, write_csv

CONDITIONS = ("nominal", "overlap", "heavy_noise")
CONDITION_NAMES = ("Gaussian", "Overlap", "Student-t")
SFD = ("SFD supplied", "SFD rotated", "SFD learned angles")
COLORS = ("#168B7B", "#D76654", "#3C72A8")
METRICS = ("upper_mass", "upper_position", "upper_width")
SCENARIOS = ("mass", "position", "width")
READOUTS = ("PCs", "PCs + density", "PCs + density + entropy", "PCs + density + SFD",
            "PCs + density + GMM memberships")
READOUT_NAMES = ("PCs", "+ KDE density", "+ GMM entropy", "+ SFD tube odds", "+ GMM memberships")


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_se(values):
    x = np.asarray(values, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def cell(values):
    mean, se = mean_se(values)
    return f"{mean:.4f} ({se:.4f})"


def longtable(path, caption, label, headings, rows):
    columns = "l" * (len(headings) - 1) + "r"
    text = [r"{\scriptsize\setlength{\tabcolsep}{3pt}",
            r"\begin{longtable}{@{}" + columns + r"@{}}",
            r"\caption{" + caption + r"}\label{" + label + r"}\\",
            r"\toprule", " & ".join(headings) + r" \\", r"\midrule\endfirsthead",
            r"\toprule", " & ".join(headings) + r" \\", r"\midrule\endhead",
            r"\bottomrule\endfoot"]
    text.extend(" & ".join(str(v) for v in row) + r" \\" for row in rows)
    text.extend([r"\end{longtable}", "}"])
    path.write_text("\n".join(text) + "\n")


def export(args):
    args.out.mkdir(parents=True, exist_ok=False)
    for folder in (args.geometry, args.pbmc):
        import json
        config = json.loads((folder / "run_parameters.json").read_text())
        if config["evidence_stage"] != "full_protocol" or config["reps"] != 25:
            raise ValueError("Paper exporter requires both complete frozen 25-repeat protocols")
    input_hashes = {str(path.relative_to(ROOT)): sha(path)
                    for folder in (args.geometry, args.pbmc) for path in folder.glob("*.csv")}
    contrasts = read(args.geometry / "contrasts.csv")
    recovery = read(args.geometry / "recovery.csv")
    scores = read(args.geometry / "scores.csv")
    tasks = read(args.pbmc / "marker_prediction.csv")
    pbmc_scores = read(args.pbmc / "scores.csv")
    assert len(scores) == 2100 and len(contrasts) == 2025
    assert len(tasks) == 125 and len(pbmc_scores) == 150
    selected = lambda condition, scenario, method, metric: [float(r["estimate"]) for r in contrasts
        if (r["condition"], r["scenario"], r["method"], r["metric"]) == (condition, scenario, method, metric)]
    set_sfd_matplotlib_style(plt)
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.7), layout="constrained")
    panel_specs = [("mass", "upper_mass", "A. Mass change"),
                   ("position", "upper_position", "B. Position change"),
                   ("width", "upper_width", "C. Width change"),
                   ("position", "upper_mass", "D. Mass under position change"),
                   ("width", "upper_mass", "E. Mass under width change")]
    for ax, (scenario, metric, title) in zip(axes.flat, panel_specs):
        target = float(next(r["truth"] for r in contrasts if r["scenario"] == scenario and r["metric"] == metric))
        ax.axhline(target, color="0.3", lw=0.9, ls="--")
        for mi, (method, color) in enumerate(zip(SFD, COLORS)):
            values = [mean_se(selected(c, scenario, method, metric)) for c in CONDITIONS]
            ax.errorbar(np.arange(3) + (mi - 1) * 0.16, [v[0] for v in values],
                        yerr=[1.96 * v[1] for v in values], fmt="o", ms=3, capsize=2,
                        color=color, label=method.removeprefix("SFD "))
        ax.set_xticks(np.arange(3), CONDITION_NAMES, fontsize=7)
        ax.set_ylabel("Estimated contrast")
        panel_title(ax, title, size=9)
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1, 2]
    for mi, (method, color) in enumerate(zip(SFD, COLORS)):
        values = [mean_se([r["angle_rmse_degrees"] for r in recovery
                           if r["condition"] == c and r["scenario"] == "reference" and r["method"] == method])
                  for c in CONDITIONS]
        ax.errorbar(np.arange(3) + (mi - 1) * 0.16, [v[0] for v in values],
                    yerr=[1.96 * v[1] for v in values], fmt="o", ms=3, capsize=2, color=color)
    ax.set_xticks(np.arange(3), CONDITION_NAMES, fontsize=7)
    ax.set_ylabel("Angle RMSE (degrees)")
    panel_title(ax, "F. Reference-group geometry", size=9)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    for extension in ("pdf", "png"):
        fig.savefig(args.out / ("geometry_validation." + extension), dpi=220)
    plt.close(fig)

    density_methods = ("Gaussian", "GMM-2", "GMM-4", "GMM-8", "KDE", "MST SFD")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), layout="constrained")
    for i, method in enumerate(density_methods):
        mean, se = mean_se([r["heldout_log_score"] for r in pbmc_scores if r["method"] == method])
        axes[0].errorbar(mean, i, xerr=1.96 * se, fmt="o", capsize=2,
                         color=COLORS[0] if method == "MST SFD" else "#6C7680", ms=4)
    axes[0].set_yticks(range(6), density_methods)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Held-out log density")
    panel_title(axes[0], "A. Training-fitted PCA representation", size=9)
    axes[1].axvline(0, color="0.4", lw=0.9, ls="--")
    for i, name in enumerate(READOUTS):
        mean, se = mean_se([r["log_loss_delta_vs_density"] for r in tasks if r["readout"] == name])
        axes[1].errorbar(mean, i, xerr=1.96 * se, fmt="o", capsize=2, ms=4,
                        color=COLORS[0] if i == 3 else COLORS[1] if i == 4 else "#6C7680")
    axes[1].set_yticks(range(5), READOUT_NAMES)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Log-loss change vs PCs + density")
    panel_title(axes[1], "B. Feature-held-out marker proxy", size=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for extension in ("pdf", "png"):
        fig.savefig(args.out / ("pbmc_inductive_validation." + extension), dpi=220)
    plt.close(fig)

    table_rows = []
    for condition, name in zip(CONDITIONS, CONDITION_NAMES):
        for scenario in SCENARIOS:
            for method in SFD:
                for metric, short in zip(METRICS, ("Mass", "Position", "Width")):
                    subset = [r for r in contrasts if (r["condition"], r["scenario"], r["method"], r["metric"])
                              == (condition, scenario, method, metric)]
                    assert len(subset) == 25
                    error = np.asarray([r["error"] for r in subset], dtype=float)
                    table_rows.append([name, scenario.capitalize(), method.removeprefix("SFD "), short,
                                       cell([r["estimate"] for r in subset]),
                                       f"{error.mean():.4f}", f"{np.sqrt(np.mean(error**2)):.4f}"])
    longtable(args.out / "geometry_contrasts.tex",
        "Geometry and noise stress contrasts over 25 repetitions. Entries are means (Monte Carlo SE); bias and RMSE compare with the generating-law contrast. Width denotes transverse standard deviation.",
        "tab:oa-geometry-contrasts", ["Noise", "Change", "Fit", "Target", "Estimate (SE)", "Bias", "RMSE"], table_rows)
    density_rows = []
    for condition, name in zip(CONDITIONS, CONDITION_NAMES):
        for scenario in ("reference",) + SCENARIOS:
            for method in ("SFD supplied", "SFD rotated", "SFD learned angles", "Gaussian", "GMM-4", "GMM-8", "KDE"):
                subset = [r for r in scores if (r["condition"], r["scenario"], r["method"]) == (condition, scenario, method)]
                assert len(subset) == 25
                density_rows.append([name, scenario.capitalize(), method.removeprefix("SFD "),
                    cell([r["heldout_log_score"] for r in subset]),
                    f"{sum(r['converged'] == 'False' for r in subset)}/25"])
    longtable(args.out / "geometry_density.tex", "Held-out density scores in the geometry stress study. Entries are means (Monte Carlo SE); the final column counts selected fits not meeting their stopping criterion.",
        "tab:oa-geometry-density", ["Noise", "Group", "Fit", "Log score (SE)", "Warnings"], density_rows)
    longtable(args.out / "pbmc_density.tex", "PBMC3k density scores over 25 training-fitted representations. Entries are means (split Monte Carlo SE).",
        "tab:oa-pbmc-inductive-density", ["Fit", "Log score (SE)", "Warnings"],
        [[method, cell([r["heldout_log_score"] for r in pbmc_scores if r["method"] == method]),
          f"{sum(r['converged'] == 'False' for r in pbmc_scores if r['method'] == method)}/25"] for method in density_methods])
    longtable(args.out / "pbmc_marker_prediction.tex", "Feature-held-out B-marker prediction over 25 cell splits. Entries are means (split Monte Carlo SE). Delta is paired log loss minus the PCs-plus-density reference; it is not an independent-subject effect.",
        "tab:oa-pbmc-marker-prediction", ["Readout", "Log loss (SE)", "AUROC (SE)", "Balanced acc.", "Delta (SE)"],
        [[label, cell([r["log_loss"] for r in tasks if r["readout"] == name]),
          cell([r["auroc"] for r in tasks if r["readout"] == name]),
          cell([r["balanced_accuracy"] for r in tasks if r["readout"] == name]),
          cell([r["log_loss_delta_vs_density"] for r in tasks if r["readout"] == name])]
         for name, label in zip(READOUTS, READOUT_NAMES)])
    paired = []
    indexed = {(r["rep"], r["readout"]): r for r in tasks}
    for rep in range(25):
        for method in READOUTS:
            for control in ("PCs + density", "PCs + density + GMM memberships"):
                paired.append(dict(rep=rep, readout=method, control=control,
                    log_loss_difference=float(indexed[str(rep), method]["log_loss"])
                    - float(indexed[str(rep), control]["log_loss"])))
    write_csv(args.out / "paired_marker_differences.csv", paired)
    write_csv(args.out / "paired_marker_summary.csv", summary(paired,
        ["readout", "control"], ["log_loss_difference"]))
    assert all(sha(ROOT / path) == digest for path, digest in input_hashes.items())
    dump_json(args.out / "export_manifest.json", dict(operation="read-only numerical export",
        input_sha256=input_hashes, exporter_sha256=sha(Path(__file__)),
        output_sha256={p.name: sha(p) for p in args.out.iterdir() if p.is_file()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=ROOT / "outputs/sfd_geometry_stress_v1")
    parser.add_argument("--pbmc", type=Path, default=ROOT / "outputs/pbmc_inductive_marker_holdout_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/sfd_validation_paper_20261018")
    export(parser.parse_args())
