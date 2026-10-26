"""Reflow saved external-density controls without fitting or rescoring."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTES = {
    "classical": {"Oracle DGP": "#3f6b57", "Gaussian": "#8ecae6", "KDE": "#d9b44a",
                  "GMM-2": "#b7a4df", "GMM-4": "#79b58c", "GMM-8": "#f28e88", "GMM-16": "#8f9aa8"},
    "copula": {"Copula oracle": "#5e8c61", "Gaussian": "#8fb9d6", "KDE": "#d4a85f",
               "GMM-2": "#b59bcf", "GMM-4": "#dc8f70", "GMM-8": "#9cbf88", "GMM-16": "#c7b36d"},
}
SCENARIOS = {
    "classical": [("mvnormal_correlated_10d", "Correlated Gaussian (10D)"),
                  ("student_t_10d", "Student-t (10D)"),
                  ("skew_mixture_10d", "Asymmetric mixture (10D)"),
                  ("gaussian_mixture_20d", "Gaussian mixture (20D)"),
                  ("banana_2d", "Banana transform (2D)")],
    "copula": [("gaussian_copula_lognormal_10d", "Gaussian / lognormal (10D)"),
               ("t_copula_t_margins_10d", "Student-t / Student-t (10D)"),
               ("clayton_copula_gamma_5d", "Clayton / gamma (5D)")],
}


def render(source, out):
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42}):
        for family in SCENARIOS:
            with (source / "evidence" / family / "benchmark_summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            fig, grid = plt.subplots(2 if family == "classical" else 1, 3,
                figsize=(7.2, 4.8 if family == "classical" else 2.65), squeeze=False,
                layout="constrained")
            for index, (name, title) in enumerate(SCENARIOS[family]):
                ax = grid.flat[index]
                selected = [r for r in rows if r["scenario"] == name]
                order = list(PALETTES[family])
                selected.sort(key=lambda r: order.index(r["method"]))
                ax.bar(range(len(selected)), [float(r["mean_heldout_log_score"]) for r in selected],
                       yerr=[float(r["se_heldout_log_score"]) for r in selected],
                       color=[PALETTES[family][r["method"]] for r in selected],
                       edgecolor="white", linewidth=.5,
                       error_kw={"elinewidth": .6, "ecolor": "#344955", "capsize": 1.6})
                ax.set_title(f"({chr(97 + index)}) {title}", loc="left", fontweight="bold", fontsize=8)
                ax.set_xticks(range(len(selected)), [r["method"] for r in selected], rotation=45, ha="right")
                ax.set_ylabel("Test log density")
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#e6ecee", linewidth=.5)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(length=2, width=.5)
            if family == "classical":
                grid.flat[-1].set_visible(False)
            for suffix in ("pdf", "png"):
                fig.savefig(out / f"{family}_density_controls.{suffix}", dpi=220)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    render(Path(__file__).resolve().parent, parser.parse_args().out)
