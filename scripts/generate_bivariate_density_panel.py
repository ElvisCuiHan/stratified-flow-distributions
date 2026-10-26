#!/usr/bin/env python3
"""Generate bivariate SFD density panels for the manuscript."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tmp" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / "tmp" / "cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "tmp" / "matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from flowdistribution import cartoon_face, ecg_like, maze_graph, snowflake
from sfd_figure_style import SFD_INK, panel_title, set_sfd_matplotlib_style


FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "sfd_density_histogram_fresh",
    [
        (0.00, "#fbfdfd"),
        (0.13, "#f2faf7"),
        (0.28, "#e4f3ed"),
        (0.44, "#ccebe2"),
        (0.60, "#8ad6c8"),
        (0.73, "#55c7b3"),
        (0.84, "#d9e8bd"),
        (0.94, "#f4c95d"),
        (1.00, "#ff8f73"),
    ],
    N=256,
)
CONTOUR_COLOR = "#72c7b8"
PANEL_FACE = "#fbfffd"
AXIS_COLOR = "#607080"
SPINE_COLOR = "#9aabb8"


def evaluate_grid(dist, xlim, ylim, *, resolution=155, n_grid=72, chunk=3500):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    dens = np.empty(pts.shape[0], dtype=float)
    for start in range(0, pts.shape[0], chunk):
        stop = min(pts.shape[0], start + chunk)
        dens[start:stop] = dist.bivariate_pdf(pts[start:stop], dims=(0, 1), n_grid=n_grid)
    return xx, yy, dens.reshape(xx.shape)


def draw_panel(ax, dist, title, xlim, ylim, *, resolution=190, n_grid=80, aspect="equal"):
    xx, yy, dens = evaluate_grid(dist, xlim, ylim, resolution=resolution, n_grid=n_grid)
    scale = np.log10(dens / max(float(np.nanmax(dens)), 1e-300) + 1e-4)
    levels = np.linspace(-4.0, 0.0, 56)
    ax.set_facecolor(PANEL_FACE)
    mesh = ax.contourf(
        xx,
        yy,
        scale,
        levels=levels,
        cmap=DENSITY_CMAP,
        extend="min",
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        scale,
        levels=np.linspace(-3.25, -0.35, 6),
        colors=CONTOUR_COLOR,
        linewidths=0.16,
        alpha=0.14,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(aspect, adjustable="box")
    panel_title(ax, title)
    ax.set_xlabel(r"$Y_1$", labelpad=2.0)
    ax.set_ylabel(r"$Y_2$", labelpad=2.0)
    ax.tick_params(labelsize=7.5, colors=AXIS_COLOR, width=0.55, length=2.5)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.55)
    return mesh


def main() -> None:
    set_sfd_matplotlib_style(plt, base_size=8.7, title_size=9.8, label_size=8.8, tick_size=7.5)
    specs = [
        (
            snowflake(d=3, sigma_shape=0.065, random_state=1),
            "a  Snowflake",
            (-1.85, 1.85),
            (-1.85, 1.85),
            "equal",
        ),
        (
            cartoon_face(d=3, sigma_curve=0.045, sigma_blob=0.060, random_state=1),
            "b  Stylized face",
            (-2.05, 2.05),
            (-2.05, 1.95),
            "equal",
        ),
        (
            maze_graph(d=3, sigma_edge=0.060, sigma_atom=0.095, random_state=1),
            "c  Maze graph",
            (-1.85, 1.85),
            (-1.45, 1.45),
            "equal",
        ),
        (
            ecg_like(sigma_x=0.010, sigma_y0=0.065, sigma_y1=0.090, random_state=1),
            "d  ECG-like flow",
            (-0.02, 1.02),
            (-0.80, 1.85),
            "auto",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.25), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.975, top=0.94, bottom=0.13, wspace=0.18, hspace=0.30)
    last_mesh = None
    for ax, spec in zip(axes.ravel(), specs):
        dist, title, xlim, ylim, aspect = spec
        last_mesh = draw_panel(ax, dist, title, xlim, ylim, aspect=aspect)

    cbar = fig.colorbar(
        last_mesh,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.035,
        pad=0.075,
        aspect=44,
    )
    cbar.set_label("relative log density", fontsize=8.2, color=SFD_INK, labelpad=2.5)
    cbar.ax.tick_params(labelsize=7.2, colors=AXIS_COLOR, width=0.55, length=2.4)
    cbar.outline.set_linewidth(0.55)
    cbar.outline.set_edgecolor(SPINE_COLOR)

    png = FIG_DIR / "fig_bivariate_density_panel.png"
    pdf = FIG_DIR / "fig_bivariate_density_panel.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
