#!/usr/bin/env python3
"""Generate 3D SFD density-surface panels for the manuscript."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tmp" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / "tmp" / "cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from flowdistribution import cartoon_face, ecg_like, maze_graph, snowflake
from sfd_figure_style import SFD_FONT_FAMILY, SFD_INK, set_sfd_matplotlib_style


FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "sfd_density_histogram_fresh_surface",
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
PANE_COLOR = (0.99, 0.997, 0.995, 1.0)
TITLE_COLOR = SFD_INK
BASE_CONTOUR_COLOR = "#72c7b8"
SURFACE_FONT_FAMILY = SFD_FONT_FAMILY


def surface_panel_title(ax, text: str, *, pad: float = 0.0, size: float = 9.5) -> None:
    ax.set_title(
        text,
        loc="left",
        pad=pad,
        fontsize=size,
        fontweight="bold",
        fontfamily=SURFACE_FONT_FAMILY,
        color=SFD_INK,
    )


def softly_light_surface(facecolors: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Apply a restrained display-only light pass to improve surface depth."""
    light = colors.LightSource(azdeg=315, altdeg=42)
    shaded = light.shade_rgb(facecolors[..., :3], height, blend_mode="soft", fraction=0.50)
    facecolors = facecolors.copy()
    facecolors[..., :3] = np.clip(0.82 * facecolors[..., :3] + 0.18 * shaded, 0.0, 1.0)
    facecolors[..., 3] = 0.99
    return facecolors


def evaluate_grid(dist, xlim, ylim, *, resolution=108, n_grid=72, chunk=3000):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    dens = np.empty(pts.shape[0], dtype=float)
    for start in range(0, pts.shape[0], chunk):
        stop = min(pts.shape[0], start + chunk)
        dens[start:stop] = dist.bivariate_pdf(pts[start:stop], dims=(0, 1), n_grid=n_grid)
    return xx, yy, dens.reshape(xx.shape)


def draw_surface(
    ax,
    dist,
    title,
    xlim,
    ylim,
    *,
    resolution=124,
    n_grid=80,
    display_mode="relative",
    height_gamma=0.36,
    height_scale=0.74,
    color_gamma=0.46,
    elev=46,
    azim=-52,
    z_aspect=0.36,
    zoom=1.27,
):
    xx, yy, dens = evaluate_grid(dist, xlim, ylim, resolution=resolution, n_grid=n_grid)
    rel = np.clip(dens / max(float(np.nanmax(dens)), 1e-300), 0.0, 1.0)
    if display_mode == "relative":
        display_rel = rel
    elif display_mode == "ridge_boost":
        display_rel = np.power(rel, 0.32)
    else:
        raise ValueError(f"Unknown display_mode {display_mode!r}.")
    height = height_scale * np.power(display_rel, height_gamma)
    norm = colors.PowerNorm(gamma=color_gamma, vmin=0.0, vmax=1.0)
    facecolors = DENSITY_CMAP(norm(display_rel))
    facecolors = softly_light_surface(facecolors, height)

    ax.plot_surface(
        xx,
        yy,
        height,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
        rasterized=True,
    )
    ax.contourf(
        xx,
        yy,
        display_rel,
        zdir="z",
        offset=0.0,
        levels=np.linspace(0.0, 1.0, 32),
        cmap=DENSITY_CMAP,
        alpha=0.74,
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        display_rel,
        zdir="z",
        offset=0.005,
        levels=np.linspace(0.10, 0.92, 7),
        colors=BASE_CONTOUR_COLOR,
        linewidths=0.34,
        alpha=0.27,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(0, 0.88)
    surface_panel_title(ax, title, pad=0, size=9.5)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1.18, 1.0, z_aspect), zoom=zoom)
    except TypeError:
        ax.set_box_aspect((1.18, 1.0, z_aspect))
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANE_COLOR)
        axis.pane.set_edgecolor("#edf4f6")
        axis._axinfo["axisline"]["color"] = (0.82, 0.88, 0.93, 0.72)
        axis._axinfo["grid"]["color"] = (0.86, 0.91, 0.95, 0.44)
        axis._axinfo["grid"]["linewidth"] = 0.35


def main() -> None:
    set_sfd_matplotlib_style(plt, base_size=8.7, title_size=9.5, label_size=8.7, tick_size=7.7)
    specs = [
        (
            snowflake(d=3, sigma_shape=0.085, random_state=1),
            "a  Snowflake",
            (-1.85, 1.85),
            (-1.85, 1.85),
            {
                "height_gamma": 0.28,
                "height_scale": 0.84,
                "color_gamma": 0.48,
                "elev": 36,
                "z_aspect": 0.52,
                "zoom": 1.23,
            },
        ),
        (
            cartoon_face(d=3, sigma_curve=0.070, sigma_blob=0.105, random_state=1),
            "b  Stylized face",
            (-2.05, 2.05),
            (-2.05, 1.95),
            {
                "display_mode": "ridge_boost",
                "height_gamma": 0.52,
                "height_scale": 0.82,
                "color_gamma": 0.58,
            },
        ),
        (
            maze_graph(d=3, sigma_edge=0.085, sigma_atom=0.135, random_state=1),
            "c  Maze graph",
            (-1.85, 1.85),
            (-1.45, 1.45),
            {},
        ),
        (
            ecg_like(sigma_x=0.018, sigma_y0=0.095, sigma_y1=0.130, random_state=1),
            "d  ECG-like flow",
            (-0.02, 1.02),
            (-0.80, 1.85),
            {},
        ),
    ]

    fig = plt.figure(figsize=(9.15, 6.55))
    positions = [
        [0.010, 0.545, 0.485, 0.405],
        [0.505, 0.545, 0.485, 0.405],
        [0.010, 0.135, 0.485, 0.405],
        [0.505, 0.135, 0.485, 0.405],
    ]
    for pos, (dist, title, xlim, ylim, display_kwargs) in zip(positions, specs):
        ax = fig.add_axes(pos, projection="3d")
        draw_surface(ax, dist, title, xlim, ylim, **display_kwargs)

    sm = ScalarMappable(norm=colors.PowerNorm(gamma=0.46, vmin=0.0, vmax=1.0), cmap=DENSITY_CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.325, 0.048, 0.35, 0.016])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.set_title(
        "relative bivariate density",
        fontsize=8.5,
        fontfamily=SURFACE_FONT_FAMILY,
        color=TITLE_COLOR,
        pad=6,
    )
    cbar.ax.tick_params(labelsize=7.5, colors=TITLE_COLOR, width=0.55, length=2)
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontfamily(SURFACE_FONT_FAMILY)
    cbar.outline.set_linewidth(0.55)
    cbar.outline.set_edgecolor("#dae5eb")

    png = FIG_DIR / "fig_3d_density_panel.png"
    pdf = FIG_DIR / "fig_3d_density_panel.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
