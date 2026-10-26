#!/usr/bin/env python3
"""Show a stylized-face SFD using its components and frozen 3D observations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np

from sfd_figure_style import SFD_FONT_FAMILY, set_sfd_matplotlib_style

TUBE = "#236EA0"
BLOB = "#C44E32"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def render(inputs: Path, out: Path) -> None:
    components = read_rows(inputs / "components.csv")
    observations = read_rows(inputs / "observations.csv")
    curves = read_rows(inputs / "curve_grid.csv")
    recipe = json.loads((inputs / "run_parameters.json").read_text())
    assert len(components) == 21 and len(observations) == 1600
    assert sum(row["kind"] == "tube" for row in components) == 16
    assert sum(row["kind"] == "gaussian" for row in components) == 5
    assert abs(sum(float(row["weight"]) for row in components) - 1) < 1e-12
    xyz = np.array([[float(row[f"Y{j}"]) for j in (1, 2, 3)] for row in observations])
    is_blob = np.array([row["kind"] == "gaussian" for row in observations])
    assert np.isfinite(xyz).all()
    out.mkdir(parents=True, exist_ok=False)
    set_sfd_matplotlib_style(plt, base_size=8.5, title_size=9.4,
                             label_size=9, tick_size=8)
    fig = plt.figure(figsize=(6, 3.6))
    left = fig.add_axes([0.075, 0.20, 0.39, 0.70])
    right = fig.add_axes([0.49, 0.18, 0.43, 0.72], projection="3d")
    for component in components:
        name = component["component"]
        if component["kind"] == "tube":
            grid = [row for row in curves if row["component"] == name]
            points = np.array([[float(row[key]) for key in ("x", "y", "z")] for row in grid])
            left.plot(points[:, 0], points[:, 1], color=TUBE, lw=0.95)
            right.plot(points[:, 0], points[:, 2], points[:, 1],
                       color="#84929D", lw=0.5, alpha=0.55, zorder=1)
        else:
            mean = [float(component[key]) for key in ("mean_x", "mean_y")]
            assert float(component["cov_xy"]) == 0
            ellipse = Ellipse(mean, width=4*np.sqrt(float(component["cov_xx"])),
                              height=4*np.sqrt(float(component["cov_yy"])),
                              facecolor=matplotlib.colors.to_rgba(BLOB, 0.18),
                              edgecolor=BLOB, lw=0.65, zorder=3)
            left.add_patch(ellipse)
            left.plot(*mean, marker="o", ms=1.1, color=BLOB, zorder=4)
    left.set(xlim=(-1.96, 1.96), ylim=(-1.96, 1.90),
             xticks=[-1, 0, 1], yticks=[-1, 0, 1],
             xlabel=r"$X_1$", ylabel=r"$X_2$")
    left.set_aspect("equal")
    left.spines[["top", "right"]].set_visible(False)
    for spine in left.spines.values():
        spine.set_color("#B1B1B1")
        spine.set_linewidth(0.5)
    left.tick_params(length=2.5, width=0.5, color="#999999", pad=3)
    for mask, color, size in [(~is_blob, TUBE, 2.7), (is_blob, BLOB, 3.2)]:
        right.scatter(xyz[mask, 0], xyz[mask, 2], xyz[mask, 1],
                      color=color, s=size, alpha=0.72, depthshade=False,
                      edgecolors="none", zorder=2)
    right.set(xlim=(-1.96, 1.96), ylim=(-0.15, 0.15), zlim=(-1.96, 1.90),
              xticks=[-1, 0, 1], yticks=[-0.1, 0.1], zticks=[-1, 0, 1],
              xlabel=r"$X_1$", ylabel=r"$X_3$", zlabel=r"$X_2$")
    right.set_box_aspect(recipe["sample_panel_box_aspect"])
    camera = recipe["sample_panel_camera"]
    right.view_init(elev=camera["elevation"], azim=camera["azimuth"])
    right.set_proj_type("ortho")
    right.grid(False)
    for axis in [right.xaxis, right.yaxis, right.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color("#B1B1B1")
        axis.line.set_linewidth(0.5)
        axis.labelpad = 1.5
    right.tick_params(pad=0, labelsize=7)
    for x, title in [(0.075, "(a) Declared components"),
                     (0.50, "(b) Three-dimensional sample")]:
        fig.text(x, 0.94, title, fontsize=9.4, fontweight="bold",
                 fontfamily=SFD_FONT_FAMILY, va="bottom")
    handles = [Line2D([], [], color=TUBE, lw=1, label="Curve / Gaussian tube"),
               Line2D([], [], marker="o", linestyle="none", color=BLOB,
                      markersize=3, label="Localized Gaussian blob")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=2, frameon=False, fontsize=8, handlelength=2,
               columnspacing=2.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [left.xaxis.label, left.yaxis.label, right.xaxis.label,
              right.yaxis.label, right.zaxis.label, *fig.texts]
    for label in labels:
        bounds = label.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    fig.savefig(out / "face_example.pdf", metadata={"Title": "Stylized-face SFD: components and 3D sample"})
    fig.savefig(out / "face_example.png", dpi=300)
    plt.close(fig)
    with (out / "component_counts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["component", "weight", "observed_count"])
        for component in components:
            writer.writerow([component["component"], component["weight"],
                             sum(row["component"] == component["component"] for row in observations)])
    caption = (
        "A stylized-face SFD: (a) the first-two-coordinate projection of 16 curve centerlines "
        "and five localized Gaussian components; (b) all 1600 frozen observations in three dimensions. "
        "Blue and red indicate generating tube and blob components. Blob contours in (a) have squared "
        "Mahalanobis distance four. All tubes have Uniform(0,1) branch coordinates, with isotropic "
        "SD 0.020 in the first two coordinates; localized Gaussian SD is 0.030. Third-coordinate "
        "kernel SD is 0.012. Panel (b) uses the actual X3 coordinate with unequal axis display scales, "
        "not density height. Curves are centerlines, not support boundaries. This is a generating-law "
        "illustration without exact atoms, fitting or new sampling.\n"
    )
    (out / "caption.txt").write_text(caption)
    (out / "runtime.json").write_text(json.dumps({
        "python": sys.version.split()[0], "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "backend": matplotlib.get_backend(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.inputs, args.out)
