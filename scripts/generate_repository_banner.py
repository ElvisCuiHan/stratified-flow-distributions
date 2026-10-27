#!/usr/bin/env python3
"""Render the repository header from released contours and generated samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = "#FAFCFC"
INK = "#243C43"
TEAL = "#27898B"
BLUE = "#568FAC"
CORAL = "#D77979"
MUTED = "#74868A"


def read_rows(relative):
    with (ROOT / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def grouped_points(rows, columns):
    paths = {}
    for row in rows:
        paths.setdefault(row["component"], []).append([float(row[k]) for k in columns])
    return {name: np.asarray(points) for name, points in paths.items()}


def render(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "svg.fonttype": "path", "svg.hashsalt": "sfd-banner-v1",
    })
    elvis = grouped_points(read_rows("data/derived/elvis_contours.csv"), ("X1", "X2"))
    face = grouped_points(read_rows("data/derived/face_3d/curve_grid.csv"), ("x", "y"))
    observations = read_rows("data/derived/face_3d/observations.csv")
    sample = np.asarray([[float(r["Y1"]), float(r["Y2"])] for r in observations])
    blobs = np.array([r["kind"] == "gaussian" for r in observations])

    fig = plt.figure(figsize=(12.8, 4.1), dpi=150, facecolor=BACKGROUND)
    fig.text(.045, .88, "SFD  /  RESEARCH SOFTWARE", fontsize=8, color=TEAL)
    title = fig.text(.042, .715, "Stratified Flow\nDistributions", va="top",
                    family="DejaVu Serif", weight="bold", fontsize=34, linespacing=1.12)
    fig.text(.045, .315, "Branch mass. Occupancy. Noise.", fontsize=12.5, color=INK)
    fig.text(.045, .135, "Probability laws on declared structure", fontsize=9, color=MUTED)

    pose_ax = fig.add_axes([.505, .225, .235, .65], facecolor=BACKGROUND)
    for name, points in elvis.items():
        color = CORAL if name in {"microphone", "cable"} else TEAL
        width = .82 if name in {"pose_outline", "cable", "microphone"} else .48
        pose_ax.plot(*points.T, color=color, lw=width, alpha=.9, solid_capstyle="round")
    pose_ax.set_aspect("equal")
    pose_ax.margins(.035)
    pose_ax.axis("off")

    face_ax = fig.add_axes([.765, .245, .205, .63], facecolor=BACKGROUND)
    for points in face.values():
        face_ax.plot(*points.T, color=BLUE, lw=.55, alpha=.22)
    face_ax.scatter(*sample[~blobs].T, s=1.5, color=BLUE, alpha=.78, linewidths=0)
    face_ax.scatter(*sample[blobs].T, s=2.5, color=CORAL, alpha=.75, linewidths=0)
    face_ax.set_aspect("equal")
    face_ax.margins(.04)
    face_ax.axis("off")
    fig.text(.622, .135, "CONTOURS / ELVIS", ha="center", fontsize=7, color=MUTED)
    fig.text(.865, .135, "SAMPLES / DORAEMON", ha="center", fontsize=7, color=MUTED)
    for left, right, color in ((.045, .51, TEAL), (.51, .765, CORAL), (.765, .955, BLUE)):
        fig.add_artist(plt.Line2D([left, right], [.06, .06], transform=fig.transFigure,
                                 color=color, lw=1.5, solid_capstyle="butt"))

    # Keep all text inside the canvas and the headline clear of the contour panel.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.texts:
        box = text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        if not (0 <= box.x0 < box.x1 <= 1 and 0 <= box.y0 < box.y1 <= 1):
            raise ValueError("Banner text falls outside the canvas: " + text.get_text())
    title_box = title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    if title_box.x1 >= .49:
        raise ValueError("Headline overlaps the illustration area")

    fig.savefig(out / "banner.png", dpi=150, facecolor=BACKGROUND)
    fig.savefig(out / "banner.svg", facecolor=BACKGROUND,
                metadata={"Date": None, "Title": "Stratified Flow Distributions",
                          "Description": "Released Elvis-derived contours and the first two coordinates of the archived Doraemon-inspired sample; illustrative, not fitted reconstruction."})
    plt.close(fig)
    contract = {
        "purpose": "repository header, not an experimental result or manuscript figure",
        "canvas_pixels": [1920, 615],
        "elvis": {"source": "data/derived/elvis_contours.csv", "components": len(elvis)},
        "doraemon": {"curves": "data/derived/face_3d/curve_grid.csv",
                     "sample": "data/derived/face_3d/observations.csv",
                     "displayed_coordinates": ["Y1", "Y2"], "observations": len(sample)},
        "new_fits": 0,
        "rights": "THIRD_PARTY_NOTICES.md; photo-derived and character elements excluded from MIT and CC BY grants",
    }
    (out / "banner_sources.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets")
    render(parser.parse_args().out)
