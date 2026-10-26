#!/usr/bin/env python3
"""Export the existing stylized-face law and frozen example observations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from flowdistribution import cartoon_face


def export(samples: Path, out: Path) -> None:
    with samples.open(newline="") as stream:
        observations = list(csv.DictReader(stream))
    law = cartoon_face(d=3, random_state=20260623)
    by_name = {component.name: component for component in law.components}
    if len(observations) != 1600:
        raise ValueError("Expected all 1600 frozen example observations.")
    for index, row in enumerate(observations, 1):
        component = by_name[row["component"]]
        if int(row["id"]) != index or row["kind"] != component.kind:
            raise ValueError("Frozen observation labels do not match the law.")
        if not np.isfinite([float(row[f"Y{j}"]) for j in (1, 2, 3)]).all():
            raise ValueError("Nonfinite observed coordinates.")

    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(samples, out / "observations.csv")
    with (out / "components.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["component", "kind", "weight", "mean_x", "mean_y",
                         "mean_z", "cov_xx", "cov_xy", "cov_yy", "cov_zz"])
        for component in law.components:
            loc = component.loc if component.kind == "gaussian" else [np.nan] * 3
            writer.writerow([component.name, component.kind, component.weight,
                             *loc, component.cov[0, 0], component.cov[0, 1],
                             component.cov[1, 1], component.cov[2, 2]])
    with (out / "curve_grid.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["component", "s", "x", "y", "z"])
        s = np.linspace(0, 1, 401)
        for component in law.components:
            if component.kind == "tube":
                for coordinate, point in zip(s, component.curve(s)):
                    writer.writerow([component.name, coordinate, *point])
    parameters = {
        "law": "cartoon_face", "dimension": 3,
        "component_panel_dimensions": [0, 1],
        "sample_panel_axis_order": [0, 2, 1],
        "sample_panel_box_aspect": [1, 0.42, 1],
        "sample_panel_camera": {"elevation": 12, "azimuth": -68},
        "sigma_curve": 0.020, "sigma_blob": 0.030, "z_sigma": 0.012,
        "coordinate_law": "Uniform(0,1)", "frozen_sample_size": 1600,
        "original_sample_seed": 20260623, "curve_grid_size": 401,
        "blob_contour_mahalanobis_squared": 4,
        "new_samples": 0, "new_fits": 0,
        "sample_source": "outputs/python_package_demo/cartoon_face.csv",
        "original_generator": "examples/simulate_sfd.py",
    }
    (out / "run_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.samples, args.out)
