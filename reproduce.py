#!/usr/bin/env python3
"""Explicit, opt-in entry points for the released SFD evidence."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ.setdefault("MPLBACKEND", "Agg")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def run(script, arguments):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
                   cwd=ROOT, check=True)


def fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify():
    from paper.extract_uncertainty import summaries, recount
    evidence = ROOT / "paper/evidence"
    for stem, geometry in (("geometry", True), ("fixed", False)):
        indexed, fields = summaries(evidence / f"{stem}_interval_summary.csv", geometry)
        count = recount(evidence / f"{stem}_intervals.csv", indexed, fields)
        print(f"{stem}: checked {count} interval records and {len(indexed)} summary cells")
    from scripts.run_known_skeleton_sfd_estimator_pilot import summarize_density, summarize_recovery
    for family, expected in (("known_skeleton", 1125), ("classical", 155), ("copula", 95)):
        directory = evidence / family
        scores = rows(directory / "heldout_log_scores.csv")
        if len(scores) != expected:
            raise ValueError(f"Incomplete {family} score archive")
        if len({(r["scenario"], r["rep"], r["method"]) for r in scores}) != expected:
            raise ValueError(f"Duplicate {family} score records")
        if family == "known_skeleton":
            import math
            calculated = summarize_density(scores)
            saved = rows(directory / "benchmark_summary.csv")
            for new, old in zip(calculated, saved):
                if (new["scenario"], new["method"]) != (old["scenario"], old["method"]):
                    raise ValueError("Summary ordering differs")
                for field in ("mean_heldout_log_score", "se_heldout_log_score"):
                    if not math.isclose(float(new[field]), float(old[field]), rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError("Saved score summary differs: " + field)
            if len(calculated) != len(saved):
                raise ValueError("Missing saved summary cells")
            recovery = summarize_recovery(rows(directory / "recovery_scores.csv"))
            if len(recovery) != 5:
                raise ValueError("Incomplete recovery summaries")
        print(f"{family}: {len(scores)} score records retained, including warning/failure statuses")
    print("Archived evidence checks passed. No fitting or downloading performed.")


def figures(out):
    out = fresh(out)
    from scripts.run_known_skeleton_sfd_estimator_pilot import maybe_plot
    evidence = ROOT / "paper/evidence/known_skeleton"
    outputs = maybe_plot(out, out, "known_skeleton_fitted",
                         rows(evidence / "benchmark_summary.csv"),
                         rows(evidence / "recovery_summary.csv"))
    if not outputs:
        raise RuntimeError("Known-skeleton figure export failed")
    from paper import plot_external_controls
    plot_external_controls.render(ROOT / "paper", out)
    from paper.extract_uncertainty import export
    export(ROOT / "paper/evidence", out / "interval_tables")
    print("Saved result figures and recounted interval tables to", out)


def elvis(out):
    import numpy as np
    from scripts.generate_sfd_elvis_example import build_law, density_grid, draw
    out = fresh(out)
    grouped = {}
    for row in rows(ROOT / "data/derived/elvis_contours.csv"):
        item = grouped.setdefault(row["component"], {"name": row["component"], "path": []})
        item["path"].append([float(row["X1"]), float(row["X2"])])
    selected = [{**item, "path": np.asarray(item["path"])} for item in grouped.values()]
    law = build_law(selected, .014)
    samples = law.rvs(18000, random_state=2026102205)
    points = np.vstack([x["path"] for x in selected])
    low, high = points.min(0) - .10, points.max(0) + .10
    edges = [np.linspace(low[j], high[j], 501) for j in (0, 1)]
    density = density_grid(law, *edges)
    draw(selected, law, samples, density, *edges, out, out / "elvis_example.pdf")
    integral = float(density.sum() * np.diff(edges[0])[0] * np.diff(edges[1])[0])
    (out / "checks.json").write_text(json.dumps({"samples": len(samples),
        "components": len(selected), "display_integral": integral,
        "source": "released derived contours; reference photo not used"}, indent=2) + "\n")
    print("Rebuilt Elvis figure from derived contours; display integral:", integral)


def fit(study, out, raw=None, smoke=False):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new run directory to preserve evidence")
    common = ["--out", out]
    if smoke or study == "known":
        run("run_known_skeleton_sfd_estimator_pilot.py", common + ([
            "--scenarios", "ecg", "--reps", "1", "--n-train", "80", "--n-test", "80",
            "--gmm-components", "2", "--no-kde", "--n-grid", "16",
            "--em-max-iter", "40", "--gmm-max-iter", "30"] if smoke else [
            "--scenarios", "snowflake,doraemon,maze,ecg,torus", "--reps", "25",
            "--n-train", "1000", "--n-test", "1000", "--seed", "20260623",
            "--gmm-components", "2,4,8,16,32", "--n-grid", "64",
            "--figure-dir", out / "figures", "--figure-prefix", "known_skeleton_fitted"]))
    elif study == "mechanism":
        run("run_sfd_mechanism_experiment.py", common + ["--reps", "25", "--n-train", "1000", "--n-test", "2000", "--seed", "20261018"])
    elif study == "geometry":
        run("run_sfd_geometry_stress.py", common + ["--reps", "25", "--n-train", "1000", "--n-test", "2000"])
    elif study == "pbmc":
        if raw is None or not Path(raw).is_file():
            raise ValueError("PBMC requires --raw pointing to the downloaded 10x tarball")
        run("run_pbmc3k_inductive_validation.py", common + ["--raw", Path(raw).resolve(), "--reps", "25", "--n-features", "1000"])
    elif study in {"classical", "copula"}:
        script = "run_large_scale_" + ("classical_distribution" if study == "classical" else "copula") + "_pilot.py"
        extra = ["--report", out / "execution_report.md"] if study == "classical" else []
        run(script, common + extra + ["--reps", "5", "--n-train", "1000", "--n-test", "1000", "--seed", "20260624"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    commands.add_parser("paper")
    for name in ("figures", "elvis", "smoke"):
        commands.add_parser(name).add_argument("--out", type=Path, required=True)
    long = commands.add_parser("fit", help="Explicit full-size fitting; may take hours")
    long.add_argument("study", choices=["known", "mechanism", "geometry", "pbmc", "classical", "copula"])
    long.add_argument("--out", type=Path, required=True)
    long.add_argument("--raw", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "figures":
        figures(args.out)
    elif args.command == "elvis":
        elvis(args.out)
    elif args.command == "smoke":
        fit("known", args.out, smoke=True)
    elif args.command == "fit":
        fit(args.study, args.out, args.raw)
    else:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "-outdir=build", "main.tex"], cwd=ROOT / "paper", check=True)


if __name__ == "__main__":
    main()
