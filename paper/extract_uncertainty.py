"""Reproduce interval tables/counts from checked summary CSVs, not model fits."""
import argparse
import csv
import math
from statistics import NormalDist, fmean, stdev
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ("nominal", "overlap", "heavy_noise")
METHODS = ("SFD supplied", "SFD rotated", "SFD learned angles")
COVARIANCES = ("information", "sandwich")
TARGETS = ("mass", "mean_position", "width")
GROUPS = ("reference", "mass", "position", "width")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summaries(path, geometry):
    rows = read_csv(path)
    prefix = ("condition", "method", "covariance") if geometry else ()
    fields = prefix + ("kind", "scenario", "target")
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        if key in indexed:
            raise ValueError("Duplicate summary cell")
        indexed[key] = row
        n, valid, failures, covered = map(int, (row[k] for k in ("total", "valid", "failures", "covered")))
        if n != 25 or valid + failures != n or not 0 <= covered <= valid <= n:
            raise ValueError("Invalid summary counts")
        for k in ("truth", "coverage_all", "coverage_mc_lower", "coverage_mc_upper",
                  "bias", "empirical_sd", "mean_standard_error"):
            if not math.isfinite(float(row[k])):
                raise ValueError("Missing/nonfinite checked summary")
        if abs(float(row["coverage_all"]) - covered/n) > 1e-12:
            raise ValueError("Coverage/count mismatch")
        if not 0 <= float(row["coverage_mc_lower"]) <= covered/n <= float(row["coverage_mc_upper"]) <= 1:
            raise ValueError("Invalid coverage Monte Carlo interval")
    base = [(kind, group, target) for kind in ("group", "contrast")
            for group in (GROUPS if kind == "group" else GROUPS[1:]) for target in TARGETS]
    expected = {(c, m, v) + k for c in CONDITIONS for m in METHODS for v in COVARIANCES
                for k in base} if geometry else set(base)
    if set(indexed) != expected:
        raise ValueError("Incomplete summary design")
    return indexed, fields


def recount(path, indexed, fields):
    groups = {}
    for row in read_csv(path):
        key = tuple(row[k] for k in fields)
        if key not in indexed:
            raise ValueError("Unexpected interval cell")
        bucket = groups.setdefault(key, {})
        rep = int(row["rep"])
        if rep in bucket:
            raise ValueError("Duplicate interval repetition")
        valid = row["valid"] == "True"
        if row["valid"] not in ("True", "False"):
            raise ValueError("Malformed interval validity")
        truth = float(row["truth"])
        if not math.isfinite(truth) or abs(truth - float(indexed[key]["truth"])) > 1e-12:
            raise ValueError("Interval truth differs from its summary")
        if valid:
            values = [float(row[k]) for k in ("estimate", "standard_error", "lower", "upper")]
            if not all(map(math.isfinite, values)) or values[1] <= 0:
                raise ValueError("Nonfinite interval or nonpositive SE")
            z = NormalDist().inv_cdf(.975)
            if (abs(values[2] - (values[0]-z*values[1])) > 1e-12
                    or abs(values[3] - (values[0]+z*values[1])) > 1e-12):
                raise ValueError("Interval endpoints differ from its estimate/SE")
        covered = valid and float(row["lower"]) <= float(row["truth"]) <= float(row["upper"])
        if row["covered"] != str(covered):
            raise ValueError("Interval coverage flag mismatch")
        bucket[rep] = (valid, covered, row)
    if set(groups) != set(indexed):
        raise ValueError("Interval/summary cells differ")
    for key, repetitions in groups.items():
        if set(repetitions) != set(range(25)):
            raise ValueError("Missing interval repetitions")
        if sum(v for v, _, _ in repetitions.values()) != int(indexed[key]["valid"]):
            raise ValueError("Valid count differs")
        if sum(c for _, c, _ in repetitions.values()) != int(indexed[key]["covered"]):
            raise ValueError("Covered count differs")
        valid_rows = [r for v, _, r in repetitions.values() if v]
        estimates = [float(r["estimate"]) for r in valid_rows]
        statistics = {
            "bias": fmean(estimates) - float(indexed[key]["truth"]),
            "empirical_sd": stdev(estimates),
            "mean_standard_error": fmean(float(r["standard_error"]) for r in valid_rows),
        }
        for name, value in statistics.items():
            if not math.isclose(value, float(indexed[key][name]), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Summary statistic differs from interval records: " + name)
    return sum(len(reps) for reps in groups.values())


def count(row):
    return row["covered"] + "/" + row["total"]


def export(evidence, destination):
    geometry, fields = summaries(evidence/"geometry_interval_summary.csv", True)
    fixed, fixed_fields = summaries(evidence/"fixed_interval_summary.csv", False)
    geometry_rows = recount(evidence/"geometry_intervals.csv", geometry, fields)
    fixed_rows = recount(evidence/"fixed_intervals.csv", fixed, fixed_fields)
    fits = read_csv(evidence/"geometry_fit_diagnostics.csv")
    expected_fits = {(c, g, str(rep), m) for c in CONDITIONS for g in GROUPS
                     for rep in range(25) for m in METHODS}
    fitted_keys = {tuple(r[k] for k in ("condition", "scenario", "rep", "method")) for r in fits}
    if len(fits) != 900 or fitted_keys != expected_fits or any(r["valid"] != "True" for r in fits):
        raise ValueError("Unexpected saved-fit inventory")
    requests = {
        "CovGaussianLearnedMass": ("nominal", "SFD learned angles", "information", "contrast", "mass", "mass"),
        "CovGaussianLearnedPosition": ("nominal", "SFD learned angles", "information", "contrast", "position", "mean_position"),
        "CovGaussianLearnedWidth": ("nominal", "SFD learned angles", "information", "contrast", "width", "width"),
        "CovHeavyRotatedInfo": ("heavy_noise", "SFD rotated", "information", "contrast", "width", "width"),
        "CovHeavyRotatedSand": ("heavy_noise", "SFD rotated", "sandwich", "contrast", "width", "width"),
    }
    macros = [rf"\newcommand{{\{name}}}{{{count(geometry[key])}}}" for name, key in requests.items()]
    macros += [rf"\newcommand{{\GeoFitCount}}{{{len(fits)}}}",
               rf"\newcommand{{\GeoIntervalCount}}{{{sum(int(r['total']) for r in geometry.values())}}}",
               rf"\newcommand{{\GeoSummaryCount}}{{{len(geometry)}}}"]
    labels = {"nominal": "Gaussian", "overlap": "Overlap", "heavy_noise": r"Student-$t_5$",
              "SFD supplied": "Supplied", "SFD rotated": "Rotated", "SFD learned angles": "Learned",
              "mass": "Mass", "mean_position": "Position", "width": "SD"}
    table = [r"\begin{tabular}{@{}lllrrrrr@{}}", r"\toprule",
             r"Condition & Directions & Target & Info. & Sand. & Bias & SE (info.) & SE (sand.) \\", r"\midrule"]
    for condition in CONDITIONS:
        for method in METHODS:
            for group, target in zip(GROUPS[1:], TARGETS):
                key = (condition, method, "information", "contrast", group, target)
                a = geometry[key]
                b = geometry[key[:2] + ("sandwich",) + key[3:]]
                if a["bias"] != b["bias"] or a["empirical_sd"] != b["empirical_sd"]:
                    raise ValueError("Covariances unexpectedly change point estimates")
                cells = [labels[condition], labels[method], labels[target], count(a), count(b),
                         f'{float(a["bias"]):.5f}', f'{float(a["mean_standard_error"]):.5f}',
                         f'{float(b["mean_standard_error"]):.5f}']
                table.append(" & ".join(cells) + r" \\")
            table.append(r"\addlinespace[3pt]")
    table += [r"\bottomrule", r"\end{tabular}"]
    fixed_table = [r"\begin{tabular}{@{}lllrrrr@{}}", r"\toprule",
                   r"Scope & Group & Target & Covered & 95\% MC interval & Empirical SD & Mean SE \\",
                   r"\midrule"]
    for kind in ("group", "contrast"):
        for group in (GROUPS if kind == "group" else GROUPS[1:]):
            for target in TARGETS:
                row = fixed[kind, group, target]
                cells = [kind.capitalize(), group.capitalize(),
                         {"mass": "Mass", "mean_position": "Position", "width": "Width"}[target],
                         count(row),
                         f'[{float(row["coverage_mc_lower"]):.3f}, {float(row["coverage_mc_upper"]):.3f}]',
                         f'{float(row["empirical_sd"]):.5f}',
                         f'{float(row["mean_standard_error"]):.5f}']
                fixed_table.append(" & ".join(cells) + r" \\")
            fixed_table.append(r"\addlinespace[3pt]")
    fixed_table.pop()  # Match the earlier table's spacing before the final rule.
    fixed_table += [r"\bottomrule", r"\end{tabular}"]
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"tables").mkdir()
    (destination/"uncertainty_macros.tex").write_text("\n".join(macros)+"\n")
    (destination/"tables/geometry_intervals.tex").write_text("\n".join(table)+"\n")
    (destination/"tables/fixed_intervals.tex").write_text("\n".join(fixed_table)+"\n")
    return dict(geometry_summary_cells=len(geometry), fixed_summary_cells=len(fixed),
                geometry_interval_rows=geometry_rows, fixed_interval_rows=fixed_rows,
                displayed_geometry_rows=27, displayed_fixed_rows=len(fixed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=HERE/"evidence")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.evidence, args.out))
