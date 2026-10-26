#!/usr/bin/env python3
"""Geometry and noise-shape sensitivity under a frozen two-arm protocol."""

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import norm, t

from scripts.run_sfd_mechanism_experiment import (
    ANGLES, BOUNDS, GROUPS, LOG_2PI, PARALLEL_SD, RESIDUAL_MEAN, RESIDUAL_SD,
    TAU, ArmLaw, coordinate_mean, longitudinal, truth,
)
from scripts.sfd_validation_common import (
    GMM_OPTIONS, begin_run, finish_run, fit_reference, logpdf, summary, write_csv,
)

CONDITIONS = ("nominal", "overlap", "heavy_noise")
METHODS = ("SFD supplied", "SFD rotated", "SFD learned angles",
           "Gaussian", "GMM-4", "GMM-8", "KDE")
ANGLE_BOUNDS = [(np.deg2rad(5), np.deg2rad(75)),
                (np.deg2rad(-75), np.deg2rad(-5))]
OPTIONS = dict(maxiter=500, ftol=1e-10, gtol=1e-6, maxls=40)


@dataclass
class GeometryLaw(ArmLaw):
    angles: np.ndarray
    heavy: bool = False

    def positive_y(self):
        raise NotImplementedError("Geometry stress uses density and recovery scores, not the fixed-angle event formula")

    def components(self, x):
        tangents = np.column_stack((np.cos(self.angles), np.sin(self.angles)))
        normals = np.column_stack((-np.sin(self.angles), np.cos(self.angles)))
        u, v = x @ tangents.T, x @ normals.T
        along, score_location = longitudinal(u, self.locations)
        transverse = (t.logpdf(v, 5, scale=self.widths * np.sqrt(3 / 5))
                      if self.heavy else norm.logpdf(v, 0, self.widths))
        residual = -np.sum((x - RESIDUAL_MEAN)**2, axis=1) / (2 * RESIDUAL_SD**2)
        residual -= LOG_2PI + 2 * np.log(RESIDUAL_SD)
        return np.column_stack((along + transverse, residual)), score_location, (u, v)

    def sample(self, n, seed):
        x, labels = super().sample(n, seed)
        rng = np.random.default_rng(seed + 700000000)
        for branch in range(2):
            mask = labels == branch
            tangent = np.array([np.cos(ANGLES[branch]), np.sin(ANGLES[branch])])
            normal = np.array([-np.sin(ANGLES[branch]), np.cos(ANGLES[branch])])
            u, v = x[mask] @ tangent, x[mask] @ normal
            if self.heavy:
                v = rng.standard_t(5, size=mask.sum()) * self.widths[branch] * np.sqrt(3 / 5)
            a = self.angles[branch]
            x[mask] = u[:, None] * np.array([np.cos(a), np.sin(a)])
            x[mask] += v[:, None] * np.array([-np.sin(a), np.cos(a)])
        return x, labels


def generating_law(condition, group):
    base = truth(group)
    angles = ANGLES / 2 if condition == "overlap" else ANGLES.copy()
    return GeometryLaw(base.weights, base.locations, base.widths, angles,
                       heavy=condition == "heavy_noise")


def unpack(p, angles=None):
    return GeometryLaw(softmax(np.r_[p[:2], 0]), p[2:4], np.exp(p[4:6]),
                       p[6:8] if angles is None else angles)


def objective(p, x, angles=None):
    law = unpack(p, angles)
    components, score_location, (u, v) = law.components(x)
    cols = components + np.log(law.weights)
    total = logsumexp(cols, axis=1)
    resp = np.exp(cols - total[:, None])
    gradient = np.r_[np.mean(resp[:, :2], axis=0) - law.weights[:2],
                     np.mean(resp[:, :2] * score_location, axis=0),
                     np.mean(resp[:, :2] * (v**2 / law.widths**2 - 1), axis=0)]
    if angles is None:
        posterior_s = score_location * TAU**2 + coordinate_mean(law.locations)
        score_u = (posterior_s - u) / PARALLEL_SD**2
        gradient = np.r_[gradient, np.mean(
            resp[:, :2] * (score_u * v + u * v / law.widths**2), axis=0)]
    return -float(np.mean(total)), -gradient


def fit_arm_model(train, angles=None):
    bounds = BOUNDS + ANGLE_BOUNDS if angles is None else BOUNDS
    results = []
    specs = [(20, [0.5, 0.5], [0.08, 0.08]),
             (35, [0.35, 0.65], [0.12, 0.12]),
             (50, [0.65, 0.35], [0.12, 0.12])]
    for index, (degrees, locations, widths) in enumerate(specs):
        p = np.r_[np.log([2, 2]), locations, np.log(widths)]
        if angles is None:
            p = np.r_[p, np.deg2rad([degrees, -degrees])]
        start = time.perf_counter()
        fit = minimize(objective, p, args=(train, angles), jac=True,
                       method="L-BFGS-B", bounds=bounds, options=OPTIONS)
        results.append((fit, {"start": index, "selected": False,
            "converged": bool(fit.success), "n_iter": int(fit.nit),
            "train_mean_log_score": -float(fit.fun), "message": str(fit.message),
            "fit_seconds": time.perf_counter() - start,
            "boundary_contact": any(abs(v - lo) < 1e-5 or abs(v - hi) < 1e-5
                                     for v, (lo, hi) in zip(fit.x, bounds))}))
    finite = [i for i, (fit, _) in enumerate(results) if np.isfinite(fit.fun)]
    if not finite:
        raise FloatingPointError("No finite SFD start")
    selected = min(finite, key=lambda i: results[i][0].fun)
    results[selected][1]["selected"] = True
    return unpack(results[selected][0].x, angles), [row for _, row in results]


def run(args):
    config = dict(vars(args), runner="scripts/run_sfd_geometry_stress.py",
                  protocol="sfd_geometry_stress_v1", gmm_options=GMM_OPTIONS,
                  optimizer=OPTIONS, conditions=CONDITIONS, methods=METHODS,
                  evidence_stage="full_protocol" if (args.reps, args.n_train, args.n_test)
                  == (25, 1000, 2000) else "smoke_or_custom")
    begin_run(args.out, config)
    scores, recovery, diagnostics, hashes, parameters = [], [], [], [], []
    for rep in range(args.reps):
        for ci, condition in enumerate(CONDITIONS):
            for gi, group in enumerate(GROUPS):
                seed = 20261019 + 1000000 * rep + 10000 * ci + 1000 * gi
                law = generating_law(condition, group)
                train, _ = law.sample(args.n_train, seed + 1)
                test, labels = law.sample(args.n_test, seed + 2)
                identity = dict(condition=condition, scenario=group, rep=rep)
                hashes.append(dict(identity, seed=seed,
                    train_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                    test_sha256=hashlib.sha256(test.tobytes()).hexdigest()))
                for mi, method in enumerate(METHODS):
                    start = time.perf_counter()
                    if method.startswith("SFD"):
                        angles = None if method == "SFD learned angles" else law.angles.copy()
                        if method == "SFD rotated":
                            angles += np.deg2rad(5)
                        fit, starts = fit_arm_model(train, angles)
                    else:
                        fit, starts = fit_reference(method, train, seed + 100 + 10 * mi)
                    elapsed = time.perf_counter() - start
                    selected = next((r for r in starts if r["selected"]), {})
                    for row in starts:
                        diagnostics.append(dict(identity, method=method,
                            detail_type="SFD" if method.startswith("SFD") else "reference",
                            **row))
                    evaluation_start = time.perf_counter()
                    score = float(np.mean(logpdf(fit, test)))
                    evaluation_seconds = time.perf_counter() - evaluation_start
                    if not np.isfinite(score):
                        raise FloatingPointError("Nonfinite test score")
                    scores.append(dict(identity, method=method, seed=seed,
                        protocol_version=config["protocol"], scenario_family="geometry_noise_stress",
                        n_train=args.n_train, n_test=args.n_test, heldout_log_score=score,
                        fit_seconds=elapsed, eval_seconds=evaluation_seconds,
                        status="ok" if selected.get("converged", True) else "warning",
                        error_message=selected.get("message", selected.get("warnings", "")),
                        converged=selected.get("converged", True)))
                    if method.startswith("SFD"):
                        resp = fit.responsibilities(test)
                        for component in range(3):
                            parameters.append(dict(identity, method=method, component=component,
                                weight=float(fit.weights[component]),
                                location=float(fit.locations[component]) if component < 2 else "",
                                transverse_sd=float(fit.widths[component]) if component < 2 else "",
                                angle_degrees=float(np.rad2deg(fit.angles[component])) if component < 2 else ""))
                        recovery.append(dict(identity, method=method,
                            upper_mass=float(fit.weights[0]), true_upper_mass=float(law.weights[0]),
                            upper_position=float(coordinate_mean(fit.locations[0])),
                            true_upper_position=float(coordinate_mean(law.locations[0])),
                            upper_width=float(fit.widths[0]), true_upper_width=float(law.widths[0]),
                            angle_rmse_degrees=float(np.sqrt(np.mean((fit.angles - law.angles)**2)) * 180 / np.pi),
                            brier_score=float(np.mean(np.sum((resp - np.eye(3)[labels])**2, axis=1))),
                            stratum_accuracy=float(np.mean(resp.argmax(axis=1) == labels))))
            print(f"rep={rep} condition={condition} complete", flush=True)
    indexed = {(r["condition"], r["scenario"], r["method"], r["rep"]): r for r in recovery}
    contrasts = []
    for row in recovery:
        if row["scenario"] == "reference":
            continue
        base = indexed[(row["condition"], "reference", row["method"], row["rep"])]
        for metric in ("upper_mass", "upper_position", "upper_width"):
            actual = row["true_" + metric] - base["true_" + metric]
            delta = row[metric] - base[metric]
            contrasts.append({k: row[k] for k in ("condition", "scenario", "method", "rep")}
                             | dict(metric=metric, truth=actual, estimate=delta,
                                    error=delta - actual, squared_error=(delta - actual)**2))
    for name, rows in (("scores", scores), ("recovery", recovery),
                       ("sample_hashes", hashes), ("contrasts", contrasts),
                       ("fitted_parameters", parameters)):
        write_csv(args.out / (name + ".csv"), rows)
    fields = sorted(set().union(*(r.keys() for r in diagnostics)))
    write_csv(args.out / "optimizer_starts.csv", diagnostics, fieldnames=fields)
    write_csv(args.out / "density_summary.csv", summary(scores,
        ["condition", "scenario", "method"], ["heldout_log_score"]))
    write_csv(args.out / "contrast_summary.csv", summary(contrasts,
        ["condition", "scenario", "method", "metric"], ["estimate", "error", "squared_error"]))
    finish_run(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=2000)
    args = parser.parse_args()
    if args.reps < 1 or args.n_train < 30 or args.n_test < 10:
        parser.error("Need positive reps, n_train>=30 and n_test>=10")
    run(args)
