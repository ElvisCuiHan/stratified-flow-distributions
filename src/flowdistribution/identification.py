"""Target-specific empirical information for fixed normalized mixture kernels.

This diagnoses a supplied evaluation design, not population identification.
Geometry, kernel scales and reference measure are held fixed; no prior is used.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class TargetInformation:
    rank: int
    dimension: int
    singular_values: np.ndarray
    null_fraction: np.ndarray
    in_numerical_row_space: np.ndarray
    conditional_variance_factor: np.ndarray


def target_information(log_kernels, weights, targets, *, evaluation_weights=None,
                       rank_rtol=1e-8, target_tol=1e-7):
    """Check linear weight targets with the last mixture weight eliminated.

    Rows of ``targets`` give coefficients a in a'weights. Evaluation weights
    default to 1/n; probability-weighted exact finite designs are also accepted.
    The reported variance factor is a'I^+a in the reduced coordinates, without
    dividing by a sample size. It is undefined (NaN) for a detected null-space
    component. It ignores estimated kernels, selection and regularization bias.
    """
    logs = np.asarray(log_kernels, dtype=float)
    w = np.asarray(weights, dtype=float)
    a = np.atleast_2d(np.asarray(targets, dtype=float))
    if (logs.ndim != 2 or not len(logs) or logs.shape[1] < 2
            or w.shape != (logs.shape[1],) or a.ndim != 2 or a.shape[1] != logs.shape[1]):
        raise ValueError("Expected n-by-K kernels, K weights and target-by-K coefficients.")
    if (np.isnan(logs).any() or np.isposinf(logs).any()
            or not np.isfinite(logs.max(axis=1)).all()
            or not np.isfinite(w).all() or np.any(w <= 0)
            or not np.isclose(w.sum(), 1.) or not np.isfinite(a).all()):
        raise ValueError("Finite targets and positive normalized mixture weights are required.")
    if not 0 < rank_rtol < 1 or not 0 < target_tol < 1:
        raise ValueError("Diagnostic tolerances must lie in (0,1).")
    if evaluation_weights is None:
        measure = np.full(len(logs), 1. / len(logs))
    else:
        measure = np.asarray(evaluation_weights, dtype=float)
        if (measure.shape != (len(logs),) or not np.isfinite(measure).all()
                or np.any(measure < 0) or not np.isclose(measure.sum(), 1.)):
            raise ValueError("Evaluation weights must be nonnegative and sum to one.")
        measure = measure / measure.sum()
    w = w / w.sum()
    total = logsumexp(logs + np.log(w), axis=1)
    ratios = np.exp(logs - total[:, None])
    scores = ratios[:, :-1] - ratios[:, -1:]
    weighted = scores * np.sqrt(measure[:, None])
    if not np.isfinite(weighted).all():
        raise ValueError("Numerical overflow in the fixed-kernel information diagnostic.")
    # full_matrices covers null directions even for n < K-1.
    _, singular, right = np.linalg.svd(weighted, full_matrices=len(weighted) < weighted.shape[1])
    cutoff = rank_rtol * singular[0] if singular.size and singular[0] > 0 else 0.
    rank = int(np.count_nonzero(singular > cutoff))
    gradient = a[:, :-1] - a[:, -1:]
    coordinates = gradient @ right.T
    norms = np.linalg.norm(gradient, axis=1)
    null = np.linalg.norm(coordinates[:, rank:], axis=1)
    fractions = np.divide(null, norms, out=np.zeros_like(null), where=norms > 0)
    supported = fractions <= target_tol
    variance = np.full(len(a), np.nan)
    variance[supported] = np.sum((coordinates[supported, :rank] / singular[:rank])**2, axis=1)
    return TargetInformation(rank, logs.shape[1] - 1, singular, fractions, supported, variance)
