"""Penalized weight learning for fixed, normalized SFD component kernels."""

from dataclasses import dataclass

import numpy as np


@dataclass
class WeightFit:
    weights: np.ndarray
    trace: np.ndarray
    converged: bool
    iterations: int
    kkt_residual: float


def fit_kernel_weights(log_kernels, reference, *, pseudocount=5., max_iter=2000,
                       objective_tol=1e-8, kkt_tol=1e-3):
    """Fit simplex weights by EM with a positive Dirichlet pseudocount.

    ``log_kernels[i, j]`` evaluates a normalized component density. The
    normalized trace columns are penalized objective and ordinary log likelihood.
    Only weights are fitted; kernel locations, covariances and geometry stay fixed.
    """
    logs = np.asarray(log_kernels, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if logs.ndim != 2 or not len(logs) or ref.shape != (logs.shape[1],):
        raise ValueError('Expected nonempty n-by-k kernels and k reference weights.')
    if np.isnan(logs).any() or np.isposinf(logs).any() or not np.isfinite(logs.max(axis=1)).all():
        raise ValueError('Every observation needs a finite component log density.')
    if not np.isfinite(ref).all() or (ref <= 0).any() or not np.isclose(ref.sum(), 1.):
        raise ValueError('Reference weights must be strictly positive and sum to one.')
    if not np.isfinite(pseudocount) or pseudocount <= 0 or max_iter < 1:
        raise ValueError('A positive pseudocount and iteration budget are required.')
    if objective_tol <= 0 or kkt_tol <= 0:
        raise ValueError('Convergence tolerances must be positive.')
    ref = ref / ref.sum()
    n = len(logs)
    offsets = logs.max(axis=1)
    kernels = np.exp(logs - offsets[:, None])
    weights = ref.copy()

    def state(w):
        normalizers = kernels @ w
        likelihood = float(np.mean(np.log(normalizers) + offsets))
        objective = likelihood + pseudocount / n * float(ref @ np.log(w))
        gradient = (kernels.T @ (1. / normalizers) + pseudocount * ref / w) / (n + pseudocount)
        return objective, likelihood, float(np.max(np.abs(gradient - 1.)))

    objective, likelihood, kkt = state(weights)
    trace = [(objective, likelihood)]
    converged = False
    for iteration in range(1, max_iter + 1):
        normalizers = kernels @ weights
        counts = weights * (kernels.T @ (1. / normalizers))
        updated = (counts + pseudocount * ref) / (n + pseudocount)
        updated /= updated.sum()
        value, likelihood, kkt = state(updated)
        if value < objective - 1e-10:
            raise ArithmeticError('Penalized EM objective decreased beyond rounding tolerance.')
        gain = value - objective
        weights, objective = updated, value
        trace.append((objective, likelihood))
        if gain <= objective_tol and kkt <= kkt_tol:
            converged = True
            break
    return WeightFit(weights, np.asarray(trace), converged, iteration, kkt)
