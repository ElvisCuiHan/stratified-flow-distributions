"""Equality-constrained Newton fitting of positive fixed-kernel mixture weights."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .occupancy import WeightFit


def fit_kernel_weights_newton(log_kernels, reference, *, pseudocount=.5,
                             max_iter=100, objective_tol=1e-10, kkt_tol=1e-7, initial=None):
    """Optimize the same strictly concave Dirichlet-MAP target as weight EM.

    A feasible Newton direction satisfies the simplex equality constraint.
    Fraction-to-boundary and Armijo checks keep weights positive and the
    penalized objective nondecreasing. Kernels must be fixed and normalized.
    """
    logs = np.asarray(log_kernels, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if (logs.ndim != 2 or not len(logs) or not logs.shape[1]
            or ref.shape != (logs.shape[1],)):
        raise ValueError("Expected n-by-K fixed kernels and K reference probabilities.")
    if (np.isnan(logs).any() or np.isposinf(logs).any()
            or not np.isfinite(logs.max(axis=1)).all()
            or not np.isfinite(ref).all() or np.any(ref <= 0) or not np.isclose(ref.sum(), 1.)):
        raise ValueError("Finite row normalizers and positive reference probabilities are required.")
    if (not isinstance(max_iter, int) or max_iter < 1
            or not all(np.isfinite(v) for v in (pseudocount, objective_tol, kkt_tol))
            or min(pseudocount, objective_tol, kkt_tol) <= 0):
        raise ValueError("Invalid Newton controls.")
    ref = ref / ref.sum()
    offsets = logs.max(axis=1)
    kernels = np.exp(logs - offsets[:, None])
    n, k = kernels.shape

    def state(w):
        denominator = kernels @ w
        likelihood = float(np.mean(np.log(denominator) + offsets))
        objective = likelihood + pseudocount / n * float(ref @ np.log(w))
        gradient = kernels.T @ (1. / denominator) + pseudocount * ref / w
        kkt = float(np.max(np.abs(gradient / (n + pseudocount) - 1.)))
        return objective, likelihood, gradient, kkt, denominator

    weights = ref.copy() if initial is None else np.asarray(initial, dtype=float).copy()
    if (weights.shape != ref.shape or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.)):
        raise ValueError("Initial weights must be positive and sum to one.")
    if initial is not None:
        weights /= weights.sum()
    objective, likelihood, gradient, kkt, denominator = state(weights)
    trace = [[objective, likelihood]]
    if kkt <= kkt_tol:
        return WeightFit(weights, np.asarray(trace), True, 0, kkt)
    for iteration in range(1, max_iter + 1):
        scaled = kernels / denominator[:, None]
        curvature = scaled.T @ scaled + np.diag(pseudocount * ref / weights**2)
        factor = cho_factor(curvature, check_finite=True)
        inverse_gradient = cho_solve(factor, gradient)
        inverse_one = cho_solve(factor, np.ones(k))
        multiplier = inverse_gradient.sum() / inverse_one.sum()
        direction = inverse_gradient - multiplier * inverse_one
        direction -= direction.sum() / k
        negative = direction < 0
        step = min(1., .99 * np.min(-weights[negative] / direction[negative])) if negative.any() else 1.
        slope = float(gradient @ direction) / n
        accepted = None
        for _ in range(60):
            candidate = weights + step * direction
            candidate /= candidate.sum()
            if np.all(candidate > 0):
                updated = state(candidate)
                rounding = 1e-13 * max(1., abs(objective))
                if updated[0] >= objective + 1e-4 * step * max(slope, 0.) - rounding:
                    accepted = candidate, updated
                    break
            step *= .5
        if accepted is None:
            return WeightFit(weights, np.asarray(trace), False, iteration - 1, kkt)
        weights, updated = accepted
        gain = updated[0] - objective
        objective, likelihood, gradient, kkt, denominator = updated
        trace.append([objective, likelihood])
        if gain <= objective_tol and kkt <= kkt_tol:
            return WeightFit(weights, np.asarray(trace), True, iteration, kkt)
    return WeightFit(weights, np.asarray(trace), False, max_iter, kkt)
