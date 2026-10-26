"""Projections of the package's Cholesky-transformed product-Laplace law."""
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, nquad
from scipy.linalg import expm, qr, solve_continuous_lyapunov
from scipy.special import logsumexp


ROOT_TWO = np.sqrt(2.)
LOG_HEIGHT = -.5 * np.log(2.)


def _phase_logpdf(values, coefficients):
    """Sum of independent scaled unit-variance Laplace variables."""
    scales = np.abs(coefficients[coefficients != 0]) / ROOT_TWO
    if len(scales) == 1:
        return -np.log(2 * scales[0]) - np.abs(values) / scales[0]
    scale = float(scales.max())
    rates = np.sort(scale / scales)
    generator = np.diag(-rates) + np.diag(rates[:-1], 1)
    exit_rate = np.zeros(len(rates))
    exit_rate[-1] = rates[-1]
    moment = solve_continuous_lyapunov(generator, -np.outer(exit_rate, exit_rate))
    vector = moment[:, 0]
    shifted = generator + np.eye(len(rates))
    output = np.empty(len(values))
    for i, value in enumerate(np.abs(values) / scale):
        factor = float(expm(shifted * value)[0] @ vector)
        if not np.isfinite(factor) or factor <= 0:
            raise ArithmeticError("Unresolved Laplace phase-type density; rescale the loading matrix.")
        output[i] = -value + np.log(factor) - np.log(scale)
    return output


def _log_absolute_integral(a, b):
    """Log integral of exp(-sqrt(2) * (|t| + sum |a-b*t|)) over the real line."""
    active = b != 0
    with np.errstate(over="ignore"):
        roots = np.r_[0., a[active] / b[active]]
    if not np.isfinite(roots).all():
        raise ArithmeticError("Unresolved Laplace integration breakpoints; rescale the query.")
    roots = np.unique(roots)
    boundaries = np.r_[-np.inf, roots, np.inf]
    tail_rate = ROOT_TWO * (1 + np.abs(b).sum())
    terms = []

    def log_height(t):
        return -ROOT_TWO * (abs(t) + np.abs(a - b * t).sum())

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if np.isneginf(left):
            terms.append(log_height(right) - np.log(tail_rate))
        elif np.isposinf(right):
            terms.append(log_height(left) - np.log(tail_rate))
        else:
            middle = left / 2 + right / 2
            slope = -ROOT_TWO * (np.sign(middle) - b @ np.sign(a - b * middle))
            width = right - left
            if slope == 0:
                terms.append(log_height(middle) + np.log(width))
            else:
                endpoint = right if slope > 0 else left
                terms.append(log_height(endpoint) + np.log(-np.expm1(-abs(slope) * width))
                             - np.log(abs(slope)))
    return float(logsumexp(terms))


class LaplaceProjection:
    """Density of selected rows of LZ, retaining the original L=chol(cov).

    One-coordinate, square-loading, independent-row and codimension-one
    cases are analytic. Other projections use bounded deterministic quadrature.
    The evaluation budget is shared across this object's density queries.
    """

    def __init__(self, cov, dims, max_evaluations=100000):
        covariance = np.asarray(cov, dtype=float)
        loading = np.linalg.cholesky(covariance)[list(dims)]
        active = np.any(loading != 0, axis=0)
        loading = loading[:, active]
        self.scale = np.sqrt(np.sum(loading**2, axis=1))
        self.loading = loading / self.scale[:, None]
        self.log_jacobian = float(np.log(self.scale).sum())
        self.k, self.latent_dim = self.loading.shape
        if (not isinstance(max_evaluations, int) or max_evaluations < 1
                or not len(set(dims)) == len(dims)):
            raise ValueError("Distinct dimensions and a positive integration budget are required.")
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.independent = np.all(np.sum(self.loading != 0, axis=0) <= 1)
        if self.k == self.latent_dim:
            self.inverse = np.linalg.inv(self.loading)
            self.log_determinant = float(np.linalg.slogdet(self.loading)[1])
        elif self.k > 1 and not self.independent:
            _, _, pivot = qr(self.loading, mode="economic", pivoting=True)
            basis = self.loading[:, pivot[:self.k]]
            self.inverse = np.linalg.inv(basis)
            self.log_determinant = float(np.linalg.slogdet(basis)[1])
            self.free = self.inverse @ self.loading[:, pivot[self.k:]]

    def _projected_logpdf(self, value):
        a = self.inverse @ value
        b = self.free[:, 0]
        if self.free.shape[1] == 1:
            return self.latent_dim * LOG_HEIGHT - self.log_determinant + _log_absolute_integral(a, b)

        def integrand(*quantiles):
            self.evaluations += 1
            if self.evaluations > self.max_evaluations:
                raise ArithmeticError("Laplace projection quadrature budget exceeded; query fewer points.")
            u = np.asarray(quantiles)
            coordinates = np.where(u < .5, np.log(2 * u), -np.log(2 * (1 - u))) / ROOT_TWO
            shifted = a - self.free[:, 1:] @ coordinates
            return np.exp(_log_absolute_integral(shifted, b))

        count = self.free.shape[1] - 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            density, error = nquad(integrand, [(0., 1.)] * count,
                opts=dict(epsabs=0., epsrel=1e-8, points=[.5], limit=150))
        if (any(issubclass(item.category, IntegrationWarning) for item in caught)
                or not np.isfinite(density) or density <= 0 or error > 1e-6 * density):
            raise ArithmeticError("Laplace projection quadrature did not resolve the requested density.")
        return (self.k + 1) * LOG_HEIGHT - self.log_determinant + np.log(density)

    def logpdf(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.k or not np.isfinite(values).all():
            raise ValueError("Expected finite n-by-k projected noise coordinates.")
        standardized = values / self.scale
        if self.k == self.latent_dim:
            latent = standardized @ self.inverse.T
            return (self.latent_dim * LOG_HEIGHT - self.log_determinant
                    - ROOT_TWO * np.abs(latent).sum(axis=1) - self.log_jacobian)
        if self.k == 1:
            return _phase_logpdf(standardized[:, 0], self.loading[0]) - self.log_jacobian
        if self.independent:
            return sum(_phase_logpdf(standardized[:, i], self.loading[i]) for i in range(self.k)) - self.log_jacobian
        return np.asarray([self._projected_logpdf(value) for value in standardized]) - self.log_jacobian
