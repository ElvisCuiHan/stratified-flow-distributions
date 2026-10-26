"""Explicit finite-coordinate laws using the existing SFD kernel methods."""
from copy import deepcopy
from operator import index
from typing import Any

import numpy as np

from .core import GaussianComponent, SFD, TubeComponent, _kernel_noise, _rng


def _stored_array(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _stored_covariance(value):
    covariance = _stored_array(value)
    if not np.isfinite(covariance).all() or not np.array_equal(covariance, covariance.T):
        raise ValueError("Finite kernels require a finite symmetric covariance.")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("Finite kernels require a positive definite covariance.") from error
    return covariance


class _FiniteTubeComponent(TubeComponent):
    """A node mixture; inherited evaluations share the stored quadrature."""

    def __init__(self, source, weight, n_grid):
        if source.kernel not in {"gaussian", "student_t", "laplace"}:
            raise ValueError("Unsupported finite tube kernel.")
        if source.kernel == "student_t" and (not np.isfinite(source.kernel_df) or source.kernel_df <= 2.):
            raise ValueError("Student-t finite kernels require finite kernel_df > 2.")
        nodes, probabilities = source._quadrature(n_grid)
        centers = np.asarray(source.curve(nodes), dtype=float)
        if centers.shape != (n_grid, source.d) or not np.isfinite(centers).all():
            raise ValueError("curve(s) must return finite centers with shape (n_grid, d).")
        self.name = source.name
        self.weight = float(weight)
        self.kind = "tube"
        self.metadata = deepcopy(source.metadata)
        self.metadata.update(coordinate_law="normalized_midpoint", coordinate_nodes=n_grid)
        self._nodes = _stored_array(nodes)
        self._probabilities = _stored_array(probabilities)
        self._centers = _stored_array(centers)
        self.cov = _stored_covariance(source.cov)
        self.kernel = source.kernel
        self.kernel_df = source.kernel_df
        self.s_dist = None
        self.curve = self._centers_at_nodes

    def _centers_at_nodes(self, s):
        values = np.asarray(s, dtype=float)
        positions = np.searchsorted(self._nodes, values)
        if np.any(positions >= self._nodes.size):
            raise ValueError("A finite tube is defined only at its stored coordinate nodes.")
        if np.any(self._nodes[positions] != values):
            raise ValueError("A finite tube is defined only at its stored coordinate nodes.")
        return self._centers[positions]

    def _quadrature(self, n_grid):
        return self._nodes, self._probabilities

    def rvs(self, size: int, random_state: Any = None):
        rng = _rng(random_state)
        selected = rng.choice(self._nodes.size, size=size, p=self._probabilities)
        noise = _kernel_noise(rng, size=size, cov=self.cov,
                              kernel=self.kernel, kernel_df=self.kernel_df)
        return self._centers[selected] + noise, {"s": self._nodes[selected]}


def discretize(law, n_grid=256):
    if isinstance(n_grid, (bool, np.bool_)):
        raise TypeError("n_grid must be an integer, not a boolean.")
    n_grid = index(n_grid)
    if n_grid < 8:
        raise ValueError("n_grid must be at least 8.")
    weights = np.asarray(law.weights, dtype=float)
    if (weights.shape != (len(law.components),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., rtol=1e-12, atol=0.)):
        raise ValueError("Source component weights must be a probability vector.")
    components = []
    for component, weight in zip(law.components, weights):
        if isinstance(component, _FiniteTubeComponent):
            raise ValueError("Already a finite tube; discretize the original continuous SFD to refine it.")
        if type(component) is TubeComponent:
            components.append(_FiniteTubeComponent(component, weight, n_grid))
        elif type(component) is GaussianComponent:
            copied = deepcopy(component)
            copied.weight = float(weight)
            if copied.loc.shape != (law.d,) or not np.isfinite(copied.loc).all():
                raise ValueError("Gaussian locations must be finite vectors of dimension d.")
            copied.loc = _stored_array(copied.loc)
            copied.cov = _stored_covariance(copied.cov)
            components.append(copied)
        else:
            raise TypeError(f"Unsupported component for discretization: {type(component).__name__}.")
    return SFD(components, name=f"{law.name}_grid{n_grid}",
               random_state=deepcopy(law.random_state))
