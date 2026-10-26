from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln, logsumexp
from scipy.optimize import brentq
from scipy.stats import beta as beta_dist
from scipy.stats import multivariate_normal
from scipy.stats import norm

from .laplace import LaplaceProjection


def _rng(random_state: Any = None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _as_2d_x(x: ArrayLike, d: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        if arr.size != d:
            raise ValueError(f"Expected a vector of length {d}, got {arr.size}.")
        return arr.reshape(1, d)
    if arr.ndim != 2 or arr.shape[1] != d:
        raise ValueError(f"Expected shape (n, {d}), got {arr.shape}.")
    return arr


def _as_2d_k(x: ArrayLike, k: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        if arr.size != k:
            raise ValueError(f"Expected a vector of length {k}, got {arr.size}.")
        return arr.reshape(1, k)
    if arr.ndim != 2 or arr.shape[1] != k:
        raise ValueError(f"Expected shape (n, {k}), got {arr.shape}.")
    return arr


def _as_1d_values(x: ArrayLike) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=float)
    is_scalar = arr.ndim == 0
    return arr.reshape(-1), is_scalar


def _cov_matrix(cov: ArrayLike, d: int) -> np.ndarray:
    arr = np.asarray(cov, dtype=float)
    if arr.ndim == 0:
        return np.eye(d) * float(arr)
    if arr.ndim == 1:
        if arr.size != d:
            raise ValueError(f"Diagonal covariance must have length {d}.")
        return np.diag(arr)
    if arr.shape != (d, d):
        raise ValueError(f"Covariance must have shape ({d}, {d}).")
    return arr


def _weighted_grid(n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    if n_grid < 8:
        raise ValueError("n_grid must be at least 8 for stable tube integration.")
    s = (np.arange(n_grid, dtype=float) + 0.5) / n_grid
    w = np.full(n_grid, 1.0 / n_grid)
    return s, w


def _normalize_dims(dims: tuple[int, ...] | list[int], d: int) -> tuple[int, ...]:
    out = tuple(int(dim) for dim in dims)
    if len(out) == 0:
        raise ValueError("dims must contain at least one coordinate.")
    if len(set(out)) != len(out):
        raise ValueError("dims must not contain duplicates.")
    if any(dim < 0 or dim >= d for dim in out):
        raise ValueError(f"dims must be between 0 and {d - 1}, got {out}.")
    return out


def _kernel_logpdf(
    x: np.ndarray,
    *,
    mean: np.ndarray,
    cov: np.ndarray,
    kernel: str,
    kernel_df: float,
) -> np.ndarray:
    if kernel == "gaussian":
        return multivariate_normal.logpdf(x, mean=mean, cov=cov)

    d = int(mean.size)
    centered = np.asarray(x, dtype=float) - mean
    if kernel == "student_t":
        if kernel_df <= 2.0:
            raise ValueError("student_t tube kernels require kernel_df > 2 so cov is finite.")
        shape = cov * ((kernel_df - 2.0) / kernel_df)
        chol = np.linalg.cholesky(shape)
        z = np.linalg.solve(chol, centered.T).T
        quad = np.sum(z**2, axis=1)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        log_const = (
            gammaln((kernel_df + d) / 2.0)
            - gammaln(kernel_df / 2.0)
            - 0.5 * (d * np.log(kernel_df * np.pi) + log_det)
        )
        return log_const - 0.5 * (kernel_df + d) * np.log1p(quad / kernel_df)

    if kernel == "laplace":
        chol = np.linalg.cholesky(cov)
        z = np.linalg.solve(chol, centered.T).T
        scale = 1.0 / np.sqrt(2.0)
        log_det = np.sum(np.log(np.diag(chol)))
        return np.sum(-np.log(2.0 * scale) - np.abs(z) / scale, axis=1) - log_det

    raise ValueError(f"Unknown tube kernel {kernel!r}.")


def _kernel_noise(
    rng: np.random.Generator,
    *,
    size: int,
    cov: np.ndarray,
    kernel: str,
    kernel_df: float,
) -> np.ndarray:
    d = int(cov.shape[0])
    if kernel == "gaussian":
        return rng.multivariate_normal(np.zeros(d), cov, size=size)
    if kernel == "student_t":
        if kernel_df <= 2.0:
            raise ValueError("student_t tube kernels require kernel_df > 2 so cov is finite.")
        shape = cov * ((kernel_df - 2.0) / kernel_df)
        z = rng.multivariate_normal(np.zeros(d), shape, size=size)
        chi = rng.chisquare(kernel_df, size=size)
        return z / np.sqrt(chi[:, None] / kernel_df)
    if kernel == "laplace":
        chol = np.linalg.cholesky(cov)
        z = rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=(size, d))
        return z @ chol.T
    raise ValueError(f"Unknown tube kernel {kernel!r}.")


class Uniform01:
    """Uniform distribution on [0, 1] for branch coordinates."""

    def rvs(self, size: int, random_state: Any = None) -> np.ndarray:
        return _rng(random_state).uniform(0.0, 1.0, size=size)

    def pdf(self, s: ArrayLike) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return np.where((0.0 <= s) & (s <= 1.0), 1.0, 0.0)

    def mean(self) -> float:
        return 0.5


@dataclass(frozen=True)
class Beta01:
    """Beta distribution on [0, 1] for branch coordinates."""

    a: float
    b: float

    def rvs(self, size: int, random_state: Any = None) -> np.ndarray:
        return beta_dist.rvs(self.a, self.b, size=size, random_state=_rng(random_state))

    def pdf(self, s: ArrayLike) -> np.ndarray:
        return beta_dist.pdf(s, self.a, self.b)

    def mean(self) -> float:
        return self.a / (self.a + self.b)


@dataclass
class Component:
    name: str
    weight: float
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def d(self) -> int:
        raise NotImplementedError

    def rvs(self, size: int, random_state: Any = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        raise NotImplementedError

    def logpdf(self, x: np.ndarray, n_grid: int = 256) -> np.ndarray:
        raise NotImplementedError

    def marginal_logpdf(
        self,
        x: np.ndarray,
        dims: tuple[int, ...],
        n_grid: int = 256,
    ) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, x: np.ndarray, n_grid: int = 256) -> np.ndarray:
        raise NotImplementedError

    def marginal_cdf(self, x: ArrayLike, dim: int = 0, n_grid: int = 512) -> np.ndarray:
        raise NotImplementedError

    def mean(self, n_grid: int = 2048) -> np.ndarray:
        raise NotImplementedError

    def second_moment(self, n_grid: int = 2048) -> np.ndarray:
        raise NotImplementedError


@dataclass
class GaussianComponent(Component):
    """Full-dimensional Gaussian or atom-like Gaussian component."""

    loc: ArrayLike = field(default_factory=list)
    cov: ArrayLike = field(default_factory=list)

    def __post_init__(self) -> None:
        self.loc = np.asarray(self.loc, dtype=float)
        self.cov = _cov_matrix(self.cov, self.loc.size)
        self.kind = "gaussian"

    @property
    def d(self) -> int:
        return int(self.loc.size)

    def rvs(self, size: int, random_state: Any = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        rng = _rng(random_state)
        x = rng.multivariate_normal(self.loc, self.cov, size=size)
        info = {"s": np.full(size, np.nan)}
        return x, info

    def logpdf(self, x: np.ndarray, n_grid: int = 256) -> np.ndarray:
        return multivariate_normal.logpdf(x, mean=self.loc, cov=self.cov)

    def marginal_logpdf(
        self,
        x: np.ndarray,
        dims: tuple[int, ...],
        n_grid: int = 256,
    ) -> np.ndarray:
        idx = np.ix_(dims, dims)
        return multivariate_normal.logpdf(x, mean=self.loc[list(dims)], cov=self.cov[idx])

    def cdf(self, x: np.ndarray, n_grid: int = 256) -> np.ndarray:
        vals = multivariate_normal.cdf(x, mean=self.loc, cov=self.cov)
        return np.atleast_1d(vals)

    def marginal_cdf(self, x: ArrayLike, dim: int = 0, n_grid: int = 512) -> np.ndarray:
        values, is_scalar = _as_1d_values(x)
        vals = norm.cdf(values, loc=self.loc[dim], scale=np.sqrt(self.cov[dim, dim]))
        return vals[0] if is_scalar else vals

    def mean(self, n_grid: int = 2048) -> np.ndarray:
        return self.loc

    def second_moment(self, n_grid: int = 2048) -> np.ndarray:
        return self.cov + np.outer(self.loc, self.loc)


@dataclass
class TubeComponent(Component):
    """One SFD tube with a continuous coordinate S in [0, 1].

    Density, CDF and moment methods use a normalized midpoint approximation;
    ``rvs`` draws from the continuous coordinate law, not from that finite grid.
    Normalization preserves component mass but does not certify grid accuracy.
    """

    curve: Callable[[np.ndarray], np.ndarray] | None = None
    cov: ArrayLike = field(default_factory=list)
    s_dist: Any = field(default_factory=Uniform01)
    kernel: str = "gaussian"
    kernel_df: float = 5.0

    def __post_init__(self) -> None:
        if self.curve is None:
            raise ValueError("TubeComponent requires a curve callable.")
        probe = np.asarray(self.curve(np.asarray([0.5])), dtype=float)
        if probe.ndim != 2 or probe.shape[0] != 1:
            raise ValueError("curve(s) must return an array with shape (len(s), d).")
        self.cov = _cov_matrix(self.cov, probe.shape[1])
        self.kernel = str(self.kernel).lower()
        if self.kernel not in {"gaussian", "student_t", "laplace"}:
            raise ValueError("kernel must be one of 'gaussian', 'student_t', or 'laplace'.")
        if self.kernel == "student_t" and self.kernel_df <= 2.0:
            raise ValueError("student_t tube kernels require kernel_df > 2 so cov is finite.")
        self.kind = "tube"

    @property
    def d(self) -> int:
        return int(self.cov.shape[0])

    def _quadrature(self, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
        s, w = _weighted_grid(n_grid)
        h = np.asarray(self.s_dist.pdf(s), dtype=float)
        if h.shape != s.shape or not np.isfinite(h).all() or np.any(h < 0):
            raise ValueError("Coordinate density must be finite and nonnegative at grid nodes.")
        mass = w * h
        total = mass.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Coordinate quadrature has no positive finite mass; refine the grid.")
        return s, mass / total

    def rvs(self, size: int, random_state: Any = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        rng = _rng(random_state)
        s = np.asarray(self.s_dist.rvs(size=size, random_state=rng), dtype=float)
        mu = np.asarray(self.curve(s), dtype=float)
        noise = _kernel_noise(
            rng,
            size=size,
            cov=self.cov,
            kernel=self.kernel,
            kernel_df=self.kernel_df,
        )
        return mu + noise, {"s": s}

    def logpdf(self, x: np.ndarray, n_grid: int = 256) -> np.ndarray:
        x = _as_2d_x(x, self.d)
        s, mass = self._quadrature(n_grid)
        mu = np.asarray(self.curve(s), dtype=float)
        with np.errstate(divide="ignore"):
            log_mass = np.log(mass)

        terms = []
        for m in mu:
            terms.append(
                _kernel_logpdf(
                    x,
                    mean=m,
                    cov=self.cov,
                    kernel=self.kernel,
                    kernel_df=self.kernel_df,
                )
            )
        log_terms = np.vstack(terms).T
        return logsumexp(log_terms + log_mass, axis=1)

    def marginal_logpdf(
        self,
        x: np.ndarray,
        dims: tuple[int, ...],
        n_grid: int = 256,
    ) -> np.ndarray:
        x = _as_2d_k(x, len(dims))
        s, mass = self._quadrature(n_grid)
        mu = np.asarray(self.curve(s), dtype=float)[:, list(dims)]
        cov = self.cov[np.ix_(dims, dims)]
        with np.errstate(divide="ignore"):
            log_mass = np.log(mass)

        projection = LaplaceProjection(self.cov, dims) if self.kernel == "laplace" else None
        terms = []
        for m in mu:
            if projection is not None:
                terms.append(projection.logpdf(x - m))
                continue
            terms.append(
                _kernel_logpdf(
                    x,
                    mean=m,
                    cov=cov,
                    kernel=self.kernel,
                    kernel_df=self.kernel_df,
                )
            )
        log_terms = np.vstack(terms).T
        return logsumexp(log_terms + log_mass, axis=1)

    def cdf(self, x: np.ndarray, n_grid: int = 256) -> np.ndarray:
        if self.kernel != "gaussian":
            raise NotImplementedError("Tube CDF is implemented only for gaussian kernels.")
        x = _as_2d_x(x, self.d)
        s, mass = self._quadrature(n_grid)
        mu = np.asarray(self.curve(s), dtype=float)
        vals = np.zeros(x.shape[0], dtype=float)
        for wi, mi in zip(mass, mu):
            vals += wi * np.atleast_1d(
                multivariate_normal.cdf(x, mean=mi, cov=self.cov)
            )
        return vals

    def marginal_cdf(self, x: ArrayLike, dim: int = 0, n_grid: int = 512) -> np.ndarray:
        if self.kernel != "gaussian":
            raise NotImplementedError("Tube marginal CDF is implemented only for gaussian kernels.")
        values, is_scalar = _as_1d_values(x)
        s, mass = self._quadrature(n_grid)
        mu = np.asarray(self.curve(s), dtype=float)
        sd = np.sqrt(self.cov[dim, dim])
        vals = np.zeros(values.size, dtype=float)
        for wi, mi in zip(mass, mu):
            vals += wi * norm.cdf(values, loc=mi[dim], scale=sd)
        return vals[0] if is_scalar else vals

    def mean(self, n_grid: int = 2048) -> np.ndarray:
        s, weights = self._quadrature(n_grid)
        mu = np.asarray(self.curve(s), dtype=float)
        return weights @ mu

    def second_moment(self, n_grid: int = 2048) -> np.ndarray:
        s, weights = self._quadrature(n_grid)
        mu = np.asarray(self.curve(s), dtype=float)
        out = np.zeros((self.d, self.d))
        for wi, mi in zip(weights, mu):
            out += wi * (self.cov + np.outer(mi, mi))
        return out


class SFD:
    """Frozen Stratified Flow Distribution.

    Parameters
    ----------
    components:
        Sequence of `TubeComponent` or `GaussianComponent` objects. Component
        weights are normalized at construction time.
    name:
        Distribution name used in reprs and metadata.
    random_state:
        Optional seed or NumPy Generator used by default in `rvs`.
    """

    def __init__(self, components: list[Component], name: str = "sfd", random_state: Any = None):
        if not components:
            raise ValueError("SFD requires at least one component.")
        dims = {comp.d for comp in components}
        if len(dims) != 1:
            raise ValueError(f"All components must have the same dimension, got {sorted(dims)}.")
        weights = np.asarray([comp.weight for comp in components], dtype=float)
        if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("Component weights must be nonnegative and have positive sum.")

        self.components = list(components)
        self.name = name
        self.d = int(next(iter(dims)))
        self.weights = weights / weights.sum()
        for comp, weight in zip(self.components, self.weights):
            comp.weight = float(weight)
        self.random_state = _rng(random_state)

    @property
    def component_names(self) -> list[str]:
        return [comp.name for comp in self.components]

    def __repr__(self) -> str:
        return f"SFD(name={self.name!r}, d={self.d}, components={len(self.components)})"

    def rvs(
        self,
        size: int | tuple[int, ...] = 1,
        random_state: Any = None,
        return_info: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        rng = self.random_state if random_state is None else _rng(random_state)
        out_shape = (size,) if isinstance(size, int) else tuple(size)
        n = int(np.prod(out_shape))
        comp_idx = rng.choice(len(self.components), size=n, p=self.weights)
        x = np.empty((n, self.d), dtype=float)
        info: dict[str, np.ndarray] = {
            "component_index": comp_idx.copy(),
            "component": np.empty(n, dtype=object),
            "kind": np.empty(n, dtype=object),
            "s": np.full(n, np.nan),
        }

        for k, comp in enumerate(self.components):
            rows = np.flatnonzero(comp_idx == k)
            if rows.size == 0:
                continue
            xk, infok = comp.rvs(rows.size, random_state=rng)
            x[rows] = xk
            info["component"][rows] = comp.name
            info["kind"][rows] = comp.kind
            info["s"][rows] = infok.get("s", np.full(rows.size, np.nan))

        x = x.reshape(out_shape + (self.d,))
        if not return_info:
            return x
        for key, value in list(info.items()):
            info[key] = value.reshape(out_shape)
        return x, info

    def discretize(self, n_grid: int = 256) -> SFD:
        """Freeze tube coordinates to a normalized midpoint probability law.

        The returned SFD samples the same finite law as its density, CDF and
        moment methods. It stores the current node probabilities, centers and
        kernels independently of this distribution. Its ``n_grid`` evaluation
        arguments are immaterial: refine by discretizing the original SFD.
        This operation does not certify approximation to a continuous law.
        """
        from .finite_grid import discretize

        return discretize(self, n_grid=n_grid)

    def component_logpdf(self, x: ArrayLike, n_grid: int = 256) -> np.ndarray:
        x_arr = _as_2d_x(x, self.d)
        cols = []
        for weight, comp in zip(self.weights, self.components):
            cols.append(np.log(weight) + comp.logpdf(x_arr, n_grid=n_grid))
        return np.vstack(cols).T

    def component_marginal_logpdf(
        self,
        x: ArrayLike,
        dims: tuple[int, ...] | list[int],
        n_grid: int = 256,
    ) -> np.ndarray:
        dims = _normalize_dims(dims, self.d)
        x_arr = _as_2d_k(x, len(dims))
        cols = []
        for weight, comp in zip(self.weights, self.components):
            cols.append(np.log(weight) + comp.marginal_logpdf(x_arr, dims=dims, n_grid=n_grid))
        return np.vstack(cols).T

    def logpdf(self, x: ArrayLike, n_grid: int = 256) -> np.ndarray:
        vals = logsumexp(self.component_logpdf(x, n_grid=n_grid), axis=1)
        return vals[0] if np.asarray(x).ndim == 1 else vals

    def pdf(self, x: ArrayLike, n_grid: int = 256) -> np.ndarray:
        return np.exp(self.logpdf(x, n_grid=n_grid))

    def marginal_logpdf(
        self,
        x: ArrayLike,
        dims: tuple[int, ...] | list[int],
        n_grid: int = 256,
    ) -> np.ndarray:
        vals = logsumexp(
            self.component_marginal_logpdf(x, dims=dims, n_grid=n_grid),
            axis=1,
        )
        return vals[0] if np.asarray(x).ndim == 1 else vals

    def marginal_pdf(
        self,
        x: ArrayLike,
        dims: tuple[int, ...] | list[int],
        n_grid: int = 256,
    ) -> np.ndarray:
        """Marginal density on selected coordinates.

        Example: ``dist.marginal_pdf(points, dims=(0, 1))`` evaluates the
        bivariate marginal density of ``(X_1, X_2)``.
        """

        return np.exp(self.marginal_logpdf(x, dims=dims, n_grid=n_grid))

    def bivariate_pdf(
        self,
        x: ArrayLike,
        dims: tuple[int, int] = (0, 1),
        n_grid: int = 256,
    ) -> np.ndarray:
        """Convenience wrapper for a two-dimensional marginal density."""

        dims = _normalize_dims(dims, self.d)
        if len(dims) != 2:
            raise ValueError("bivariate_pdf requires exactly two dimensions.")
        return self.marginal_pdf(x, dims=dims, n_grid=n_grid)

    def cdf(self, x: ArrayLike, n_grid: int = 256) -> np.ndarray:
        """Joint CDF ``P(X_1 <= x_1, ..., X_d <= x_d)``.

        For tube components this is evaluated by midpoint quadrature over the
        latent branch coordinate. For coordinate-wise distribution functions,
        use `marginal_cdf`.
        """

        x_arr = _as_2d_x(x, self.d)
        vals = np.zeros(x_arr.shape[0], dtype=float)
        for weight, comp in zip(self.weights, self.components):
            vals += weight * comp.cdf(x_arr, n_grid=n_grid)
        return vals[0] if np.asarray(x).ndim == 1 else vals

    def marginal_cdf(self, x: ArrayLike, dim: int = 0, n_grid: int = 512) -> np.ndarray:
        """Marginal CDF for one coordinate, ``P(X_dim <= x)``."""

        if not 0 <= dim < self.d:
            raise ValueError(f"dim must be in [0, {self.d - 1}], got {dim}.")
        values, is_scalar = _as_1d_values(x)
        vals = np.zeros(values.size, dtype=float)
        for weight, comp in zip(self.weights, self.components):
            vals += weight * comp.marginal_cdf(values, dim=dim, n_grid=n_grid)
        return vals[0] if is_scalar else vals

    def ppf(
        self,
        q: ArrayLike,
        dim: int = 0,
        n_grid: int = 512,
        bracket: tuple[float, float] | None = None,
        xtol: float = 1e-6,
    ) -> np.ndarray:
        """Marginal percent-point function, i.e. coordinate-wise quantile.

        Multivariate distributions do not have a unique scalar quantile. This
        method follows the practical SciPy convention used for one-dimensional
        marginals: it returns `x` such that `P(X_dim <= x) = q`.
        """

        if not 0 <= dim < self.d:
            raise ValueError(f"dim must be in [0, {self.d - 1}], got {dim}.")
        probs, is_scalar = _as_1d_values(q)
        if np.any((probs <= 0.0) | (probs >= 1.0)):
            raise ValueError("q must be strictly between 0 and 1.")

        if bracket is None:
            mu = self.mean(n_grid=n_grid)[dim]
            sd = float(np.sqrt(max(self.cov(n_grid=n_grid)[dim, dim], 1e-12)))
            lo, hi = mu - 12.0 * sd, mu + 12.0 * sd
        else:
            lo, hi = map(float, bracket)
            if not lo < hi:
                raise ValueError("bracket must satisfy lo < hi.")

        def f(x0: float, prob: float) -> float:
            return float(self.marginal_cdf(x0, dim=dim, n_grid=n_grid) - prob)

        out = np.empty(probs.size, dtype=float)
        for i, prob in enumerate(probs):
            left, right = lo, hi
            for _ in range(12):
                if f(left, prob) <= 0.0 and f(right, prob) >= 0.0:
                    break
                width = right - left
                left -= width
                right += width
            out[i] = brentq(lambda z: f(z, prob), left, right, xtol=xtol)
        return out[0] if is_scalar else out

    def quantile(self, q: ArrayLike, dim: int = 0, **kwargs: Any) -> np.ndarray:
        """Alias for `ppf`, provided for users who prefer quantile terminology."""

        return self.ppf(q, dim=dim, **kwargs)

    def responsibilities(self, x: ArrayLike, n_grid: int = 256) -> np.ndarray:
        log_comp = self.component_logpdf(x, n_grid=n_grid)
        log_total = logsumexp(log_comp, axis=1, keepdims=True)
        return np.exp(log_comp - log_total)

    def mean(self, n_grid: int = 2048) -> np.ndarray:
        out = np.zeros(self.d)
        for weight, comp in zip(self.weights, self.components):
            out += weight * comp.mean(n_grid=n_grid)
        return out

    def cov(self, n_grid: int = 2048) -> np.ndarray:
        second = np.zeros((self.d, self.d))
        for weight, comp in zip(self.weights, self.components):
            second += weight * comp.second_moment(n_grid=n_grid)
        mu = self.mean(n_grid=n_grid)
        return second - np.outer(mu, mu)
