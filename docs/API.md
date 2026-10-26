# flowdistribution Python package

Small SciPy-style simulation package for Stratified Flow Distributions (SFDs).

```python
from flowdistribution import snowflake, cartoon_face

dist = snowflake(random_state=20260623)
x, info = dist.rvs(size=1000, return_info=True)

print(x.shape)
print(dist.pdf(x[:5]))
print(dist.cdf(x[:5]))
print(dist.mean())
print(dist.ppf([0.025, 0.5, 0.975], dim=0))
print(info["component"][:5])
```

Install locally from this repository:

```bash
python3 -m pip install -e .
```

Run the small fitted-model example after installation:

```bash
python reproduce.py smoke --out runs/api-smoke
```

Run the first JMLR-style density benchmark scaffold:

```bash
PYTHONPATH=src python3 scripts/run_jmlr_density_benchmark.py \
  --scenarios snowflake,doraemon,maze,ecg \
  --n-train 300 --n-test 250 --reps 3
```

This writes held-out log-score and runtime tables under
`outputs/jmlr_density_benchmark/`. The scaffold compares a generating-law
reference with Gaussian, Gaussian-mixture and KDE baselines. A finite test
sample need not give the generating density the highest empirical score.
The controlled SFD fitting studies use separate experiment runners under
`scripts/`.

The API intentionally follows the familiar `scipy.stats` rhythm:

- `dist.rvs(size, random_state=None, return_info=False)`
- `dist.pdf(x, n_grid=256)`
- `dist.logpdf(x, n_grid=256)`
- `dist.marginal_pdf(x, dims=(0, 1), n_grid=256)` for selected-coordinate densities
- `dist.bivariate_pdf(x, dims=(0, 1), n_grid=256)` for bivariate marginal densities
- `dist.cdf(x, n_grid=256)` for the joint CDF
- `dist.marginal_cdf(x, dim=0, n_grid=512)` for coordinate-wise CDFs
- `dist.ppf(q, dim=0, n_grid=512)` for coordinate-wise quantiles
- `dist.quantile(q, dim=0, n_grid=512)` as an alias for `ppf`
- `dist.responsibilities(x, n_grid=256)`
- `dist.mean(n_grid=2048)`
- `dist.cov(n_grid=2048)`

For multivariate SFDs, `quantile` is intentionally marginal because there is
no unique scalar multivariate quantile. Use `dim=0`, `dim=1`, etc. to choose
the coordinate.

## Numerical Probability Laws

`TubeComponent.rvs` draws a continuous branch coordinate from `s_dist`.
Density, CDF and moment calculations use normalized midpoint quadrature.
At a fixed `n_grid`, these calculations describe a finite mixture of the
same observation kernels. Refine the grid to check its approximation to the
continuous-coordinate law; normalization alone does not establish accuracy.
Use the same grid when comparing those numerical summaries.

For simulation and evaluation under exactly the same finite-coordinate law,
construct an explicit discretization:

```python
grid = dist.discretize(n_grid=256)
x, info = grid.rvs(size=1000, random_state=20261020, return_info=True)
print(grid.logpdf(x[:5]))
print(grid.mean())
```

The returned SFD stores normalized node probabilities, kernel centers and
kernel parameters, preserving component labels and weights. Its tube
coordinates in `info["s"]` are grid nodes. Density, CDF, quantile and moment
methods all use that stored finite law, regardless of their `n_grid` argument.
It is independent of later changes to the original distribution, and creating
it does not advance the original random generator. To refine, call
`dist.discretize` on the original continuous-coordinate SFD with a new grid
size. Discretizing an already finite tube is rejected. Ordinary `dist.rvs`
continues to draw continuous coordinates, with unchanged seeded behavior.
Discretization gives a coherent finite probability model; accuracy relative
to the continuous-coordinate model still requires a refinement check.
Only the built-in tube and Gaussian components are supported. The operation
requires finite centers, symmetric positive-definite covariances and valid
probability weights; malformed parameters are rejected before sampling.

Gaussian and Student-t noise retain their usual selected-coordinate families.
The `laplace` kernel is `L @ Z`, with `L = chol(cov)` and independent,
unit-variance univariate Laplace coordinates in `Z`. This is a frame-dependent
product-Laplace law. Its marginals project the original loading matrix;
rebuilding a Laplace kernel from a covariance submatrix gives a different
law in general.

One-coordinate and several multivariate Laplace projections are analytic.
General correlated projections use deterministic quadrature with an evaluation
cap and raise `ArithmeticError` when unresolved. High-dimensional correlated
projections can be expensive; use small query batches. Joint/marginal CDFs and
quantiles remain available for Gaussian tube kernels only.

Available factory functions:

- `snowflake()`
- `cartoon_face()`
- `koch_prefractal()`
- `torus_knot()`
- `maze_graph()`
- `ecg_like()`
