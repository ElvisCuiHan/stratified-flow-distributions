import numpy as np

from flowdistribution import cartoon_face, ecg_like, maze_graph, snowflake, torus_knot
from scripts.run_jmlr_density_benchmark import fit_gaussian, fit_gmm_em, mean_log_score


def test_snowflake_rvs_pdf_shapes():
    dist = snowflake(random_state=1)
    x, info = dist.rvs(size=32, random_state=2, return_info=True)
    assert x.shape == (32, 3)
    assert info["component"].shape == (32,)
    p = dist.pdf(x[:4], n_grid=64)
    assert p.shape == (4,)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0)
    bp = dist.bivariate_pdf(x[:4, :2], dims=(0, 1), n_grid=64)
    assert bp.shape == (4,)
    assert np.all(np.isfinite(bp))
    assert np.all(bp >= 0)
    c = dist.cdf(x[:4], n_grid=32)
    assert c.shape == (4,)
    assert np.all((0 <= c) & (c <= 1))
    mc = dist.marginal_cdf([0.0, 0.5], dim=0, n_grid=64)
    assert mc.shape == (2,)
    assert mc[0] <= mc[1]
    qs = dist.ppf([0.25, 0.5, 0.75], dim=0, n_grid=64)
    assert np.all(np.diff(qs) > 0)
    assert np.isclose(dist.quantile(0.5, dim=0, n_grid=64), qs[1])


def test_factory_dimensions():
    for factory in [cartoon_face, maze_graph, torus_knot]:
        dist = factory(random_state=3)
        x = dist.rvs(size=8, random_state=4)
        assert x.shape[0] == 8
        assert x.shape[1] == dist.d
        assert np.all(np.isfinite(dist.mean(n_grid=64)))

    ecg = ecg_like(random_state=5)
    x = ecg.rvs(size=8, random_state=6)
    assert x.shape == (8, 2)


def test_alternative_tube_kernels_smoke():
    for kernel in ["student_t", "laplace"]:
        dist = snowflake(kernel=kernel, kernel_df=5.0, random_state=11)
        x = dist.rvs(size=16, random_state=12)
        assert x.shape == (16, 3)
        logp = dist.logpdf(x[:5], n_grid=32)
        assert logp.shape == (5,)
        assert np.all(np.isfinite(logp))
        bp = dist.bivariate_pdf(x[:5, :2], dims=(0, 1), n_grid=32)
        assert bp.shape == (5,)
        assert np.all(np.isfinite(bp))


def test_jmlr_benchmark_baselines_smoke():
    dist = snowflake(random_state=7)
    train = dist.rvs(size=40, random_state=8)
    test = dist.rvs(size=12, random_state=9)

    gaussian = fit_gaussian(train, reg=1e-5)
    gaussian_score = mean_log_score(gaussian.logpdf(test))
    assert np.isfinite(gaussian_score)

    gmm = fit_gmm_em(
        train,
        2,
        random_state=10,
        reg=1e-5,
        max_iter=4,
        tol=1e-6,
    )
    gmm_score = mean_log_score(gmm.logpdf(test))
    assert np.isfinite(gmm_score)
    assert np.isclose(gmm.weights.sum(), 1.0)
