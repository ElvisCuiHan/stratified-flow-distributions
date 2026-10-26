from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import beta, laplace, multivariate_normal, multivariate_t, norm

from flowdistribution import GaussianComponent, SFD, TubeComponent


def mixture(kernel="gaussian", coordinate=None, covariance=None):
    tube = TubeComponent(name="branch", weight=.7, kind="tube",
        curve=lambda s: np.column_stack((s, s**2)),
        cov=np.array([[.04, .012], [.012, .09]]) if covariance is None else covariance,
        s_dist=beta(.3, 2) if coordinate is None else coordinate,
        kernel=kernel, kernel_df=7.)
    background = GaussianComponent(name="background", weight=.3, kind="gaussian",
        loc=[-.2, .6], cov=np.diag([.2, .3]))
    return SFD([tube, background], random_state=191), tube, background


def independent_nodes(n):
    s = (np.arange(n) + .5) / n
    p = beta.pdf(s, .3, 2.)
    return s, p / p.sum()


def independent_kernel(x, mean, covariance, kernel):
    if kernel == "gaussian":
        return multivariate_normal.logpdf(x, mean=mean, cov=covariance)
    if kernel == "student_t":
        return multivariate_t.logpdf(x, loc=mean, shape=covariance * (5. / 7.), df=7.)
    loading = np.linalg.cholesky(covariance)
    standardized = np.linalg.solve(loading, (x - mean).T).T
    return laplace.logpdf(standardized, scale=1 / np.sqrt(2)).sum(axis=1) - np.log(np.diag(loading)).sum()


@pytest.mark.parametrize("kernel", ["gaussian", "student_t", "laplace"])
def test_finite_joint_density_and_responsibilities_match_independent_mixture(kernel):
    law, tube, background = mixture(kernel)
    frozen = law.discretize(8)
    s, probabilities = independent_nodes(8)
    points = np.array([[.1, -.4], [.7, .3], [2., -1.]])
    centers = np.column_stack((s, s**2))
    branch = logsumexp(np.array([independent_kernel(points, center, tube.cov, kernel)
        for center in centers]).T + np.log(probabilities), axis=1)
    component_logs = np.column_stack((np.log(.7) + branch,
        np.log(.3) + multivariate_normal.logpdf(points, mean=background.loc, cov=background.cov)))
    expected = logsumexp(component_logs, axis=1)
    np.testing.assert_allclose(frozen.component_logpdf(points), component_logs, atol=2e-14)
    np.testing.assert_allclose(frozen.logpdf(points), expected, atol=2e-14)
    np.testing.assert_allclose(frozen.pdf(points), np.exp(expected), rtol=2e-14)
    np.testing.assert_allclose(frozen.responsibilities(points),
        np.exp(component_logs - expected[:, None]), atol=2e-14)
    assert frozen.logpdf(points[0]) == pytest.approx(expected[0], abs=2e-14)
    np.testing.assert_allclose(frozen.pdf(points, n_grid=4096), np.exp(expected), rtol=2e-14)


@pytest.mark.parametrize("kernel", ["gaussian", "student_t", "laplace"])
def test_marginal_projection_uses_original_kernel_frame(kernel):
    law, tube, background = mixture(kernel)
    frozen = law.discretize(8)
    s, probabilities = independent_nodes(8)
    points = np.array([-.5, 0., .8])
    if kernel == "laplace":
        a, b = np.abs(np.linalg.cholesky(tube.cov)[1]) / np.sqrt(2)
        distance = np.abs(points[:, None] - s[None, :]**2)
        densities = (a * np.exp(-distance / a) - b * np.exp(-distance / b)) / (2 * (a*a - b*b))
    elif kernel == "student_t":
        from scipy.stats import t
        densities = t.pdf(points[:, None], df=7, loc=s[None, :]**2,
                          scale=np.sqrt(tube.cov[1, 1] * (5. / 7.)))
    else:
        densities = norm.pdf(points[:, None], loc=s[None, :]**2, scale=np.sqrt(tube.cov[1, 1]))
    expected = .7 * (densities @ probabilities) + .3 * norm.pdf(points,
        loc=background.loc[1], scale=np.sqrt(background.cov[1, 1]))
    np.testing.assert_allclose(frozen.marginal_pdf(points[:, None], (1,)), expected, rtol=2e-14)
    joint_points = np.column_stack((points, points + .1))
    np.testing.assert_allclose(frozen.bivariate_pdf(joint_points[:, ::-1], (1, 0)),
                               frozen.pdf(joint_points), rtol=2e-14)


@pytest.mark.parametrize("kernel", ["gaussian", "student_t", "laplace"])
def test_finite_moments_use_same_coarse_law_as_density(kernel):
    law, tube, background = mixture(kernel)
    frozen = law.discretize(8)
    s, probabilities = independent_nodes(8)
    centers = np.column_stack((s, s**2))
    mean = .7 * (probabilities @ centers) + .3 * background.loc
    second = .7 * (tube.cov + np.einsum("i,ij,ik->jk", probabilities, centers, centers))
    second += .3 * (background.cov + np.outer(background.loc, background.loc))
    np.testing.assert_allclose(frozen.mean(), mean, rtol=2e-14)
    np.testing.assert_allclose(frozen.cov(), second - np.outer(mean, mean), atol=2e-14)
    assert abs(frozen.mean()[0] - (.7 * beta.mean(.3, 2.) + .3 * background.loc[0])) > .03
    np.testing.assert_array_equal(frozen.mean(n_grid=2048), frozen.mean(n_grid=8))


def test_gaussian_joint_cdf_and_quantiles_match_independent_finite_mixture():
    law, tube, background = mixture(covariance=np.diag([.04, .09]))
    frozen = law.discretize(8)
    s, probabilities = independent_nodes(8)
    centers = np.column_stack((s, s**2))
    point = np.array([.25, .4])
    expected = .7 * (np.prod(norm.cdf((point - centers) / np.sqrt(np.diag(tube.cov))), axis=1) @ probabilities)
    expected += .3 * np.prod(norm.cdf((point - background.loc) / np.sqrt(np.diag(background.cov))))
    assert frozen.cdf(point) == pytest.approx(expected, abs=2e-14)
    values = np.array([-.3, .1, .8])
    def cdf(x):
        return .7 * (norm.cdf((np.asarray(x)[..., None] - s**2) / .3) @ probabilities) + .3 * norm.cdf(x,
            loc=background.loc[1], scale=np.sqrt(background.cov[1, 1]))
    np.testing.assert_allclose(frozen.marginal_cdf(values, dim=1), cdf(values), atol=2e-14)
    assert frozen.marginal_cdf(.1, dim=1) == pytest.approx(cdf(.1), abs=2e-14)
    q = np.array([.01, .5, .99])
    quantiles = frozen.ppf(q, dim=1, xtol=1e-11)
    np.testing.assert_allclose(cdf(quantiles), q, atol=2e-12)
    np.testing.assert_array_equal(frozen.quantile(q, dim=1), frozen.ppf(q, dim=1))


def replay_noise(rng, n, covariance, kernel):
    if kernel == "gaussian":
        return rng.multivariate_normal(np.zeros(2), covariance, size=n)
    if kernel == "student_t":
        normal = rng.multivariate_normal(np.zeros(2), covariance * (5. / 7.), size=n)
        return normal / np.sqrt(rng.chisquare(7., size=n)[:, None] / 7.)
    return rng.laplace(scale=1 / np.sqrt(2), size=(n, 2)) @ np.linalg.cholesky(covariance).T


@pytest.mark.parametrize("kernel", ["gaussian", "student_t", "laplace"])
def test_seeded_sampler_replays_component_node_and_kernel_draws(kernel):
    law, tube, background = mixture(kernel)
    frozen = law.discretize(8)
    actual, info = frozen.rvs(43, random_state=20261020, return_info=True)
    rng = np.random.default_rng(20261020)
    labels = rng.choice(2, size=43, p=[.7, .3])
    s, probabilities = independent_nodes(8)
    expected = np.empty((43, 2))
    coordinates = np.full(43, np.nan)
    rows = np.flatnonzero(labels == 0)
    nodes = rng.choice(8, size=rows.size, p=probabilities)
    coordinates[rows] = s[nodes]
    expected[rows] = np.column_stack((s[nodes], s[nodes]**2)) + replay_noise(rng, rows.size, tube.cov, kernel)
    rows = np.flatnonzero(labels == 1)
    expected[rows] = rng.multivariate_normal(background.loc, background.cov, size=rows.size)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(info["s"], coordinates)
    np.testing.assert_array_equal(info["component_index"], labels)
    np.testing.assert_array_equal(info["component"], np.where(labels == 0, "branch", "background"))
    np.testing.assert_array_equal(info["kind"], np.where(labels == 0, "tube", "gaussian"))


@pytest.mark.parametrize("size,shape", [(1, (1,)), (0, (0,)), ((2, 3), (2, 3)), ((2, 0, 3), (2, 0, 3))])
def test_sample_shapes_and_empty_draws(size, shape):
    frozen = mixture()[0].discretize(8)
    samples, info = frozen.rvs(size, random_state=11, return_info=True)
    assert samples.shape == shape + (2,)
    assert all(value.shape == shape for value in info.values())


def test_zero_probability_nodes_are_never_sampled():
    class HalfInterval:
        def pdf(self, s):
            return np.where(s > .5, 2., 0.)
    law, _, _ = mixture(coordinate=HalfInterval())
    frozen = law.discretize(8)
    _, info = frozen.rvs(1000, random_state=12, return_info=True)
    s = info["s"][info["kind"] == "tube"]
    assert s.size and np.all(s > .5)
    np.testing.assert_allclose(frozen.components[0]._quadrature(999)[1], [0., 0., 0., 0., .25, .25, .25, .25])


def test_snapshot_is_independent_of_source_parameters_curve_and_density():
    class Coordinate:
        exponent = 1
        def pdf(self, s):
            return s**self.exponent
    coordinate = Coordinate()
    law, tube, background = mixture(coordinate=coordinate)
    shift = np.zeros(2)
    tube.curve = lambda s: np.column_stack((s, s**2)) + shift
    tube.metadata["nested"] = [1]
    frozen = law.discretize(8)
    x = np.array([[.1, .2], [-.3, .8]])
    before = (frozen.logpdf(x), frozen.mean(), frozen.cov(), frozen.rvs(23, random_state=15))
    shift[:] = 10.
    coordinate.exponent = 9
    tube.cov[:] *= 3.
    tube.metadata["nested"].append(2)
    tube.kernel = "laplace"
    background.loc[:] = 10.
    background.cov[:] *= 4.
    law.weights[:] = [.1, .9]
    after = (frozen.logpdf(x), frozen.mean(), frozen.cov(), frozen.rvs(23, random_state=15))
    for old, new in zip(before, after):
        np.testing.assert_array_equal(old, new)
    assert frozen.components[0].metadata["nested"] == [1]


def test_discretization_does_not_advance_or_share_source_random_generator():
    law, _, _ = mixture()
    saved_rng = deepcopy(law.random_state)
    frozen = law.discretize(8)
    np.testing.assert_array_equal(law.rvs(19), law.rvs(19, random_state=saved_rng))
    source_state = deepcopy(law.random_state)
    frozen.rvs(101)
    np.testing.assert_array_equal(law.rvs(19), law.rvs(19, random_state=source_state))
    finite_state = deepcopy(frozen.random_state)
    law.rvs(23)
    np.testing.assert_array_equal(frozen.rvs(23), frozen.rvs(23, random_state=finite_state))


@pytest.mark.parametrize("kernel", ["student_t", "laplace"])
def test_unsupported_non_gaussian_cdf_stays_explicit(kernel):
    frozen = mixture(kernel)[0].discretize(8)
    with pytest.raises(NotImplementedError):
        frozen.marginal_cdf(.1)
    with pytest.raises(NotImplementedError):
        frozen.cdf([.1, .2])
    with pytest.raises(NotImplementedError):
        frozen.ppf(.5)


@pytest.mark.parametrize("value", [True, np.bool_(True), 8.5, "8", np.nan])
def test_non_integer_grid_sizes_are_rejected(value):
    with pytest.raises(TypeError):
        mixture()[0].discretize(value)


def test_small_grid_and_rediscretization_are_rejected():
    law = mixture()[0]
    with pytest.raises(ValueError, match="at least 8"):
        law.discretize(7)
    finite = law.discretize(np.int64(8))
    with pytest.raises(ValueError, match="original continuous"):
        finite.discretize(16)


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_invalid_coordinate_mass_is_rejected_before_sampling(value):
    class Invalid:
        def pdf(self, s):
            return np.full_like(s, value)
    with pytest.raises(ValueError, match="density|quadrature"):
        mixture(coordinate=Invalid())[0].discretize(8)


@pytest.mark.parametrize("malformed", ["shape", "nan"])
def test_invalid_node_centers_are_rejected(malformed):
    law, tube, _ = mixture()
    tube.curve = lambda s: np.zeros((len(s), 3)) if malformed == "shape" else np.full((len(s), 2), np.nan)
    with pytest.raises(ValueError, match="finite centers"):
        law.discretize(8)


def test_custom_components_are_not_silently_reinterpreted():
    class CustomGaussian(GaussianComponent):
        pass
    custom = CustomGaussian("custom", 1., "gaussian", loc=[0., 0.], cov=np.eye(2))
    with pytest.raises(TypeError, match="Unsupported component"):
        SFD([custom]).discretize(8)


@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize("covariance", [
    [[1., .2], [.1, 1.]], [[1., 0.], [0., 0.]], [[1., 2.], [2., 1.]],
    [[np.nan, 0.], [0., 1.]], [[np.inf, 0.], [0., 1.]]])
def test_invalid_covariances_cannot_define_a_finite_probability_kernel(component, covariance):
    law = mixture()[0]
    law.components[component].cov = np.asarray(covariance)
    with pytest.raises(ValueError, match="covariance"):
        law.discretize(8)


@pytest.mark.parametrize("df", [2., np.nan, np.inf])
def test_invalid_student_t_degrees_of_freedom_are_rejected(df):
    law, tube, _ = mixture("student_t")
    tube.kernel_df = df
    with pytest.raises(ValueError, match="kernel_df"):
        law.discretize(8)


@pytest.mark.parametrize("weights", [[1.], [.1, .2], [1., -1.], [np.nan, 1.]])
def test_source_weights_must_still_be_a_probability_vector(weights):
    law = mixture()[0]
    law.weights = np.asarray(weights)
    with pytest.raises(ValueError, match="probability vector"):
        law.discretize(8)


def test_invalid_kernel_and_nonfinite_gaussian_location_are_rejected():
    law, tube, background = mixture()
    tube.kernel = "unknown"
    with pytest.raises(ValueError, match="kernel"):
        law.discretize(8)
    tube.kernel = "gaussian"
    background.loc[0] = np.nan
    with pytest.raises(ValueError, match="finite vectors"):
        law.discretize(8)


def test_gaussian_only_distribution_and_source_weights_are_preserved():
    gaussian = GaussianComponent("g", 1., "gaussian", loc=[.1, .2], cov=np.eye(2))
    law = SFD([gaussian])
    frozen = law.discretize(8)
    np.testing.assert_array_equal(law.rvs(19, random_state=13), frozen.rvs(19, random_state=13))
    np.testing.assert_array_equal(law.mean(), frozen.mean())
    np.testing.assert_array_equal(law.cov(), frozen.cov())


def test_zero_outer_mass_preserves_labels_without_generating_that_component():
    law, _, background = mixture()
    law.weights[:] = [0., 1.]
    frozen = law.discretize(8)
    samples, info = frozen.rvs(23, random_state=13, return_info=True)
    assert frozen.component_names == ["branch", "background"]
    assert np.all(info["component"] == "background") and np.isnan(info["s"]).all()
    with np.errstate(divide="ignore"):
        np.testing.assert_allclose(frozen.logpdf(samples[:3]),
            multivariate_normal.logpdf(samples[:3], background.loc, background.cov), atol=2e-14)


@pytest.mark.parametrize("kernel", ["gaussian", "student_t", "laplace"])
def test_legacy_continuous_draws_remain_exactly_unchanged(kernel):
    path = Path(__file__).resolve().parent / "fixtures/core_before_finite_grid.py"
    spec = importlib.util.spec_from_file_location("flowdistribution._legacy_finite_grid_test", path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    law, tube, background = mixture(kernel)
    old_tube = legacy.TubeComponent(name="branch", weight=.7, kind="tube", curve=tube.curve,
        cov=tube.cov, s_dist=tube.s_dist, kernel=kernel, kernel_df=7.)
    old_background = legacy.GaussianComponent("background", .3, "gaussian", loc=background.loc, cov=background.cov)
    old = legacy.SFD([old_tube, old_background])
    law.discretize(8)
    actual, info = law.rvs(37, random_state=31, return_info=True)
    expected, old_info = old.rvs(37, random_state=31, return_info=True)
    np.testing.assert_array_equal(actual, expected)
    for key in info:
        np.testing.assert_array_equal(info[key], old_info[key])
    for method in ("logpdf", "mean", "cov"):
        args = (actual[:3],) if method == "logpdf" else ()
        np.testing.assert_array_equal(getattr(law, method)(*args, n_grid=8),
                                      getattr(old, method)(*args, n_grid=8))
