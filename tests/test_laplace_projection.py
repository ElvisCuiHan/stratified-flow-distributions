from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import laplace, norm

from flowdistribution import GaussianComponent, SFD, TubeComponent
from flowdistribution.core import _kernel_logpdf, _kernel_noise
from flowdistribution.laplace import LaplaceProjection, _log_absolute_integral


def integrate_piecewise(function, breaks):
    ends = np.r_[-np.inf, np.unique(breaks), np.inf]
    return sum(quad(function, a, b, epsabs=1e-11, epsrel=1e-10)[0]
               for a, b in zip(ends[:-1], ends[1:]))


def summed_two_laplace(x, a, b):
    x = np.abs(np.asarray(x))
    if a == b:
        return np.exp(-x / a) * (1 + x / a) / (4 * a)
    return (a * np.exp(-x / a) - b * np.exp(-x / b)) / (2 * (a**2 - b**2))


def correlated_tube(kernel="laplace"):
    return TubeComponent(name="branch", kind="tube", weight=1.,
        curve=lambda s: np.column_stack((s, s**2)), cov=np.array([[1., .8], [.8, 1.]]),
        kernel=kernel)


@pytest.mark.parametrize("rho", [.8, -.8])
def test_univariate_projection_is_sum_not_subcovariance_laplace(rho):
    covariance = np.array([[1., rho], [rho, 1.]])
    points = np.array([-4., -.5, 0., .2, 2.])
    scales = np.abs(np.linalg.cholesky(covariance)[1]) / np.sqrt(2)
    actual = np.exp(LaplaceProjection(covariance, (1,)).logpdf(points[:, None]))
    expected = summed_two_laplace(points, *scales)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert actual[2] == pytest.approx(.5050762722761087)
    assert abs(actual[2] - laplace.pdf(0, scale=1 / np.sqrt(2))) > .2


@pytest.mark.parametrize("perturbation", [0., 1e-9])
def test_equal_and_near_equal_phase_rates(perturbation):
    loading = np.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 1. + perturbation]])
    points = np.array([0., .3, 2., 8.])
    value = np.exp(LaplaceProjection(loading @ loading.T, (2,)).logpdf(points[:, None]))
    b = 1 / np.sqrt(2)
    t = points / b
    expected = np.exp(-t) * (t**2 + 3 * t + 3) / (16 * b)
    np.testing.assert_allclose(value, expected, rtol=2e-8)


def test_phase_log_density_does_not_underflow_in_far_tail():
    covariance = np.array([[1., .8], [.8, 1.]])
    value = LaplaceProjection(covariance, (1,)).logpdf(np.array([[1000.], [-1000.]]))
    assert np.isfinite(value).all()
    np.testing.assert_array_equal(value[:1], value[1:])
    assert value[0] < -1000


def test_projected_density_has_unit_mass_and_correct_variance():
    loading = np.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 1.]])
    projection = LaplaceProjection(loading @ loading.T, (2,))
    density = lambda x: np.exp(projection.logpdf(np.array([[x]]))[0])
    assert quad(density, -np.inf, np.inf, epsabs=1e-9)[0] == pytest.approx(1., abs=1e-9)
    assert quad(lambda x: x**2 * density(x), -np.inf, np.inf, epsabs=1e-8)[0] == pytest.approx(3., abs=1e-8)


@pytest.mark.parametrize("value", [0., .2, -1.4])
def test_univariate_density_matches_direct_joint_integration(value):
    covariance = np.array([[1., .8], [.8, 1.]])
    density = lambda x: np.exp(_kernel_logpdf(np.array([[x, value]]), mean=np.zeros(2),
        cov=covariance, kernel="laplace", kernel_df=5)[0])
    direct = integrate_piecewise(density, [0., value / .8])
    actual = np.exp(LaplaceProjection(covariance, (1,)).logpdf(np.array([[value]]))[0])
    assert actual == pytest.approx(direct, rel=1e-10)


def test_absolute_value_integral_matches_direct_quad_with_tied_breaks():
    a = np.array([.3, -.6, 0.])
    b = np.array([1., -2., 0.])
    density = lambda t: np.exp(-np.sqrt(2) * (abs(t) + np.abs(a - b * t).sum()))
    direct = integrate_piecewise(density, [0., .3])
    assert np.exp(_log_absolute_integral(a, b)) == pytest.approx(direct, rel=1e-12)


@pytest.mark.parametrize("kernel", ["gaussian", "student_t", "laplace"])
def test_full_coordinate_permutation_keeps_original_joint_law(kernel):
    model = correlated_tube(kernel)
    points = np.array([[.4, -.8], [0., 0.], [2., 1.]])
    np.testing.assert_allclose(model.marginal_logpdf(points[:, ::-1], (1, 0), 8),
                               model.logpdf(points, 8), rtol=1e-13, atol=1e-13)


def test_two_coordinate_projection_matches_joint_integral():
    loading = np.array([[1., 0., 0.], [.3, .9, 0.], [.7, -.4, .8]])
    covariance = loading @ loading.T
    inverse = np.linalg.inv(loading)
    points = np.array([[.2, -.1], [1., .5]])
    projection = LaplaceProjection(covariance, (0, 2))
    actual = np.exp(projection.logpdf(points))
    for point, result in zip(points, actual):
        fixed = np.array([point[0], 0., point[1]])
        intercept, slope = inverse @ fixed, inverse[:, 1]
        roots = -intercept[slope != 0] / slope[slope != 0]
        def density(t):
            x = fixed.copy()
            x[1] = t
            return np.exp(_kernel_logpdf(x[None, :], mean=np.zeros(3), cov=covariance,
                                        kernel="laplace", kernel_df=5)[0])
        direct = integrate_piecewise(density, roots)
        assert result == pytest.approx(direct, rel=1e-10)
    reordered = LaplaceProjection(covariance, (2, 0)).logpdf(points[:, ::-1])
    np.testing.assert_allclose(np.log(actual), reordered, atol=1e-12)


def general_loading():
    return np.array([[1., 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1]])


def test_general_projection_quad_matches_independent_convolution():
    loading = general_loading()
    projection = LaplaceProjection(loading @ loading.T, (2, 3))
    points = np.array([[.2, -.1], [1., .5]])
    actual = np.exp(projection.logpdf(points))
    b = 1 / np.sqrt(2)
    for (x, y), result in zip(points, actual):
        density = lambda t: laplace.pdf(t, scale=b) * laplace.pdf(x - t, scale=b) * summed_two_laplace(y - t, b, b)
        expected = integrate_piecewise(density, [0., x, y])
        assert result == pytest.approx(expected, rel=2e-8)
    assert projection.evaluations > 0
    reordered = LaplaceProjection(loading @ loading.T, (3, 2)).logpdf(points[:, ::-1])
    np.testing.assert_allclose(np.log(actual), reordered, rtol=1e-8, atol=1e-8)


def test_general_projection_budget_failure_is_explicit():
    loading = general_loading()
    projection = LaplaceProjection(loading @ loading.T, (2, 3), max_evaluations=1)
    with pytest.raises(ArithmeticError, match="budget exceeded"):
        projection.logpdf(np.array([[.2, -.1]]))


def test_independent_rows_keep_diagonal_factory_marginals():
    covariance = np.diag([.3, .4, .8])
    points = np.array([[.2, -.1], [1., .5]])
    actual = np.exp(LaplaceProjection(covariance, (2, 0)).logpdf(points))
    expected = laplace.pdf(points[:, 0], scale=np.sqrt(.8 / 2)) * laplace.pdf(points[:, 1], scale=np.sqrt(.3 / 2))
    np.testing.assert_allclose(actual, expected, rtol=1e-13)


def test_disjoint_loading_rows_factor_into_independent_convolutions():
    loading = np.array([[1., 0, 0, 0], [.6, .8, 0, 0], [0, 0, 1, 0], [0, 0, .6, .8]])
    projection = LaplaceProjection(loading @ loading.T, (1, 3))
    points = np.array([[.2, -.1], [1., .5]])
    a, b = .6 / np.sqrt(2), .8 / np.sqrt(2)
    expected = summed_two_laplace(points[:, 0], a, b) * summed_two_laplace(points[:, 1], a, b)
    np.testing.assert_allclose(np.exp(projection.logpdf(points)), expected, rtol=1e-12)
    assert projection.independent
    assert projection.evaluations == 0


def test_tube_and_outer_mixture_use_correct_projected_kernels():
    model = correlated_tube()
    points = np.array([-.3, 0., 1.2])
    s, mass = model._quadrature(8)
    a, b = np.abs(np.linalg.cholesky(model.cov)[1]) / np.sqrt(2)
    expected = sum(w * summed_two_laplace(points - u**2, a, b) for u, w in zip(s, mass))
    np.testing.assert_allclose(np.exp(model.marginal_logpdf(points[:, None], (1,), 8)), expected, rtol=1e-12)
    background = GaussianComponent("background", .3, "gaussian", loc=[0., 0.], cov=np.eye(2))
    law = SFD([replace(model, weight=.7), background])
    np.testing.assert_allclose(law.marginal_pdf(points[:, None], (1,), 8),
                               .7 * expected + .3 * norm.pdf(points), rtol=1e-12)


def test_sampling_event_matches_correct_correlated_marginal():
    covariance = np.array([[1., .8], [.8, 1.]])
    samples = _kernel_noise(np.random.default_rng(1901), size=20000, cov=covariance,
                            kernel="laplace", kernel_df=5)
    a, b = np.abs(np.linalg.cholesky(covariance)[1]) / np.sqrt(2)
    probability = quad(lambda t: summed_two_laplace(t, a, b), -.5, .5, epsabs=1e-11)[0]
    assert np.mean(np.abs(samples[:, 1]) <= .5) == pytest.approx(probability, abs=.012)


@pytest.mark.parametrize("kernel", ["gaussian", "student_t", "laplace"])
def test_joint_kernels_and_sampling_match_legacy_source_exactly(kernel):
    path = Path(__file__).resolve().parent / "fixtures/core_before_projection.py"
    spec = importlib.util.spec_from_file_location("legacy_sfd_core_projection_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    options = dict(cov=np.array([[1., .8], [.8, 1.]]), kernel=kernel, kernel_df=5)
    points = np.array([[.4, -.8], [0., 0.], [2., 1.]])
    np.testing.assert_array_equal(_kernel_logpdf(points, mean=np.zeros(2), **options),
                                  module._kernel_logpdf(points, mean=np.zeros(2), **options))
    np.testing.assert_array_equal(_kernel_noise(np.random.default_rng(71), size=37, **options),
                                  module._kernel_noise(np.random.default_rng(71), size=37, **options))
