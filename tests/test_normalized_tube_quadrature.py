import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.stats import beta, multivariate_normal

from flowdistribution.core import TubeComponent
from flowdistribution.occupancy import fit_kernel_weights


def tube(s_dist):
    return TubeComponent(name="branch", weight=1., kind="tube",
                         curve=lambda s: np.column_stack((s, s * 0)),
                         cov=np.diag([.04, .09]), s_dist=s_dist)


def test_coarse_grid_is_a_probability_law():
    model = tube(beta(.3, 2.))
    s, mass = model._quadrature(8)
    raw_mass = beta(.3, 2.).pdf(s).mean()
    assert abs(raw_mass - 1.) > .1
    assert mass.sum() == pytest.approx(1.)
    x = np.linspace(-3, 4, 2001)
    p = np.exp(model.marginal_logpdf(x[:, None], (0,), n_grid=8))
    assert simpson(p, x=x) == pytest.approx(1., abs=1e-10)
    assert model.marginal_cdf(20., n_grid=8) == pytest.approx(1.)
    assert model.cdf([[20., 20.]], n_grid=8)[0] == pytest.approx(1.)
    assert model.mean(8)[0] == pytest.approx(mass @ s)
    assert model.second_moment(8)[0, 0] == pytest.approx(mass @ s**2 + .04)


def test_grid_density_equals_explicit_constrained_gaussian_mixture():
    model = tube(beta(.7, 3.))
    s, mass = model._quadrature(16)
    x = np.random.default_rng(1901).normal(size=(29, 2))
    expected = sum(w * multivariate_normal.pdf(x, [u, 0], model.cov)
                   for u, w in zip(s, mass))
    np.testing.assert_allclose(np.exp(model.logpdf(x, 16)), expected, rtol=1e-13)


@pytest.mark.parametrize("value", [-1., np.nan, np.inf, 0.])
def test_invalid_coordinate_density_rejected(value):
    class Invalid:
        def pdf(self, s):
            return np.full_like(s, value)
    with pytest.raises(ValueError, match="density|quadrature"):
        tube(Invalid()).logpdf([[0., 0.]], 8)


def test_zero_density_nodes_remain_zero():
    class HalfInterval:
        def pdf(self, s):
            return np.where(s > .5, 2., 0.)
    model = tube(HalfInterval())
    _, mass = model._quadrature(8)
    assert np.count_nonzero(mass) == 4
    assert np.isfinite(model.logpdf([[0., 0.]], 8)).all()


def test_occupancy_derivative_requires_grid_normalizer():
    s = (np.arange(8) + .5) / 8
    counts = np.arange(1., 9.)
    theta = -.8
    logits = theta * s
    probabilities = np.exp(logits) / np.exp(logits).sum()
    analytic = counts @ s - counts.sum() * (probabilities @ s)

    def objective(t):
        v = np.exp(t * s)
        return counts @ np.log(v / v.sum())

    numeric = (objective(theta + 1e-6) - objective(theta - 1e-6)) / 2e-6
    assert analytic == pytest.approx(numeric, abs=1e-7)
    assert abs(analytic - counts @ s) > 1.


def test_penalty_can_select_weights_without_likelihood_identification():
    x = np.linspace(-2, 2, 37)
    kernels = np.repeat((-.5 * x**2)[:, None], 2, axis=1)
    for ref in (np.array([.2, .8]), np.array([.7, .3])):
        fit = fit_kernel_weights(kernels, ref)
        np.testing.assert_allclose(fit.weights, ref, atol=1e-14)
        np.testing.assert_allclose(np.exp(kernels) @ fit.weights, np.exp(kernels[:, 0]))
