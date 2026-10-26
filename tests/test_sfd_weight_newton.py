import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from flowdistribution.occupancy import fit_kernel_weights
from flowdistribution.weight_newton import fit_kernel_weights_newton


def test_same_objective_as_converged_weight_em():
    logs = np.random.default_rng(13).normal(size=(90, 7))
    reference = np.full(7, 1. / 7)
    em = fit_kernel_weights(logs, reference, pseudocount=.5, max_iter=20000,
                           objective_tol=1e-12, kkt_tol=1e-8)
    newton = fit_kernel_weights_newton(logs, reference, kkt_tol=1e-8)
    assert em.converged and newton.converged
    np.testing.assert_allclose(em.weights, newton.weights, atol=1e-7)
    assert newton.trace[-1, 0] == pytest.approx(em.trace[-1, 0], abs=1e-12)
    assert np.all(np.diff(newton.trace[:, 0]) >= -1e-12)


def test_duplicate_kernels_prior_preserves_internal_ratio():
    logs = np.random.default_rng(17).normal(size=(100, 2))[:, [0, 0, 1]]
    fit = fit_kernel_weights_newton(logs, [.1, .2, .7])
    assert fit.converged
    assert fit.weights[0] / fit.weights[1] == pytest.approx(.5, abs=1e-8)


def test_binary_solution_agrees_with_independent_scalar_optimization():
    logs = np.random.default_rng(23).normal(size=(100, 2))
    kernels = np.exp(logs)
    reference = np.array([.3, .7])
    objective = lambda p: -(np.log(kernels @ np.array([p, 1 - p])).sum()
                            + .5 * (reference @ np.log([p, 1 - p])))
    solution = minimize_scalar(objective, bounds=(1e-9, 1 - 1e-9), method="bounded",
                               options={"xatol": 1e-12})
    fit = fit_kernel_weights_newton(logs, reference)
    assert fit.converged
    assert fit.weights[0] == pytest.approx(solution.x, abs=1e-7)


def test_limited_budget_keeps_nonconvergence_visible():
    fit = fit_kernel_weights_newton(np.random.default_rng(1).normal(size=(100, 6)),
                                     np.full(6, 1/6), max_iter=1)
    assert not fit.converged
    assert np.all(fit.weights > 0) and np.isclose(fit.weights.sum(), 1.)


def test_initialization_does_not_change_reference_prior():
    logs = np.random.default_rng(29).normal(size=(110, 5))
    reference = np.full(5, .2)
    a = fit_kernel_weights_newton(logs, reference)
    b = fit_kernel_weights_newton(logs, reference, initial=[.05, .4, .3, .2, .05])
    assert a.converged and b.converged
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-7)


@pytest.mark.parametrize("options", [{"pseudocount": 0}, {"max_iter": 0}, {"kkt_tol": np.nan}])
def test_invalid_controls(options):
    with pytest.raises(ValueError):
        fit_kernel_weights_newton(np.zeros((10, 2)), [.5, .5], **options)
