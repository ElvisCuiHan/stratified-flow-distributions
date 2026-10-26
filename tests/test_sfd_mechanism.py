import inspect

import numpy as np
from scipy.integrate import quad
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import norm, truncnorm

from scripts.run_sfd_mechanism_experiment import (
    PARALLEL_SD, TAU, ArmLaw, coordinate_mean, density_plot_summary, fit_sfd, longitudinal,
    objective, truth,
)


def test_longitudinal_convolution_matches_quadrature():
    for m in (0.25, 0.5, 0.7):
        for u in (-0.1, 0.2, 0.55, 1.05):
            numerical = quad(
                lambda s: truncnorm.pdf(s, -m / TAU, (1 - m) / TAU, loc=m, scale=TAU)
                * norm.pdf(u, s, PARALLEL_SD), 0, 1, epsabs=1e-13, epsrel=1e-10,
                points=[min(max(u, 0), 1)],
            )[0]
            analytical = np.exp(longitudinal(np.array([u]), m)[0][0])
            np.testing.assert_allclose(analytical, numerical, rtol=1e-8, atol=1e-12)


def test_longitudinal_density_normalizes():
    for m in (0.1, 0.5, 0.9):
        integral = quad(lambda u: float(np.exp(longitudinal(np.array([u]), m)[0][0])),
                        -0.5, 1.5, epsabs=1e-10)[0]
        np.testing.assert_allclose(integral, 1, atol=1e-9)


def test_analytic_likelihood_gradient():
    x, _ = truth("width").sample(250, 102)
    parameters = np.array([0.5, 0.8, 0.44, 0.62, np.log(0.11), np.log(0.08)])
    _, gradient = objective(parameters, x)
    numerical = approx_derivative(lambda p: objective(p, x)[0], parameters, method="3-point")
    np.testing.assert_allclose(gradient, numerical.ravel(), atol=2e-6, rtol=2e-5)


def test_finite_tail_density_and_gradient():
    x = np.array([[-20, 0], [20, 10], [0, -20], [0, 0], [0.5, 0.3]])
    parameters = np.array([0.5, 0.8, 0.44, 0.62, np.log(0.11), np.log(0.08)])
    value, gradient = objective(parameters, x)
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))


def test_each_alternative_changes_one_parameter_block():
    base = truth("reference")
    for group, expected in (("mass", (True, False, False)),
                            ("position", (False, True, False)),
                            ("width", (False, False, True))):
        law = truth(group)
        actual = tuple(not np.array_equal(a, b) for a, b in
                       zip((base.weights, base.locations, base.widths),
                           (law.weights, law.locations, law.widths)))
        assert actual == expected
    np.testing.assert_allclose(coordinate_mean(0.5), 0.5, atol=1e-14)
    assert 0.68 < coordinate_mean(0.7) < 0.7


def test_sampling_is_deterministic_and_train_test_independent():
    law = truth("mass")
    x, labels = law.sample(100, 12)
    y, labels_y = law.sample(100, 12)
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(labels, labels_y)
    assert not np.array_equal(x, law.sample(100, 13)[0])
    np.testing.assert_allclose(law.responsibilities(x).sum(axis=1), 1, atol=1e-14)


def test_joint_fit_recovers_known_fixture_without_labels():
    law = ArmLaw(np.array([0.55, 0.25, 0.2]), np.array([0.67, 0.45]), np.array([0.12, 0.06]))
    x, _ = law.sample(5000, 202601)
    fit, diagnostics = fit_sfd(x)
    np.testing.assert_allclose(fit.weights, law.weights, atol=0.025)
    np.testing.assert_allclose(fit.locations, law.locations, atol=0.025)
    np.testing.assert_allclose(fit.widths, law.widths, atol=0.015)
    assert next(d for d in diagnostics if d["selected"])["converged"]
    assert set(inspect.signature(fit_sfd).parameters) == {"x", "fixed_kernels"}


def test_fixed_kernel_fit_does_not_modify_kernels():
    law = truth("position")
    x, _ = law.sample(600, 204)
    fixed = truth("reference")
    fit, _ = fit_sfd(x, fixed_kernels=fixed)
    np.testing.assert_array_equal(fit.locations, fixed.locations)
    np.testing.assert_allclose(fit.widths, fixed.widths, atol=1e-16)


def test_observable_event_probability_matches_sampling():
    for group in ("reference", "mass", "position", "width"):
        law = truth(group)
        x, _ = law.sample(30000, 905)
        assert abs(np.mean(x[:, 1] > 0) - law.positive_y()) < 0.012


def test_prediction_plot_uses_all_replicates():
    rows = [{"scenario": "reference", "method": "Gaussian", "heldout_log_score": value,
             "fit_seconds": 0, "eval_seconds": 0, "status": "ok"} for value in (1.0, 3.0)]
    row = density_plot_summary(rows, "Gaussian")["reference"]
    assert row["mean_heldout_log_score"] == 2.0
    assert row["se_heldout_log_score"] == 1.0
    assert row["reps"] == 2
