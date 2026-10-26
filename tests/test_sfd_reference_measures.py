import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm


def test_double_cover_uses_preimage_average_under_own_pushforward():
    s = (np.arange(4000) + .5) / 4000
    h = 2 * s
    representative = np.mod(s, .5)
    own_density = (2 * representative + 2 * (representative + .5)) / 2
    wrong_sum = 2 * own_density
    assert own_density.mean() == pytest.approx(1.)
    assert wrong_sum.mean() == pytest.approx(2.)
    for test_function in (np.sin, np.cos):
        observed = test_function(4 * np.pi * s)
        assert (own_density @ observed) / len(s) == pytest.approx((h @ observed) / len(s), abs=1e-13)
    expected = quad(lambda t: 2 * t * np.sin(4 * np.pi * t), 0, 1, epsabs=1e-11)[0]
    assert (own_density @ np.sin(4 * np.pi * s)) / len(s) == pytest.approx(expected, abs=1e-7)


def test_identical_branch_references_require_common_density_factor():
    # Two copies of uniform coordinate measure give nu(dx)=2 dx on [0,1].
    first, second = .3, .7
    common_density = lambda x: (first * 2 * x + second * 2 * (1 - x)) / 2
    assert quad(lambda x: common_density(x) * 2, 0, 1)[0] == pytest.approx(1.)
    assert quad(lambda x: (first * 2 * x + second * 2 * (1 - x)) * 2, 0, 1)[0] == pytest.approx(2.)
    for x in (.1, .5, .9):
        correct_posterior = (first * x) / common_density(x)
        assert correct_posterior == pytest.approx(first * x / (first * x + second * (1 - x)))


def test_projected_atom_excludes_continuous_density_under_mixed_reference():
    # nu=lambda+delta_0; Gaussian RN contribution at the atom is zero.
    atom_mass, continuous_mass = .3, .7
    atom_likelihood, lebesgue_reference_factor = 1., 0.
    gaussian_likelihood = norm.pdf(0.) * lebesgue_reference_factor
    total = atom_mass * atom_likelihood + continuous_mass * gaussian_likelihood
    posterior = np.array([atom_mass * atom_likelihood, continuous_mass * gaussian_likelihood]) / total
    np.testing.assert_array_equal(posterior, [1., 0.])
    wrong_total = atom_mass / total + continuous_mass * norm.pdf(0.) / total
    assert wrong_total > 1.9
    for y in (-.5, .2):
        density = continuous_mass * norm.pdf(y)
        assert continuous_mass * norm.pdf(y) / density == pytest.approx(1.)


def test_continuous_density_uses_reference_factor_at_exact_atom():
    atom_mass, blob_mass, residual_mass = .2, .3, .5
    continuous = lambda x: blob_mass * norm.pdf(x) + residual_mass * norm.pdf(x, loc=1.)
    continuous_mass = quad(continuous, -np.inf, np.inf, epsabs=1e-11)[0]
    correct_atom_contribution = atom_mass
    assert continuous_mass + correct_atom_contribution == pytest.approx(1.)
    wrong_atom_contribution = atom_mass + continuous(0.)
    assert continuous_mass + wrong_atom_contribution > 1.24


def test_regularized_blob_occurs_once_in_density_decomposition():
    # With no Dirac components, nu=lambda and the atomic contribution is empty.
    blob_mass, residual_mass = .3, .7
    correct = quad(lambda x: blob_mass * norm.pdf(x) + residual_mass * norm.pdf(x, loc=1.),
                   -np.inf, np.inf)[0]
    duplicated = correct + quad(lambda x: blob_mass * norm.pdf(x), -np.inf, np.inf)[0]
    assert correct == pytest.approx(1.)
    assert duplicated == pytest.approx(1.3)
