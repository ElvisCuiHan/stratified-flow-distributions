import numpy as np
import pytest

from flowdistribution.identification import target_information


def test_group_mass_identified_while_within_group_weights_are_not():
    kernels = np.array([[.8, .8, .2], [.2, .2, .8]])
    weights = np.array([.2, .3, .5])
    measure = kernels @ weights
    result = target_information(np.log(kernels), weights, [[1, 1, 0], [1, 0, 0]],
                                evaluation_weights=measure)
    assert result.rank == 1
    np.testing.assert_array_equal(result.in_numerical_row_space, [True, False])
    assert result.conditional_variance_factor[0] == pytest.approx(1 / 1.44)
    assert np.isnan(result.conditional_variance_factor[1])


def test_fully_aliased_targets_are_not_rescued_by_positive_weights():
    logs = np.zeros((11, 3))
    result = target_information(logs, [.2, .3, .5], [[1, 0, 0], [1, 1, 1]])
    assert result.rank == 0
    np.testing.assert_array_equal(result.in_numerical_row_space, [False, True])
    assert result.conditional_variance_factor[1] == 0.


def test_binary_disjoint_components_give_binomial_variance():
    logs = np.array([[0., -np.inf], [-np.inf, 0.]])
    result = target_information(logs, [.3, .7], [[1, 0]], evaluation_weights=[.3, .7])
    assert result.conditional_variance_factor[0] == pytest.approx(.3 * .7)


def test_n_less_than_parameter_dimension_retains_null_space():
    result = target_information(np.log([[.2, .4, .7, .9]]), [.25] * 4, np.eye(4))
    assert result.rank == 1
    assert result.dimension == 3
    assert not result.in_numerical_row_space.all()


def test_information_invariant_to_common_log_density_offset_and_target_constant():
    logs = np.random.default_rng(7).normal(size=(20, 3))
    a = target_information(logs, [.2, .3, .5], [[1, 1, 0]])
    b = target_information(logs + np.arange(20)[:, None], [.2, .3, .5], [[8, 8, 7]])
    np.testing.assert_allclose(a.conditional_variance_factor, b.conditional_variance_factor)


@pytest.mark.parametrize("weights", [[0., .5, .5], [.2, .3, .6], [.2, np.nan, .8]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        target_information(np.zeros((5, 3)), weights, [[1, 0, 0]])
