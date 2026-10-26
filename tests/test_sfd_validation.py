import inspect
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize._numdiff import approx_derivative

from scripts.run_pbmc3k_inductive_validation import (
    MARKER_SETS, fit_representation, logistic_predict, marker_targets, predictor_counts,
)
from scripts.run_sfd_geometry_stress import (
    ANGLES, GeometryLaw, fit_arm_model, generating_law, objective,
)
from scripts.run_sfd_mechanism_experiment import truth
from scripts.sfd_validation_common import fit_reference, logpdf, summary


def test_geometry_likelihood_equals_original_at_original_angles():
    law = truth("position")
    x, _ = law.sample(100, 88)
    new = GeometryLaw(law.weights, law.locations, law.widths, ANGLES)
    np.testing.assert_allclose(new.logpdf(x), law.logpdf(x), atol=1e-13)


def test_geometry_eight_parameter_gradient():
    x, _ = generating_law("overlap", "width").sample(120, 155)
    p = np.r_[0.3, 0.7, 0.42, 0.64, np.log([0.09, 0.1]), np.deg2rad([27, -21])]
    _, analytical = objective(p, x)
    numerical = approx_derivative(lambda v: objective(v, x)[0], p, method="3-point")
    np.testing.assert_allclose(analytical, numerical.ravel(), atol=2e-6, rtol=2e-5)


def test_rotated_and_heavy_sampling_keep_targets_and_seed():
    for condition in ("nominal", "overlap", "heavy_noise"):
        law = generating_law(condition, "width")
        x, y = law.sample(250, 225)
        np.testing.assert_array_equal(x, law.sample(250, 225)[0])
        np.testing.assert_allclose(law.responsibilities(x).sum(axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(law.weights, truth("width").weights)
        assert np.isfinite(law.logpdf(x)).all()
        assert set(y) == {0, 1, 2}


def test_angles_are_learned_without_true_labels():
    law = generating_law("overlap", "mass")
    x, _ = law.sample(1800, 31)
    fit, diagnostics = fit_arm_model(x)
    np.testing.assert_allclose(fit.angles, law.angles, atol=np.deg2rad(4))
    np.testing.assert_allclose(fit.weights, law.weights, atol=0.06)
    assert len(diagnostics) == 3
    assert sum(row["selected"] for row in diagnostics) == 1
    assert set(inspect.signature(fit_arm_model).parameters) == {"train", "angles"}


def test_markers_excluded_from_predictors_and_normalization():
    counts = sparse.csr_matrix([[4, 1, 2], [8, 2, 3], [3, 0, 1]])
    first, genes, libraries = predictor_counts(counts, ["gene1", "CD79A", "gene2"])
    changed = counts.toarray()
    changed[:, 1] += 10000
    second, _, _ = predictor_counts(sparse.csr_matrix(changed), ["gene1", "CD79A", "gene2"])
    np.testing.assert_array_equal(genes, [0, 2])
    np.testing.assert_array_equal(libraries, [6, 11, 4])
    np.testing.assert_array_equal(first.toarray(), second.toarray())


def test_test_cells_do_not_change_training_representation():
    rng = np.random.default_rng(18)
    matrix = sparse.csr_matrix(np.log1p(rng.poisson(3, size=(50, 30))))
    train = np.arange(35)
    first = fit_representation(matrix, train, n_features=10, seed=21)
    changed = matrix.toarray()
    changed[35:] *= 1000
    second = fit_representation(sparse.csr_matrix(changed), train, n_features=10, seed=21)
    np.testing.assert_array_equal(first.genes, second.genes)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.sd, second.sd)
    np.testing.assert_array_equal(first.pca.components_, second.pca.components_)
    np.testing.assert_allclose(first.transform(matrix)[train], second.transform(matrix)[train])


def test_reference_gmm_and_kde_keep_all_selection_records():
    train = np.random.default_rng(25).normal(size=(100, 2))
    for method in ("GMM-4", "KDE"):
        fit, starts = fit_reference(method, train, 45)
        assert len(starts) == 3
        assert sum(row["selected"] for row in starts) == 1
        assert np.isfinite(logpdf(fit, train)).all()
        expected = "training" if method == "GMM-4" else "inner_validation"
        assert all(row["selection_set"] == expected for row in starts)


def test_logistic_selection_does_not_receive_test_targets():
    rng = np.random.default_rng(37)
    x = rng.normal(size=(80, 3))
    y = (x[:, 0] > 0).astype(int)
    prediction, rows = logistic_predict(x, y, x[:7], 92)
    assert prediction.shape == (7,)
    assert len(rows) == 4
    assert sum(row["selected"] for row in rows) == 1
    assert set(inspect.signature(logistic_predict).parameters) == {"train_x", "train_y", "test_x", "seed"}
    _, changed_test_rows = logistic_predict(x, y, 100 * x[:7], 92)
    assert rows == changed_test_rows


def test_marker_target_uses_frozen_threshold_and_group_comparison():
    groups = list(MARKER_SETS)
    symbols = [MARKER_SETS[group][0] for group in groups]
    values = np.zeros((3, len(groups)))
    b = groups.index("B_marker_proxy")
    other = (b + 1) % len(groups)
    values[0, b] = 5
    values[1, b] = 5
    values[1, other] = 6
    labels, _ = marker_targets(sparse.csr_matrix(values), symbols, np.full(3, 10000))
    np.testing.assert_array_equal(labels, [1, 0, 0])


def test_nonconverged_gmm_starts_are_retained(monkeypatch):
    from scripts import sfd_validation_common as common

    class NonconvergedFit:
        converged_ = False
        n_iter_ = 2000

        def __init__(self, **kwargs):
            self.seed = kwargs["random_state"]

        def fit(self, train):
            return self

        def score(self, train):
            return float(self.seed)

    monkeypatch.setattr(common, "GaussianMixture", NonconvergedFit)
    fit, starts = common.fit_reference("GMM-4", np.zeros((20, 2)), 5)
    assert len(starts) == 3
    assert all(not row["converged"] for row in starts)
    assert starts[2]["selected"]
    assert fit.seed == 7


def test_validation_summary_uses_all_replicates():
    rows = [dict(method="SFD", error=value) for value in (1, 3)]
    row = summary(rows, ["method"], ["error"])[0]
    assert row["mean_error"] == 2
    assert row["se_error"] == 1


def test_location_and_gaussian_driver_do_not_reuse_latent_dimension():
    root = Path(__file__).resolve().parents[1]
    main = (root / "paper/main.tex").read_text()
    appendix = "\n".join((root / "paper" / name).read_text() for name in
                         ("supplement.tex", "theory_calculus.tex", "theory_tails.tex"))
    for source in (main, appendix):
        assert r"\Normal(m_b," not in source
        assert r"\mu_b^{\mathrm{loc}}" in source
    assert r"m_b(y" not in main
    assert r"g_{b,-J}(y\mid s)" in appendix
    assert r"G\sim\Normal(0,I_{r_b^{\varepsilon}})" in appendix


def test_paper_export_averages_all_25_replicates():
    from scripts.export_sfd_validation_results import mean_se
    values = np.arange(25, dtype=float)
    mean, se = mean_se(values)
    assert mean == 12
    np.testing.assert_allclose(se, values.std(ddof=1) / 5)


def test_paper_longtable_has_one_row_break_per_record(tmp_path):
    from scripts.export_sfd_validation_results import longtable
    path = tmp_path / "table.tex"
    longtable(path, "Caption", "tab:test", ["Method", "Value"], [["SFD", "1.0"]])
    text = path.read_text()
    assert r"\label{tab:test}" in text
    assert "SFD & 1.0 " + "\\" * 2 in text
    assert r"\end{longtable}" in text
