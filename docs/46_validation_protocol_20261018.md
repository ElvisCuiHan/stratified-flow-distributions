# Geometry Stress and Inductive PBMC Validation

Frozen before new results on 2026-10-18. Protocol IDs are
`sfd_geometry_stress_v1` and `pbmc_inductive_marker_holdout_v1`.
Existing experiment outputs and submission packages are not overwritten.

## Geometry Stress

Reuse the four group laws, locations, widths, weights, tangent noise, and
residual law from `docs/44_mechanism_experiment_protocol_20261018.md`.
There are three sampling conditions: Gaussian tubes at +/-30 degrees;
overlapping Gaussian tubes at +/-15 degrees; and +/-30-degree arms with
Student-t transverse noise (5 degrees of freedom, scaled to the same SD).
The Student-t law changes noise shape, not the target transverse SD.

Fit three Gaussian-tube SFDs: true supplied angles; supplied angles rotated
by +5 degrees; and angles estimated jointly from training observations.
The first two are sensitivity diagnostics with known geometry information;
the third does not receive true angles. The origin, two-arm family, tangent
scale, coordinate scale, and residual law remain supplied in every SFD fit.
Labels are used only for scoring; upper/lower orientation defines matching.

Joint fits use six mass/location/log-width parameters and, when active,
two angles bounded to [5,75] and [-75,-5] degrees. Three starts use
angles +/-20, +/-35, +/-50 degrees, locations (0.5,0.5), (0.35,0.65),
(0.65,0.35), and widths (0.08,0.08), (0.12,0.12), (0.12,0.12).
L-BFGS-B uses the original six-parameter bounds, maxiter 500, ftol 1e-10,
gtol 1e-6, maxls 40. Retain all starts and the best finite training objective;
record selected convergence and boundary contacts without discarding runs.

Density baselines are a Gaussian, full-covariance GMM-4/GMM-8, and KDE.
GMM uses scikit-learn 1.4.2 with three separately seeded k-means starts,
maxiter 2000, tol 1e-5, and covariance regularization 1e-5. Select by actual
training log score and retain warnings. KDE chooses 0.5, 1, or 2 times Scott's
bandwidth on a deterministic 80/20 inner split, then refits on all training
rows. No hyperparameter is chosen on the outer test set.

Full design: 25 repetitions, 1000 training/2000 test rows per group;
seed = 20261019 + 1000000*rep + 10000*condition_index + 1000*group_index;
training/test offsets 1/2. Smoke runs use explicitly smaller CLI sizes and
are workflow evidence only. Full runs require confirmation after timing.

Primary estimands are the within-repetition upper-arm mass, normalized
coordinate mean, and transverse-SD contrasts against the reference group.
Report bias, RMSE, Monte Carlo SE, latent-label Brier score/accuracy for
SFD only, and shared held-out log scores. Student-t oracle responsibilities
are available from the generating law. Predictive scores and latent recovery
have separate interpretations; generic mixture components are not relabeled
to branches. Produce row-level CSVs, start logs, seeds, hashes, configuration,
runtime, summary tables, and PDF/PNG plots after numerical verification.

## Inductive PBMC

Use only the cached 10x PBMC3k tarball. Do not download external data.
Exclude every gene in the existing predeclared PBMC marker sets from predictor
selection and predictor library-size normalization. Retain cells with nonzero
non-marker library size; this per-cell rule is fixed rather than learned.

Split cell IDs 80/20 before any learned preprocessing. Within each training
set, retain genes detected in at least 3 training cells, select the 1000
largest log-CP10k variances (gene index breaks ties), fit training means/SDs,
clip standardized values to [-10,10], and fit randomized two-component PCA.
Transform test cells with the saved training parameters. Save train/test IDs,
selected gene IDs, means, SDs, PCA components, seeds, and raw-file hash.

Fit an MST-skeleton tube plus Gaussian residual using the existing estimator
with 24 landmarks, 12 quadrature nodes, min_sigma 1e-3, sigma_scale 1,
covariance regularization 1e-5, min_weight 1e-4, maxiter 200, mass-change
tolerance 1e-6. Its historical width update is a heuristic, not an exact
likelihood-maximizing M-step; expose convergence under that stopping rule.
Density references use Gaussian, GMM-2/4/8 and selected-bandwidth KDE with
the baseline settings above and the identical train/test representation.
Do not claim an inductive density for the full gene-count space: scores
refer to the two-dimensional training-fitted representation.

The secondary task predicts a B-marker proxy from genes held out of all
predictor preprocessing. Per group, use the mean log-CP10k over available
marker genes, normalized by the non-marker library size. Label positive if
the B-marker score is at least 0.5 and strictly exceeds every other marker
group score. This is a feature-held-out internal proxy, not external cell-type
truth or independent biological validation.

Compare logistic predictors using PCs alone; PCs plus KDE log density;
PCs plus KDE density and GMM-8 posterior entropy; and PCs plus KDE density
and SFD tube log odds. Each uses training-fitted standardization. Select
L2 inverse penalty from [0.01,0.1,1,10] on a stratified 75/25 inner training
split by binary log loss; refit all outer training cells. The same inner
split is used across readouts. Report test log loss (primary), AUROC and
balanced accuracy (secondary), plus paired differences against the
PC-plus-density reference. No results-based proxy/threshold replacement.

Full design: 25 splits, seeds 20261020 + 10000*rep, rep=0..24; model seeds
use fixed method offsets. Split-to-split SE describes this cell collection,
not variability between people. Full repeated runs need confirmation after
the smoke timing; external or donor-level validation remains a separate gate.
