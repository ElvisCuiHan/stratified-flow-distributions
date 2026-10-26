# Branch Mass, Position, and Width: Frozen Protocol

Protocol: `sfd_mechanism_v1`, fixed before the repeated experiment.

## Question

Can a likelihood fit distinguish a change in branch probability from a change
in longitudinal occupancy or observation noise when geometry is known?

This is a correctly specified, two-branch parametric recovery experiment.
It is not a test of learned geometry, general covariance fields, or biological
interpretation. The new joint fit is an experiment-local likelihood estimator,
separate from the earlier weight-only estimator.

## Law

Two unit-length arms share the origin, at angles +30 and -30 degrees.
For arm b, let t_b and n_b be orthonormal tangent and normal vectors.
Conditional on Z=b,

    S ~ Normal(m_b, 0.16^2), truncated to [0, 1]
    X = S t_b + epsilon_parallel t_b + epsilon_perpendicular n_b
    epsilon_parallel ~ Normal(0, 0.025^2)
    epsilon_perpendicular ~ Normal(0, sigma_b^2).

All draws are independent conditional on the component. The residual is
Normal((0.4, 0), 0.55^2 I_2). Its distribution and both arms are known to the
SFD estimators. The conditional Gaussian kernels have full-rank covariance.
The longitudinal convolution has a closed form, checked against quadrature.

Four independent groups share this geometry:

| Group | Upper/lower/residual mass | Upper/lower m | Upper/lower sigma |
| --- | --- | --- | --- |
| Reference | 0.4 / 0.4 / 0.2 | 0.5 / 0.5 | 0.07 / 0.07 |
| Mass | 0.6 / 0.2 / 0.2 | 0.5 / 0.5 | 0.07 / 0.07 |
| Position | 0.4 / 0.4 / 0.2 | 0.7 / 0.5 | 0.07 / 0.07 |
| Width | 0.4 / 0.4 / 0.2 | 0.5 / 0.5 | 0.14 / 0.07 |

The reported position is E(S | Z=upper), the normalized truncated-normal mean,
not its untruncated location parameter m. Contrasts subtract the independently
fitted reference group within each Monte Carlo replicate.

## Fits

- Oracle law: known generating density, with no fit.
- SFD joint: estimate two independent mass logits, two m parameters, and two
  log widths; hold arm geometry, longitudinal scale, parallel noise, and
  residual law fixed. Use SciPy L-BFGS-B with an analytic gradient.
- SFD fixed kernels: estimate weights with the reference m and widths frozen
  in every group. This is an explicit misspecification ablation.
- SFD oracle kernels: estimate weights with the true group-specific kernels.
  This isolates weight estimation and uses privileged kernel information.
- Gaussian: existing full-covariance fitter, regularization 1e-5.
- GMM-4 and GMM-8: existing EM fitter, regularization 1e-5, tolerance 1e-6,
  200 iterations, two starts; select the larger training likelihood.
- KDE: existing Scott-bandwidth fitter; use 1e-6 jitter only after a singular
  covariance error.

All SFD fits use the same three prespecified starts: weights (0.4,0.4,0.2),
m=(0.5,0.5), widths=(0.08,0.08); m=(0.35,0.65), widths=(0.12,0.12);
and m=(0.65,0.35), widths=(0.12,0.12). Weight-only starts retain their frozen
kernel values. Bounds: logits [-5,5], m [0.1,0.9], widths [0.025,0.3].
L-BFGS-B uses 300 iterations, ftol=1e-10, gtol=1e-6, maxls=40.
Select the finite fit with the largest training likelihood and report its
optimizer status, iterations, gradient, and boundary contact. No truth labels
or test observations enter fitting or start selection.

## Repetition and Extraction

- Main run: 25 replicates; 1000 training and 2000 independent test observations
  per group. No grouping or patient-level uncertainty is implied.
- Base seed 20261018; replicate r and zero-based group index g use
  base + 100000*r + 1000*g. Training and test RNGs use offsets 1 and 2;
  baseline fits use deterministic method-specific offsets.
- The smoke run uses a separate directory and smaller sample sizes. It checks
  code, not the paper's effect sizes or estimator ranking.
- Density metrics: test mean log score and error in Pr(X_2 > 0). The latter is
  an observable event probability, computed for every density method.
- SFD recovery: upper-branch mass, E(S | Z=upper), transverse width, component
  mass RMSE, posterior discrepancy from the true responsibility, latent-label
  Brier score, and classification accuracy. Truth labels are used only here.
- GMM/KDE components are not relabeled as true SFD branches. They receive the
  same density and observable-event metrics, not an arbitrary latent-mass score.
- Report replicate means, standard errors, bias/RMSE, failure and warning rates.
  Intervals in figures are mean +/- 1.96 Monte Carlo SE, not per-dataset
  parameter confidence intervals. Failed fits remain in status denominators.
- No nonfinite test scores are dropped. A nonfinite score is a failed run.

## Outputs and Checks

Run output: `outputs/sfd_mechanism_v1/`. Refuse to overwrite a run directory.
Write protocol/configuration and script hashes before generating data, with
per-group train/test hashes, score rows, fitted parameters, recovery rows,
within-replicate contrasts, summary tables, CSV figure inputs, and PDF/PNG
figures. Raw labels do not appear in estimator arguments.

Required tests: density normalization; convolution versus quadrature;
analytic versus finite-difference gradient; deterministic sampling and fitting;
known-parameter recovery on a fixed fixture; no test/truth inputs to fit;
correct zero/nonzero generating contrasts; and finite tail evaluation.
Record all starts and selected-start diagnostics. Use one BLAS/OpenMP thread.

Interpretation is limited to the declared law and contrasts. In particular,
an ablation's fixed position/width is not an attempted estimate of those
parameters, and recovery under known geometry is not evidence of recovery
after unknown support selection.
