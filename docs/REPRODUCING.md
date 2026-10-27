# Reproducing the Paper

The release accompanies the 27 October 2026, 34-page author preprint.
It contains executable software, archived numeric evidence, and document
sources. It is not a claim that every historical fit has been rerun for
this release.

## Environment

Use Python 3.10 or newer; the numerical reference environment is Python
3.12.4, NumPy 1.26.4, SciPy 1.13.1, and scikit-learn 1.4.2.

```bash
conda env create -f environment.yml
conda activate sfd-reproduce
python -m pip install -e '.[reproduce,test]'
python -m pytest -q
```

The pip-only installation is given in the root README. Set
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before formal
comparisons. Floating-point optimizers, BLAS implementations, fonts and
package versions can affect numerical and visual agreement.

## No-Fit Evidence Replay

```bash
python reproduce.py verify
python reproduce.py figures --out runs/figures
```

Verification checks unique run keys, the full score-row counts, recomputed
known-skeleton log-score means and Monte Carlo standard errors, and interval
endpoints, validity, coverage flags, counts and summaries. The interval
recount uses 9,450 geometry records and 525 fixed-geometry records.
Classical/copula controls retain all recorded warning/failure rows;
their row-count checks do not certify optimizer convergence.

The Python result export regenerates the pastel known-skeleton figure,
the classical/copula controls, and checked interval tables. No new fitting,
scoring, method selection, or observations are involved.

The three main mechanism/geometry/PBMC result figures use R:

```r
install.packages(c("ggplot2", "patchwork", "ragg", "digest", "jsonlite"))
```

```bash
Rscript scripts/plot_sfd_evidence.R runs/r-figures
```

The public R adapter reads the released evidence directly and omits the
older MD/ECG development plots, which are not figures in this paper.
Uncertainty bars in these three figures are mean +/- 1.96 Monte Carlo SE;
the pastel known-skeleton figure uses +/- one Monte Carlo SE. Neither is
a per-fit parameter confidence interval. PBMC split variability is not
independent-subject uncertainty.

## Refit the Estimators

```bash
python reproduce.py smoke --out runs/smoke
python reproduce.py fit known --out runs/known
python reproduce.py fit mechanism --out runs/mechanism
python reproduce.py fit geometry --out runs/geometry
python reproduce.py fit classical --out runs/classical
python reproduce.py fit copula --out runs/copula
```

Every output directory must be new. The smoke run uses only 80 training and
80 test samples, one ECG-like scenario and one repetition; it is a workflow
test, not a paper result. Full profiles preserve the released scripts'
formal seeds and defaults. They can require hours. Formal result directories
contain parameters, selection records, failures and timing logs; keep them
even when a fit does not converge. The copula runner records optional vine
dependency feasibility; it does not fit a vine-copula comparator or silently
add one to the recorded comparisons.

| Study | Repeats | Training / test | What is learned |
|:--|--:|:--|:--|
| Known skeleton, five shapes | 25 | 1,000 / 1,000 | Component weights only |
| Four two-arm mechanism groups | 25 | 1,000 / 2,000 | Mass, position, width; fixed/oracle controls |
| Three geometry/noise conditions | 25 | 1,000 / 2,000 | Supplied, rotated or learned angles |
| PBMC3k | 25 overlapping splits | Training-fitted preprocessing | MST SFD and marker-proxy readouts |
| Classical / copula controls | 5 | 1,000 / 1,000 | No fitted SFD in these controls |

The release supports rebuilding interval tables from saved per-repetition
estimates and standard errors. It does **not** provide a one-command new-fit
Hessian/sandwich interval audit: the original interval audit bound itself to
an internal source snapshot and exact runtime. New fits require a new
interval audit rather than relabeling the archived intervals.

## Public PBMC3k Data

Input is the 10x Genomics **PBMCs from a Healthy Donor (3k), Cell Ranger 1.1.0**
filtered gene-barcode matrix. [Provider dataset page](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0).
Raw counts are not bundled. The helper downloads only after explicit invocation:

```bash
python scripts/download_pbmc3k.py --out data/pbmc3k.tar.gz
python reproduce.py fit pbmc --raw data/pbmc3k.tar.gz --out runs/pbmc
```

The downloader verifies the frozen source archive and refuses to overwrite
files. Check the provider's applicable data terms. Marker genes are excluded
from predictor selection and library-size normalization, and preprocessing
and model selection use training data. The held-out marker is a proxy label,
not clinical ground truth.

## Illustration Replay

```bash
python reproduce.py elvis --out runs/elvis
python scripts/replay_shape_examples.py --out runs/shapes
```

The Elvis command starts with released selected contours, applies the
length-proportional component masses and isotropic Gaussian tube SD 0.014,
then generates 18,000 draws with seed 2026102205. Density display uses
binned midpoint quadrature and Gaussian convolution. It needs no photo.
The original photo-to-contour extraction is not fully reproducible without
the reference image and annotation; that stage and its rights are not claimed
to be cleared by releasing derived geometry.

The shape command redraws the volcano, projected Fermat-quintic section and
reference-sample histograms from the released laws and selected input rows.
The Fermat section is an illustrative projection, not a learned Calabi-Yau
manifold or a claim about Calabi-Yau metrics. Other figure recipes appear in
the [figure index](FIGURES.md).

## Compile the Paper

```bash
python reproduce.py paper
```

Requires `latexmk`, pdfLaTeX, `CJKutf8`, and Arphic gkai fonts for the upright
Chinese name. The output is `paper/build/main.pdf`. The checked PDF and all
figure PDFs are already included. Rebuilding the paper uses archived figures,
not newly refitted results. See `paper/README.md` for source-license details.

## Public-Export Adaptations

Core library and scientific fitting scripts are copied unchanged. The export
adds an installation manifest, explicit orchestration, derived illustration
inputs and tests for public-file integrity. Historical numerical-regression
snapshots are kept under `tests/fixtures/` instead of internal output folders.
The notation test reads the current modular preprint sources. Two internal
archive-preservation checks, which only compare old draft documents to stored
digests, are not exported; all mathematical reference-measure tests remain.
The R plotting adapter changes input locations and defaults to the three
current-paper figures. Earlier document-compilation packages alone do not
contain this full software environment.
