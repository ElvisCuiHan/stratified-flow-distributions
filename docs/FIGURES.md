# Figure-to-Evidence Index

Numbers refer to the 34-page 26 October 2026 preprint.
All archived figure PDFs are in `paper/figures/`.

| Figure | Subject | Reproduction input / implementation |
|--:|:--|:--|
| 1 | Elvis contours, tube density, draws, surface | `python reproduce.py elvis --out runs/elvis`; released `data/derived/elvis_contours.csv` |
| 2 | Volcano / circle-tube density | `scripts/replay_shape_examples.py`; `volcano_law` and `volcano_pdf` in `generate_sfd_shape_gallery.py` |
| 3 | Projected Fermat-quintic section | Same replay; `fermat_law`, `fermat_projection`; seed 2026102207 |
| 4 | Reference-sample histograms | Same replay; `data/derived/reference_histograms.csv`, three 500-row training subsets |
| 5 | Snowflake, Doraemon-inspired face, maze, ECG surfaces | `python scripts/generate_3d_density_plot.py`; output `figures/fig_3d_density_panel.pdf`; nonlinear relative density height |
| 6 | Actual weight-only fitted SFD, five scenarios | `python reproduce.py figures --out runs/figures`; `paper/evidence/known_skeleton/` |
| 7 | Mechanism separation | `Rscript scripts/plot_sfd_evidence.R runs/r-figures`; `evidence/sfd_mechanism_v1/contrasts.csv` |
| 8 | Geometry and noise stress | Same R command; `evidence/sfd_geometry_stress_v1/` |
| 9 | PBMC held-out density and marker proxy | Same R command; `evidence/pbmc_inductive_marker_holdout_v2/` |
| 10 | Face example in three observed coordinates | `python scripts/plot_sfd_face_example.py --inputs data/derived/face_3d --out runs/face`; archived generated sample |
| 11 | Geometry-interval coverage | `Rscript paper/plot_geometry_intervals.R paper/evidence/geometry_interval_summary.csv runs/interval-figures` |
| 12 | Classical-distribution controls | `python reproduce.py figures --out runs/figures`; `paper/evidence/classical/` |
| 13 | Copula controls | Same Python command; `paper/evidence/copula/` |

Figures 1-5 and 10 illustrate declared laws or samples, not fitted-method
evidence. Figures 12-13 include oracle/control baselines but no fitted SFD.
Rendering may differ with fonts, Matplotlib/R versions or PDF backends.
