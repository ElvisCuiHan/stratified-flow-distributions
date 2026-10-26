# Pre-Repeat Control Addendum

Frozen on 2026-10-18 after single-split workflow checks and before the formal
25-split experiment. The base protocol is `docs/46_validation_protocol_20261018.md`.
This addendum creates `pbmc_inductive_marker_holdout_v2`; the geometry protocol
is unchanged. Preserve the earlier smoke outputs and their source snapshots.

Add one prespecified marker-prediction control: PCs plus KDE log density plus
the first seven GMM-8 posterior component probabilities. Dropping the eighth
avoids an exactly redundant simplex coordinate. This tests whether a benefit
of the SFD summary is specific to the tube representation or can also be
obtained from generic cluster-membership features. It is a stronger control
than GMM posterior entropy alone. Keep all four original readouts and add
this fifth readout; use the same training/validation cells, standardization,
penalty grid, selection rule, and test metrics. Do not select a favorable
subset of controls or replace the marker target after comparing results.

No earlier single-split numerical result enters a manuscript performance
claim. Full repeated results are evaluated under this amended protocol.
