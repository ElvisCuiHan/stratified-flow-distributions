# Author Preprint

This directory is the self-contained 26 October 2026 source snapshot,
including the 34-page PDF, thirteen figures, saved numerical evidence and
table extraction code. It preserves the definitions, related work, formal
results, assumptions and proofs. It is an author version, not a claim of
journal acceptance or a published repository DOI.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
python extract_uncertainty.py --out fresh_interval_tables
```

CJKutf8 and Arphic gkai fonts are needed for the Chinese model name.
The public figure-source and experiment implementations are in the parent
repository. Figure/source mapping is in `../docs/FIGURES.md`.

Author-owned text is CC BY 4.0 with the exclusions in
`MANUSCRIPT_LICENSE.txt` and `THIRD_PARTY_NOTICES.txt`. `LICENSE` here is
the upstream document-style license, not the root software license.
