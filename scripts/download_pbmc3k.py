#!/usr/bin/env python3
"""Explicitly fetch and verify the frozen public 10x PBMC3k input archive."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
DIGEST = "847d6ebd9a1ec9a768f2be7e40ca42cbfe75ebeb6d76a4c24167041699dc28b5"


def download(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents overwriting an existing download.
    with out.open("xb") as target:
        try:
            with urllib.request.urlopen(URL, timeout=120) as response:
                digest = hashlib.sha256()
                while block := response.read(1024 * 1024):
                    target.write(block)
                    digest.update(block)
            if digest.hexdigest() != DIGEST:
                raise ValueError("Archive differs from the frozen PBMC3k source")
        except BaseException:
            target.close()
            out.unlink()
            raise
    print("Verified PBMC3k archive:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    download(parser.parse_args().out)
