#!/usr/bin/env python3
"""Replay only the three released shape examples; no external image inputs."""
import argparse
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import generate_sfd_shape_gallery as gallery


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    gallery.CLASSICAL = Path(__file__).resolve().parents[1] / "data/derived/reference_histograms.csv"
    with PdfPages(out / "shape_examples.pdf") as book:
        gallery.draw_volcano(out, book)
        gallery.draw_fermat(out, book)
        gallery.draw_existing_histograms(out, book)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
