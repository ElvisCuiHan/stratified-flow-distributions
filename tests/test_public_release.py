import csv
from pathlib import Path
import re

import numpy as np
from pypdf import PdfReader
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_readme_local_links_and_images_exist():
    text = (ROOT / "README.md").read_text()
    paths = re.findall(r"\]\(([^)]+)\)|src=\"([^\"]+)\"", text)
    for pair in paths:
        target = next(x for x in pair if x)
        if not target.startswith(("http:", "https:", "#")):
            assert (ROOT / target.split("#")[0]).is_file(), target
    assert "docs/assets/elvis.png" in text
    assert "docs/assets/doraemon-and-friends.png" in text


def test_checked_paper_has_34_pages_and_no_private_paths():
    reader = PdfReader(ROOT / "paper/sfd_preprint.pdf")
    assert len(reader.pages) == 34
    text = "\n".join(page.extract_text() for page in reader.pages)
    for forbidden in ("/Users/", "/var/folders/", "/private/", "SHA256", "sha256"):
        assert forbidden not in text


def test_reference_histogram_rows_are_training_only():
    with (ROOT / "data/derived/reference_histograms.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1500
    assert {r["split"] for r in rows} == {"train"}
    assert {r["rep"] for r in rows} == {"1"}
    for scenario in {r["scenario"] for r in rows}:
        assert sum(r["scenario"] == scenario for r in rows) == 500


def test_download_failure_does_not_leave_partial_archive(tmp_path, monkeypatch):
    from scripts import download_pbmc3k
    def unavailable(*args, **kwargs):
        raise OSError("simulated unavailable source")
    monkeypatch.setattr(download_pbmc3k.urllib.request, "urlopen", unavailable)
    output = tmp_path / "counts.tar.gz"
    with pytest.raises(OSError):
        download_pbmc3k.download(output)
    assert not output.exists()


def test_download_refuses_to_overwrite_existing_file(tmp_path):
    from scripts.download_pbmc3k import download
    output = tmp_path / "counts.tar.gz"
    output.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        download(output)
    assert output.read_bytes() == b"preserved"


def test_readme_api_example():
    from flowdistribution import cartoon_face
    law = cartoon_face()
    x = law.rvs(40, random_state=42)
    assert np.isfinite(law.logpdf(x, n_grid=64)).all()
    np.testing.assert_allclose(law.responsibilities(x, n_grid=64).sum(axis=1), 1)
    finite = law.discretize(n_grid=64)
    assert np.isfinite(finite.logpdf(finite.rvs(10, random_state=42))).all()
