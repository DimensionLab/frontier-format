# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from frontier_data import ahmedml
from frontier_data.sources import huggingface as hf


def test_parse_runs() -> None:
    assert hf.parse_runs("1-4") == [1, 2, 3, 4]
    assert hf.parse_runs("1,3,5-7, 2") == [1, 2, 3, 5, 6, 7]
    with pytest.raises(ValueError):
        hf.parse_runs("5-1")


def test_run_files_and_forbidden() -> None:
    files = ahmedml.run_files(7)
    assert files["surface"] == "run_7/boundary_7.vtp"
    assert files["cell_area"] == "run_7/boundary_cell_area_7.npy"
    assert "volume" not in " ".join(files.values())
    assert ahmedml.is_forbidden("run_7/volume_7.vtu")
    assert not ahmedml.is_forbidden("run_7/boundary_7.vtp")
    with pytest.raises(hf.ForbiddenFileError):
        hf.download_file("run_7/volume_7.vtu", Path("/nonexistent"))


def test_provenance_store(tmp_path: Path) -> None:
    f = tmp_path / "run_1" / "x.csv"
    f.parent.mkdir()
    f.write_text("cd,cl\n1,2\n")
    store = hf.ProvenanceStore(tmp_path)
    rec = store.record("run_1/x.csv", f, "https://example/x.csv", None)
    assert rec.sha256 == hf.sha256_of(f) and rec.size == f.stat().st_size
    store.save()
    again = hf.ProvenanceStore(tmp_path).record("run_1/x.csv", f, "https://example/x.csv", None)
    assert again.sha256 == rec.sha256
    assert hf.resolve_url("neashton/ahmedml", "run_1/x.csv", None, "dataset") == (
        "https://huggingface.co/datasets/neashton/ahmedml/resolve/main/run_1/x.csv"
    )


def test_reference_conditions_consistent() -> None:
    c = ahmedml.CONDITIONS
    assert c["re_L"] == pytest.approx(c["u_inf"] * c["reference_length"] / c["nu"])
    assert c["re_H"] == pytest.approx(7.68e5, rel=1e-3)
