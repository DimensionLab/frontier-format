# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from typer.testing import CliRunner

from frontier_data.cli import app
from frontier_data.dataset import Dataset, FieldInfo, Manifest, SampleEntry


def catalog(root, count=45):
    root.mkdir()
    manifest = Manifest(
        name="Metadata-only test catalog",
        minor=1,
        sample_kind="grid",
        fields=[FieldInfo(name="pressure", units="Pa")],
    )
    manifest.samples = [
        SampleEntry(id=f"case-{i:04d}", path=f"samples/case-{i:04d}", sample_kind="grid")
        for i in range(count)
    ]
    (root / "manifest.json").write_text(manifest.model_dump_json())
    return root


def test_default_inspection_uses_catalog_only_and_pages(tmp_path, monkeypatch):
    path = catalog(tmp_path / "data")

    def forbidden(*args):
        raise AssertionError("Inspection must not scan payloads or load sample arrays")

    monkeypatch.setattr(Dataset, "size_bytes", forbidden)
    monkeypatch.setattr(Dataset, "sample", forbidden)
    runner = CliRunner()
    first = runner.invoke(app, ["inspect", str(path), "--json"])
    assert first.exit_code == 0, first.output
    metadata = json.loads(first.output)
    assert metadata["total_samples"] == 45
    assert len(metadata["samples"]) == 20
    assert metadata["page"]["next_offset"] == 20
    assert metadata["storage_bytes"] is None and not metadata["storage_measured"]
    assert metadata["fields"][0]["units"] == "Pa"
    page = runner.invoke(app, ["inspect", str(path), "--json", "--offset", "40", "--limit", "10"])
    metadata = json.loads(page.output)
    assert [s["id"] for s in metadata["samples"]] == [f"case-{i:04d}" for i in range(40, 45)]
    assert metadata["page"]["next_offset"] is None
    empty = runner.invoke(app, ["inspect", str(path), "--json", "--offset", "100"])
    assert json.loads(empty.output)["samples"] == []
    human = runner.invoke(app, ["inspect", str(path), "--limit", "2"])
    assert human.exit_code == 0, human.output
    assert "case-0000" in human.output and "case-0002" not in human.output
    assert "--offset 2 --limit 2" in human.output


def test_storage_inventory_requires_explicit_option(tmp_path, monkeypatch):
    path = catalog(tmp_path / "data", 1)
    calls = []

    def measure(self):
        calls.append(self.root)
        return 123456

    monkeypatch.setattr(Dataset, "size_bytes", measure)
    result = CliRunner().invoke(app, ["inspect", str(path), "--measure-storage", "--json"])
    assert result.exit_code == 0, result.output
    assert calls == [path]
    assert json.loads(result.output)["storage_bytes"] == 123456
    assert json.loads(result.output)["storage_measured"] is True


@pytest.mark.parametrize(
    "options",
    [
        ["--limit", "0"],
        ["--limit", "1001"],
        ["--offset", "-1"],
        ["--sample", "case-0000", "--offset", "1"],
        ["--sample", "missing"],
    ],
)
def test_bad_inspection_requests_fail_cleanly(tmp_path, options):
    path = catalog(tmp_path / "data", 1)
    result = CliRunner().invoke(app, ["inspect", str(path), *options])
    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_selected_sample_isolated_and_quality_path_is_not_followed(tmp_path):
    import numpy as np

    from frontier_data.dataset import DatasetWriter

    root = tmp_path / "actual"
    writer = DatasetWriter(root, "pressure fields")
    for name in ("first", "second"):
        writer.write_grid_sample(
            name,
            fields={"p": np.ones((2, 2, 2, 1), dtype=np.float32)},
            spacing=[1, 1, 1],
            field_units={"p": "Pa"},
        )
    writer.manifest.quality = "/a/path/that/must/not/be/opened.json"
    writer.finalize()
    result = CliRunner().invoke(app, ["inspect", str(root), "--sample", "second", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [entry["id"] for entry in data["samples"]] == ["second"]
    assert data["sample_metadata"]["fields"][0]["units"] == "Pa"
    assert data["page"]["next_offset"] is None
    human = CliRunner().invoke(app, ["inspect", str(root)])
    assert human.exit_code == 0, human.output
    assert "quality report:" in human.output
