# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from typer.testing import CliRunner

from frontier_data.catalog import index_dataset, inspect_catalog
from frontier_data.cli import app
from frontier_data.dataset import Dataset, FieldInfo, Manifest, SampleEntry, SampleMeta


def fixture(root):
    root.mkdir()
    manifest = Manifest(
        name="CFD with metadata",
        samples=[
            SampleEntry(id="case-1", path="samples/one", sample_kind="grid"),
            SampleEntry(id="case-2", path="samples/two", sample_kind="grid"),
        ],
        splits={"train": ["case-1"], "test": ["case-2"]},
    )
    (root / "manifest.json").write_text(manifest.model_dump_json())
    for entry in manifest.samples:
        path = root / entry.path
        path.mkdir(parents=True)
        meta = SampleMeta(
            id=entry.id,
            dataset="CFD",
            mesh={"n_points": 0, "n_faces": 0, "bounds": [0, 1, 0, 1, 0, 1], "area": 0},
            fields=[
                FieldInfo(name="pressure", units="Pa", location="cell"),
                FieldInfo(name="pressure", units="Pa", location="point"),
                FieldInfo(name="flow_rate", units="m^3/s", location="global"),
            ],
        )
        (path / "meta.json").write_text(meta.model_dump_json())
    return root


def test_catalog_customer_commands_preserve_scientific_metadata(tmp_path, monkeypatch):
    root = fixture(tmp_path / "data")

    def forbidden(*args):
        raise AssertionError("Catalog must not load numerical arrays")

    monkeypatch.setattr(Dataset, "sample", forbidden)
    target = tmp_path / "catalog.sqlite"
    runner = CliRunner()
    built = runner.invoke(app, ["catalog-build", str(root), str(target)])
    assert built.exit_code == 0, built.output
    receipt = json.loads(built.output)
    # Browsing stays independent of the original dataset and all its numerical files.
    root.rename(tmp_path / "moved-away")
    result = runner.invoke(app, ["catalog-inspect", str(target), "--sample", "case-2", "--verify"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["revision"] == receipt["revision"] == data["verified_revision"]
    assert data["samples"][0]["splits"] == ["test"]
    assert [
        (f["name"], f["location"], f["units"]) for f in data["samples"][0]["metadata"]["fields"]
    ] == [("pressure", "cell", "Pa"), ("pressure", "point", "Pa"), ("flow_rate", "global", "m^3/s")]
    assert inspect_catalog(target, limit=1)["next_ordinal"] == 1
    assert inspect_catalog(target, start=1)["next_ordinal"] is None
    missing = runner.invoke(app, ["catalog-inspect", str(target), "--sample", "missing"])
    assert missing.exit_code != 0
    assert "Invalid value" in missing.output


@pytest.mark.parametrize("escape", ["../outside", "/outside", "C:\\outside"])
def test_catalog_rejects_escaped_source_paths(tmp_path, escape):
    root = fixture(tmp_path / "data")
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["samples"][0]["path"] = escape
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaves"):
        index_dataset(root, tmp_path / "catalog")
    assert not (tmp_path / "catalog").exists()


def test_catalog_rejects_meta_symlink_outside_dataset(tmp_path):
    root = fixture(tmp_path / "data")
    path = root / "samples/one/meta.json"
    outside = tmp_path / "outside"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        index_dataset(root, tmp_path / "catalog")


def test_catalog_detects_manifest_change_during_conversion(tmp_path, monkeypatch):
    import frontier_data.catalog as module

    root = fixture(tmp_path / "data")
    original = module.write_catalog

    def change(path, records, **kwargs):
        (root / "manifest.json").write_text("{}")
        return original(path, records, **kwargs)

    monkeypatch.setattr(module, "write_catalog", change)
    with pytest.raises(ValueError, match="changed"):
        index_dataset(root, tmp_path / "catalog")
    assert not (tmp_path / "catalog").exists()


def test_catalog_rejects_orphan_split_and_mismatched_metadata_id(tmp_path):
    root = fixture(tmp_path / "data")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["train"].append("missing")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unknown sample"):
        index_dataset(root, tmp_path / "catalog")
    manifest["splits"]["train"].remove("missing")
    manifest_path.write_text(json.dumps(manifest))
    meta_path = root / "samples/one/meta.json"
    meta = json.loads(meta_path.read_text())
    meta["id"] = "unrelated"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="metadata id"):
        index_dataset(root, tmp_path / "catalog")
    assert not (tmp_path / "catalog").exists()
