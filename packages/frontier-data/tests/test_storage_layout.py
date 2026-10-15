# SPDX-License-Identifier: Apache-2.0
"""Flat-key clones preserve encoded chunks, metadata and sparse fill semantics."""

import json

import numpy as np
import pytest
import zarr

from frontier_data import Dataset, DatasetWriter
from frontier_data.storage_layout import chunk_inventory, chunk_key, repack_dataset


def dataset(path):
    writer = DatasetWriter(path, "layout")
    values = np.arange(5 * 3 * 4 * 3, dtype=np.float32).reshape(5, 3, 4, 3, 1)
    writer.write_grid_sample(
        "a",
        fields={"u": values, "zero": values * 0},
        times=np.arange(5),
        spacing=[1] * 3,
        static={"body": np.zeros((3, 4, 3))},
    )
    writer.manifest.splits = {"train": ["a"], "val": [], "test": []}
    writer.finalize()
    root = zarr.open_group(str(path / "samples/a/grid.zarr"), mode="r+")
    root.create_array("partial", data=np.arange(35).reshape(5, 7), chunks=(2, 3))
    root.create_array("scalar", data=np.array(7.0), chunks=())
    return values


def test_flat_clone_is_lossless_independent_metadata_and_sparse_aware(tmp_path):
    source, out = tmp_path / "source", tmp_path / "flat"
    expected = dataset(source)
    (source / "provenance").mkdir()
    (source / "provenance/parent-manifest.json").write_text('{"ancestor": true}\n')
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    report = repack_dataset(source, out)
    assert report["implicit_fill_chunks"] > 0
    assert report["shared_compressed_bytes"] > 0
    assert before == {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    assert (out / "provenance/parent-manifest.json").read_bytes() == (
        source / "provenance/parent-manifest.json"
    ).read_bytes()
    ds = Dataset(out)
    np.testing.assert_array_equal(ds.sample("a").field("u"), expected)
    np.testing.assert_array_equal(ds.sample("a").field("zero"), expected * 0)
    assert ds.manifest.splits == Dataset(source).manifest.splits
    root = zarr.open_group(str(out / "samples/a/grid.zarr"), mode="r")
    partial, scalar = root["partial"], root["scalar"]
    assert isinstance(partial, zarr.Array) and isinstance(scalar, zarr.Array)
    np.testing.assert_array_equal(partial[:], np.arange(35).reshape(5, 7))
    assert scalar[()] == 7.0
    array = source / "samples/a/grid.zarr/fields/u"
    metadata = json.loads((array / "zarr.json").read_text())
    source_chunks, _ = chunk_inventory(array, metadata)
    old = source_chunks[(0, 0, 0, 0, 0)]
    new = out / "samples/a/grid.zarr/fields/u" / chunk_key((0, 0, 0, 0, 0), ".")
    assert old.stat().st_ino == new.stat().st_ino
    assert not new.stat().st_mode & 0o222
    assert not list(out.rglob("c/"))
    assert (source / "samples/a/meta.json").stat().st_ino != (
        out / "samples/a/meta.json"
    ).stat().st_ino
    assert len(ds.manifest.source["storage_layout"]["inventory_sha256"]) == 64
    inventory = json.loads((out / "storage-layout.json").read_text())["arrays"]
    assert sum(item["chunks"] for item in inventory) == report["chunks"]
    # A flat source can be cloned again without changing decoding or chunk identity.
    repack_dataset(out, tmp_path / "flat-again")
    np.testing.assert_array_equal(Dataset(tmp_path / "flat-again").sample("a").field("u"), expected)


def test_flat_clone_rejects_missing_chunks_in_strict_mode(tmp_path):
    source = tmp_path / "source"
    dataset(source)
    with pytest.raises(ValueError, match="missing chunks"):
        repack_dataset(source, tmp_path / "out", require_complete_chunks=True)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out-*"))


@pytest.mark.parametrize("kind", ["noncanonical", "out-of-bounds", "wrong-separator"])
def test_flat_clone_rejects_duplicate_noncanonical_and_wrong_separator_keys(tmp_path, kind):
    source = tmp_path / "source"
    dataset(source)
    array = source / "samples/a/grid.zarr/fields/u"
    metadata = json.loads((array / "zarr.json").read_text())
    separator = metadata["chunk_key_encoding"]["configuration"]["separator"]
    first = "00" if kind == "noncanonical" else "99" if kind == "out-of-bounds" else "0"
    if kind == "wrong-separator":
        separator = "/" if separator == "." else "."
    bad = array / separator.join(["c", first, "0", "0", "0", "0"])
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="chunk"):
        repack_dataset(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_flat_clone_rejects_symlinks_and_existing_targets(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    dataset(source)
    out.mkdir()
    with pytest.raises(ValueError, match="new directory"):
        repack_dataset(source, out)
    (source / "link").symlink_to(source / "manifest.json")
    with pytest.raises(ValueError, match="symlink"):
        repack_dataset(source, tmp_path / "other")


def test_chunk_inventory_rejects_unsupported_encodings(tmp_path):
    with pytest.raises(ValueError, match="default chunk keys"):
        chunk_inventory(tmp_path, {"zarr_format": 2})


def test_frontier_writes_flat_arrays_and_still_reads_legacy_nested_fields(tmp_path):
    writer = DatasetWriter(tmp_path, "flat-default")
    values = np.arange(72, dtype=np.float32).reshape(3, 2, 3, 4, 1)
    writer.write_grid_sample(
        "a",
        fields={"u": values},
        times=[0, 1, 2],
        spacing=[1, 1, 1],
        static={"body": np.zeros((2, 3, 4))},
    )
    writer.finalize()
    store = tmp_path / "samples/a/grid.zarr"
    for path in store.rglob("zarr.json"):
        metadata = json.loads(path.read_text())
        if metadata["node_type"] == "array":
            assert metadata["chunk_key_encoding"]["configuration"]["separator"] == "."
            assert not (path.parent / "c").is_dir()
    np.testing.assert_array_equal(Dataset(tmp_path).sample("a").times, [0, 1, 2])
    array = store / "fields/u"
    metadata = json.loads((array / "zarr.json").read_text())
    chunks, _ = chunk_inventory(array, metadata)
    for index, path in chunks.items():
        nested = array / chunk_key(index, "/")
        nested.parent.mkdir(parents=True, exist_ok=True)
        path.rename(nested)
    metadata["chunk_key_encoding"]["configuration"]["separator"] = "/"
    (array / "zarr.json").write_text(json.dumps(metadata))
    np.testing.assert_array_equal(Dataset(tmp_path).sample("a").field("u"), values)
