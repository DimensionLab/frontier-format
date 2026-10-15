# SPDX-License-Identifier: Apache-2.0
"""Versions: baseline, structural sharing, diff, revert, lineage, gc, PROV export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from frontier_data import Dataset, DatasetWriter
from frontier_data.dataset import write_json
from frontier_data.versions import Version, VersionStore, export_prov_json


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    w = DatasetWriter(root, name="vtest", domain="cfd")
    rng = np.random.default_rng(0)
    for i in range(3):
        pts = rng.random((50, 2)).astype(np.float32)
        cells = rng.integers(0, 50, (80, 3)).astype(np.int32)
        w.write_mesh_sample(
            f"case_{i}",
            points=pts,
            cells=cells,
            fields={"p": rng.random((4, 50)).astype(np.float32)},
            times=np.arange(4) * 0.1,
            parameters={"Re": 100.0 * (i + 1)},
        )
    w.manifest.splits = {"train": ["case_0", "case_1"], "val": ["case_2"], "test": []}
    w.finalize()
    return root


def _scale_field(root: Path, sid: str, factor: float) -> None:
    s = Dataset(root).sample(sid)
    arr = s.array("p")
    data = np.asarray(arr[...]) * factor
    chunk_key_encoding = arr.metadata.to_dict()["chunk_key_encoding"]
    g: Any = zarr.open_group(str(s.store_path), mode="r+")
    shutil.rmtree(s.store_path / "fields" / "p")
    a = g["fields"].create_array(
        "p",
        shape=data.shape,
        dtype=data.dtype,
        chunks=arr.chunks,
        chunk_key_encoding=chunk_key_encoding,
    )
    a[...] = data
    a.attrs["dims"] = arr.attrs["dims"]


def test_init_commit_sharing(ds: Path) -> None:
    vs = VersionStore(ds)
    v0 = vs.init()
    assert vs.initialized and vs.head == v0.id and "baseline" in v0.tags
    assert vs.status().clean
    assert vs.init().id == v0.id  # idempotent
    assert vs.commit("nothing") is None
    _scale_field(ds, "case_1", 2.0)
    st = vs.status()
    assert st.modified and not st.added and not st.removed
    v1 = vs.commit("scale", actor={"kind": "human", "name": "t"})
    assert v1 is not None and v1.parent == v0.id
    # only the changed sample's subtree differs; the other two subtrees are shared
    assert v1.samples["case_0"] == v0.samples["case_0"]
    assert v1.samples["case_1"] != v0.samples["case_1"]
    assert v1.stats["n_new_bytes"] < v0.stats["n_bytes"] / 2
    # objects are hardlinked into the working tree: no duplicated bytes
    obj = vs.object_path(vs.flatten(v1)["manifest.json"].hash)
    assert obj.stat().st_ino == (ds / "manifest.json").stat().st_ino
    assert not (obj.stat().st_mode & 0o222)


def test_version_id_is_content_addressed(ds: Path, tmp_path: Path) -> None:
    v0 = VersionStore(ds).init()
    copy = tmp_path / "copy"
    shutil.copytree(ds, copy, ignore=shutil.ignore_patterns(".frontier"))
    assert VersionStore(copy).init().id == v0.id


def test_diff_revert_lineage_gc(ds: Path) -> None:
    vs = VersionStore(ds)
    v0 = vs.init()
    _scale_field(ds, "case_1", 3.0)
    m = json.loads((ds / "manifest.json").read_text())
    m["splits"] = {"train": ["case_0"], "val": ["case_1"], "test": ["case_2"]}
    write_json(ds / "manifest.json", m)  # atomic replace: versioned files are read-only
    v1 = vs.commit("scale + resplit")
    assert v1 is not None
    d = vs.diff(v0.id, v1.id)
    assert d["samples"]["modified"].keys() == {"case_1"}
    arr = d["samples"]["modified"]["case_1"]["arrays"]["mesh.zarr/fields/p"]
    assert arr["frac_changed"] > 0.9 and arr["rel_l2_diff"] == pytest.approx(2.0, rel=1e-5)
    assert arr["after"]["mean"] == pytest.approx(3 * arr["before"]["mean"], rel=1e-5)
    assert d["manifest"]["splits"]["val"] == {"added": ["case_1"], "removed": ["case_2"]}
    v2 = vs.revert(v0.id)
    assert v2.stats["content_id"] == v0.id and v2.parent == v1.id
    assert [v.id for v in vs.log()] == [v2.id, v1.id, v0.id]
    assert np.allclose(Dataset(ds).sample("case_1").field("p"), np.asarray(_orig(vs, v0)))
    lin = vs.lineage("samples/case_1")
    assert [x["version"] for x in lin] == [v0.id, v1.id, v2.id]
    assert vs.lineage("samples/case_0")[-1]["version"] == v0.id
    assert vs.resolve("HEAD~2") == v0.id and vs.resolve(v1.id[:8]) == v1.id
    # gc: nothing to drop while every version is kept
    assert vs.gc()["removed_objects"] == 0
    # checkout an old version, drop the newer version files, gc frees the scaled chunks
    vs.checkout(v0.id)
    for v in (v1, v2):
        (vs.versions_dir / f"{v.id}.json").unlink()
    assert vs.gc()["removed_objects"] > 0


def _orig(vs: VersionStore, v0: Version) -> np.ndarray:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        vs.materialize(v0.id, tmp, samples=["case_1"])
        return Dataset(tmp).sample("case_1").field("p")


def test_provenance_and_prov_export(ds: Path) -> None:
    vs = VersionStore(ds)
    v0 = vs.init()
    rid = vs.append_provenance(
        {"kind": "assertion", "version": v0.id, "check": "nan_inf", "status": "pass"}
    )
    vs.approve(v0.id, "alice", note="ok")
    assert vs.record(rid) is not None
    assert vs.assertions(v0.id)[0]["check"] == "nan_inf"
    assert vs.approvals(v0.id)[0]["approved_by"] == "alice"
    prov = export_prov_json(vs)
    assert f"frontier:version/{v0.id}" in prov["entity"]
    assert any(a["frontier:op"] == "prep.baseline" for a in prov["activity"].values())
    assert any(
        g["prov:entity"] == f"frontier:version/{v0.id}" for g in prov["wasGeneratedBy"].values()
    )
    assert "frontier:agent/human/alice" in prov["agent"]


def test_migrate_is_idempotent(ds: Path, tmp_path: Path) -> None:
    from frontier_data.versions import find_datasets, migrate

    root = tmp_path
    (root / "raw" / "junk").mkdir(parents=True)
    (root / "raw" / "junk" / "manifest.json").write_text("{}")
    assert find_datasets(root) == [ds]
    first = migrate(root)
    assert [r["action"] for r in first] == ["initialized"]
    assert VersionStore(ds).tags()["baseline"] == first[0]["version"]
    second = migrate(root)
    assert [r["action"] for r in second] == ["unchanged"]
    assert second[0]["version"] == first[0]["version"]
    write_json(ds / "manifest.json", {**json.loads((ds / "manifest.json").read_text()), "x": 1})
    assert migrate(root)[0]["action"] == "dirty"


def test_revert_to_head_content_is_a_noop_and_log_is_cycle_safe(ds: Path) -> None:
    vs = VersionStore(ds)
    v0 = vs.init()
    assert vs.revert(v0.id).id == v0.id and len(vs.all_versions()) == 1
    assert vs.revert("HEAD").id == v0.id
    # a hand-written self-parented version must not hang the reader
    bad = json.loads((vs.versions_dir / f"{v0.id}.json").read_text())
    bad["parent"] = v0.id
    (vs.versions_dir / f"{v0.id}.json").write_text(json.dumps(bad))
    assert [v.id for v in vs.log()] == [v0.id]


def test_clone_and_hardlink_copies_do_not_write_into_the_original(ds: Path, tmp_path: Path) -> None:
    import os

    from frontier_data.versions import clone

    vs = VersionStore(ds)
    v0 = vs.init()
    n0 = len(vs.provenance())
    c = clone(ds, tmp_path / "clone")
    cs = VersionStore(c)
    assert cs.head == v0.id and cs.status().clean
    _scale_field(c, "case_0", 2.0)
    v1 = cs.commit("scaled in the clone")
    assert v1 is not None
    cs.tag("train-ready")
    cs.append_provenance({"kind": "note", "note": "clone only"})
    assert VersionStore(ds).head == v0.id and VersionStore(ds).status().clean
    assert len(VersionStore(ds).provenance()) == n0 and "train-ready" not in VersionStore(ds).tags()
    # a plain hardlink copy (cp -al): appends and tags still stay out of the original
    raw = tmp_path / "cp-al"
    shutil.copytree(ds, raw, copy_function=os.link)
    rs = VersionStore(raw)
    rs.append_provenance({"kind": "note", "note": "copy only"})
    rs.tag("baseline", "HEAD")
    rs.tag("mine")
    assert len(VersionStore(ds).provenance()) == n0 and "mine" not in VersionStore(ds).tags()


def test_revert_moves_the_gate_tag_only_onto_identical_content(ds: Path) -> None:
    from typer.testing import CliRunner

    from frontier_data.cli_versions import versions_app

    vs = VersionStore(ds)
    v0 = vs.init()  # never gated
    _scale_field(ds, "case_1", 2.0)
    a = vs.commit("A, gated")
    assert a is not None
    vs.tag("train-ready", a.id)
    _scale_field(ds, "case_1", 3.0)
    b = vs.commit("B, not gated")
    assert b is not None and vs.gate_status() == {
        "head": b.id,
        "tagged": a.id,
        "train_ready": False,
    }

    # back to the gated content: the tag follows, with a provenance note naming the source
    r = vs.revert(a.id)
    assert r.id not in (a.id, b.id) and r.parent == b.id
    assert vs.tags()["train-ready"] == r.id == vs.head and "train-ready" in r.tags
    assert vs.gate_status()["train_ready"]
    notes = [n for n in vs.provenance("note") if n.get("tag") == "train-ready"]
    assert len(notes) == 1 and notes[0]["version"] == r.id and notes[0]["from_version"] == a.id
    assert a.id in notes[0]["note"] and "identical" in notes[0]["note"]

    # back to content the gate never judged: HEAD is untagged, the tag stays where it was
    u = vs.revert(v0.id)
    assert u.tags == [] and vs.tags()["train-ready"] == r.id != vs.head
    assert len([n for n in vs.provenance("note") if n.get("tag") == "train-ready"]) == 1

    # the CLI says so and how to gate; with --json the same facts are fields
    res = CliRunner().invoke(versions_app, ["revert", str(ds), b.id])
    assert res.exit_code == 0, res.output
    head = vs.head
    expected = (
        f"HEAD {head} is not tagged train-ready (the tag is on {r.id}); "
        f"run `python -m frontier_prep run prep.quality.gate -d {ds}`"
    )
    assert expected in "".join(res.output.split("\n"))
    res = CliRunner().invoke(versions_app, ["revert", str(ds), r.id, "--json"])
    out = json.loads(res.output)
    assert out["train_ready"] and out["tagged"] == out["head"] == vs.head and out["hint"] is None
    assert "train-ready" in out["tags"]

    # a HEAD left untagged by an older revert heals: same content as the tagged version
    vs.tag("train-ready", a.id)
    n_versions = len(vs.all_versions())
    healed = vs.revert(a.id)
    assert healed.id == vs.head == out["head"] and vs.tags()["train-ready"] == healed.id
    assert len(vs.all_versions()) == n_versions  # the no-op revert recorded no version


def test_revert_without_any_gate_tag_says_no_version_carries_it(ds: Path) -> None:
    from typer.testing import CliRunner

    from frontier_data.cli_versions import versions_app

    vs = VersionStore(ds)
    v0 = vs.init()
    _scale_field(ds, "case_0", 2.0)
    assert vs.commit("change") is not None
    res = CliRunner().invoke(versions_app, ["revert", str(ds), v0.id, "--json"])
    out = json.loads(res.output)
    assert not out["train_ready"] and out["tagged"] is None
    assert "no version carries the tag" in out["hint"] and "prep.quality.gate" in out["hint"]
    assert "train-ready" not in vs.tags()
