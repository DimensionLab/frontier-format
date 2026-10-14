# SPDX-License-Identifier: Apache-2.0
"""Stored LOD levels follow level 0 through every op: names, units, values, geometry, the
readers and the manifest stay consistent at every level, and level 0 stays bitwise."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frontier_data import Dataset, DatasetWriter
from frontier_data.mesh import SurfaceMesh
from frontier_data.validate import validate_dataset
from frontier_data.versions import VersionStore
from frontier_prep.base import DatasetRef
from frontier_prep.benchmarks import RHO, _box, _forces, _meta, make_case
from frontier_prep.io import SampleEditor
from frontier_prep.op import Context, Selection, get_op
from frontier_prep.ops.mesh import LOD_TOLERANCE, lod_target, rebuild_lods

LODS = {"levels": [1, 2], "targets": {"1": 200, "2": 100}}
LOD_KEYS = {f"lod_{k}" for k in LOD_TOLERANCE}


def _run(root: Path, op: str, params: dict[str, Any] | None = None, workers: int = 1) -> Any:
    return get_op(op).run(DatasetRef(root), params or {}, ctx=Context(workers=workers))


def _with_lods(root: Path, case: str = "naming") -> Path:
    make_case(case, root)
    r = _run(root, "prep.mesh.decimate", LODS)
    assert all(a.status == "pass" for a in r.assertions)
    assert r.determinism == "tolerance" and r.tolerance == LOD_TOLERANCE
    return root


def _levels(root: Path, sid: str) -> list[int]:
    return SampleEditor(root / "samples" / sid, sid).lod_levels_on_disk()


def _weighted_mean(root: Path, sid: str, name: str, lod: int) -> float:
    """Area-weighted mean of component 0 of a cell field at a level."""
    s = Dataset(root).sample(sid)
    g: Any = s.group if lod == 0 else s.group["lod"][str(lod)]
    area = np.asarray(g["cell_area"][...], dtype=np.float64)
    v = s.field(name, location="cell", lod=lod).astype(np.float64)
    return float((v[:, 0] * area).sum() / area.sum())


def _assert_consistent(root: Path, expect_levels: list[int] | None = None) -> None:
    """Every stored level describes level 0: same field names at each location, array
    lengths that match the level's own geometry, levels listed the same everywhere."""
    ds = Dataset(root)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = {e["id"]: e for e in manifest["samples"]}
    for sid in ds.sample_ids:
        s = ds.sample(sid)
        levels = _levels(root, sid)
        if expect_levels is not None:
            assert levels == expect_levels, (sid, levels)
        present = [0, *levels]
        assert [int(x) for x in s.attrs["lod_levels"]] == present
        assert s.meta.lod_levels == present
        assert entries[sid]["lod_levels"] == present
        for lvl in levels:
            g: Any = s.group["lod"][str(lvl)]
            n_points, n_faces = int(g["points"].shape[0]), int(g["faces"].shape[0])
            assert int(g["cell_area"].shape[0]) == n_faces
            assert int(np.asarray(g["faces"][...]).max()) < n_points
            assert s.field_names(lod=lvl) == s.field_names(), (sid, lvl)
            for loc, n in (("point", n_points), ("cell", n_faces)):
                assert s.field_names(loc, lod=lvl) == s.field_names(loc), (sid, lvl, loc)
                for name in s.field_names(loc, lod=lvl):
                    arr = s.array(name, loc, lvl)
                    assert arr.shape[-2] == n, (sid, lvl, loc, name, arr.shape)
                    assert arr.shape[-1] == s.array(name, loc).shape[-1]
                    assert s.dims(name, loc, lvl) == s.dims(name, loc)
                    assert arr.dtype == s.array(name, loc).dtype or arr.dtype == np.float32
                    assert np.isfinite(np.asarray(arr[...])).all() == bool(
                        np.isfinite(s.field(name, location=loc)).all()
                    )
            st = s.meta.lod_stats[str(lvl)]
            assert (st["n_faces"], st["n_points"]) == (n_faces, n_points)
            assert abs(st["n_faces"] - st["target"]) <= 0.05 * st["target"] + 2
        # one location and matching dims per field name, in the sample and in the manifest
        for f in s.meta.fields:
            if f.location in ("point", "cell") and f.dims:
                assert f.location in f.dims, (sid, f.name, f.location, f.dims)
                assert f.dims == s.dims(f.name, f.location)
    names = [f["name"] for f in manifest["fields"]]
    assert len(names) == len(set(names))
    for f in manifest["fields"]:
        if f["location"] in ("point", "cell") and f.get("dims"):
            assert f["location"] in f["dims"], f
    report = validate_dataset(root)
    assert not report.errors, [(i.where, i.message) for i in report.errors]
    assert VersionStore(root).status().clean


def _lod_changes(result: Any) -> list[str]:
    return [c.target for c in result.changes if "/lod/" in c.target]


def _level0(root: Path) -> dict[str, dict[str, list]]:
    """Level-0 file hashes of HEAD per sample (everything outside ``lod/`` except
    ``meta.json``, which records the face counts of the rebuilt levels)."""
    store = VersionStore(root)
    v = store.version("HEAD")
    return {
        sid: {
            p: e
            for p, e in store.subtree(h).items()
            if "/lod/" not in p and not p.endswith("meta.json")
        }
        for sid, h in v.samples.items()
    }


def test_rename_is_applied_to_every_level_without_a_rebuild(tmp_path: Path) -> None:
    root = _with_lods(tmp_path / "ds")
    chunk = root / "samples/run_1/surface.zarr/lod/2/cell_fields/pMean/zarr.json"
    inode = chunk.stat().st_ino
    r = _run(root, "prep.fields.rename", {"map": {"pMean": "p_kin"}})
    assert not _lod_changes(r) and r.tolerance is None and r.determinism == "bitwise"
    moved = root / "samples/run_1/surface.zarr/lod/2/cell_fields/p_kin/zarr.json"
    assert moved.stat().st_ino == inode and not chunk.exists()
    assert "p_kin" in Dataset(root).sample("run_1").field_names(lod=1)
    _assert_consistent(root, [1, 2])


def test_fields_map_then_convert_then_derive_keep_every_level_in_line(tmp_path: Path) -> None:
    root = _with_lods(tmp_path / "ds")
    r = _run(root, "prep.fields.map")
    assert not _lod_changes(r)  # renames only
    s = Dataset(root).sample("run_1")
    assert "kinematic_pressure" in s.field_names(lod=2) and "pMean" not in s.field_names(lod=2)
    _assert_consistent(root, [1, 2])

    before = {n: _weighted_mean(root, "run_1", "kinematic_pressure", n) for n in (0, 1, 2)}
    r = _run(
        root,
        "prep.units.convert",
        {
            "conversions": {"kinematic_pressure": "Pa"},
            "multiply_by": {"kinematic_pressure": {"rho": RHO}},
            "rename": {"kinematic_pressure": "pressure"},
        },
    )
    assert _lod_changes(r) == ["samples/run_1/lod/1", "samples/run_1/lod/2"]
    assert all(
        c.detail.get("reason") == "level 0 changed" for c in r.changes if "/lod/" in c.target
    )
    # level 0 keeps the op's class; the rebuilt levels are recorded with decimate's tolerance
    assert r.determinism == "bitwise" and set(r.tolerance or {}) == LOD_KEYS
    assert r.stats["lods_rebuilt"]["levels"] == [1, 2]
    rec = VersionStore(root).record(r.provenance_id)
    assert rec is not None and rec["lods_rebuilt"] == {"run_1": [1, 2]}
    assert set(rec["tolerance"]) == LOD_KEYS
    for n in (0, 1, 2):
        after = _weighted_mean(root, "run_1", "pressure", n)
        assert after == pytest.approx(RHO * before[n], rel=0.02), n
    assert _weighted_mean(root, "run_1", "pressure", 2) == pytest.approx(
        _weighted_mean(root, "run_1", "pressure", 0), abs=1.5
    )  # values span +-54 Pa around a mean near zero
    _assert_consistent(root, [1, 2])
    info = {f.name: f for f in Dataset(root).sample("run_1").meta.fields}
    assert info["pressure"].units == "Pa"

    r = _run(root, "prep.scalars.derive", {"derive": ["cd", "cp"], "normals": "into_body"})
    assert _lod_changes(r) == ["samples/run_1/lod/1", "samples/run_1/lod/2"]
    s = Dataset(root).sample("run_1")
    added = set(s.field_names()) - {"pressure", "kinematic_wall_shear_stress", "y_plus"}
    assert added and all(n in s.field_names(lod=2) for n in added)
    _assert_consistent(root, [1, 2])

    # a rename with a units-only metadata change rebuilds nothing
    r = _run(root, "prep.fields.rename", {"map": {"y_plus": "yp"}, "units": {"yp": "1"}})
    assert not _lod_changes(r)
    _assert_consistent(root, [1, 2])


def test_merge_seams_rebuilds_the_levels_from_the_merged_geometry(tmp_path: Path) -> None:
    root = _with_lods(tmp_path / "ds", "seam_soup")
    before = {sid: Dataset(root).sample(sid).points().shape[0] for sid in Dataset(root).sample_ids}
    r = _run(root, "prep.mesh.merge_seams", workers=2)
    assert len(_lod_changes(r)) == 2 * 3 and set(r.tolerance or {}) == LOD_KEYS
    ds = Dataset(root)
    for sid in ds.sample_ids:
        s = ds.sample(sid)
        assert s.points().shape[0] < before[sid]
        assert s.meta.lod_stats["0"]["n_points"] == s.points().shape[0]
        # cell-only fields reach the levels by area-weighted averaging: the mean is kept
        for lvl in (1, 2):
            assert _weighted_mean(root, sid, "pressure", lvl) == pytest.approx(
                _weighted_mean(root, sid, "pressure", 0), abs=1.5
            )
    _assert_consistent(root, [1, 2])


def test_nan_fix_reaches_the_levels(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    make_case("naming", root)
    ed = SampleEditor(root / "samples" / "run_1", "run_1")
    p = ed.field("pMean", location="cell").copy()
    p[::40] = np.nan
    ed.write_field("pMean", p, location="cell")
    assert ed.lods_stale and not ed.geometry_stale
    _run(root, "prep.mesh.decimate", LODS)
    assert not np.isfinite(Dataset(root).sample("run_1").field("pMean", lod=2)).all()
    r = _run(root, "prep.fields.nan_fix", {"policy": "fill_mean"})
    assert _lod_changes(r) == ["samples/run_1/lod/1", "samples/run_1/lod/2"]
    for lvl in (0, 1, 2):
        assert np.isfinite(Dataset(root).sample("run_1").field("pMean", lod=lvl)).all()
    _assert_consistent(root, [1, 2])


def test_point_copies_integer_and_deleted_fields(tmp_path: Path) -> None:
    root = _with_lods(tmp_path / "ds")
    # a point copy next to the native cell field: the entry stays cell with cell dims
    r = _run(root, "prep.fields.cast_location", {"to": "point", "fields": ["pMean"]})
    assert _lod_changes(r)
    _assert_consistent(root, [1, 2])
    r = _run(
        root,
        "prep.units.convert",
        {"conversions": {"pMean": "Pa"}, "multiply_by": {"pMean": {"rho": RHO}}},
    )
    s = Dataset(root).sample("run_1")
    info = {f.name: f for f in s.meta.fields}["pMean"]
    assert info.location == "cell" and info.dims == ["cell", "component"]
    assert s.field_names("point", lod=2) == ["pMean"]
    _assert_consistent(root, [1, 2])
    # integer labels are gathered (nearest face), never averaged
    r = _run(root, "prep.mesh.part_ids", {})
    s = Dataset(root).sample("run_1")
    name = next(n for n in s.field_names("cell") if n not in info.name and "part" in n)
    assert s.array(name, "cell", 2).dtype.kind == "i"
    assert set(np.unique(s.field(name, lod=2))) <= set(np.unique(s.field(name)))
    _assert_consistent(root, [1, 2])
    # deleting a level-0 field through the editor marks the levels stale; the rebuild drops it
    ed = SampleEditor(root / "samples" / "run_1", "run_1")
    ed.delete_field("yPlusMean")
    assert ed.lods_stale
    built = rebuild_lods(ed)
    assert sorted(built) == [1, 2] and not ed.lods_stale
    assert "yPlusMean" not in Dataset(root).sample("run_1").field_names(lod=1)


def test_level0_is_independent_of_the_worker_count(tmp_path: Path) -> None:
    src = _with_lods(tmp_path / "src", "seam_soup")
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(src, a)
    shutil.copytree(src, b)
    convert = {"conversions": {"pressure": "kPa"}, "from_units": {"pressure": "Pa"}}
    for root, workers in ((a, 1), (b, 2)):
        _run(root, "prep.mesh.merge_seams", workers=workers)
        r = _run(root, "prep.units.convert", convert, workers=workers)
        assert len(_lod_changes(r)) == 6
        _assert_consistent(root, [1, 2])
    la, lb = _level0(a), _level0(b)
    assert la == lb and all(any("cell_fields/pressure" in p for p in t) for t in la.values())
    # the engine did not touch level 0 beyond what the op wrote: same as a copy without levels
    c = tmp_path / "c"
    make_case("seam_soup", c)
    _run(c, "prep.mesh.merge_seams")
    _run(c, "prep.units.convert", convert)
    lc = _level0(c)
    for sid in la:
        zarr_only = {p: e for p, e in la[sid].items() if p != "surface.zarr/zarr.json"}
        assert zarr_only == {p: e for p, e in lc[sid].items() if p != "surface.zarr/zarr.json"}


def test_preview_rebuilds_the_levels_of_its_scratch_copy_only(tmp_path: Path) -> None:
    root = _with_lods(tmp_path / "ds")
    _run(root, "prep.fields.map")
    head = VersionStore(root).head
    pv = get_op("prep.units.convert").preview(
        DatasetRef(root),
        {
            "conversions": {"kinematic_pressure": "Pa"},
            "multiply_by": {"kinematic_pressure": {"rho": 2}},
        },
        sample=Selection(renders=False),
    )
    assert pv.samples == ["run_1"] and not pv.notes, pv.notes
    assert VersionStore(root).head == head and VersionStore(root).status().clean
    _assert_consistent(root, [1, 2])


def test_preview_skips_the_rebuild_of_a_large_mesh_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import frontier_prep.base as base

    root = _with_lods(tmp_path / "ds")
    monkeypatch.setattr(base, "PREVIEW_LOD_MAX_FACES", 10)
    pv = get_op("prep.fields.nan_fix").preview(
        DatasetRef(root), {"policy": "fill_value"}, sample=Selection(renders=False)
    )
    pv2 = get_op("prep.units.convert").preview(
        DatasetRef(root),
        {"conversions": {"pMean": "Pa"}, "multiply_by": {"pMean": {"rho": 2}}},
        sample=Selection(renders=False),
    )
    assert not pv.notes  # nothing to fix, nothing stale
    assert any("not rebuilt in the preview" in n for n in pv2.notes), pv2.notes


def test_lod_target_prefers_the_recorded_target() -> None:
    assert lod_target(2, {"2": {"target": 150, "n_faces": 148}}) == 150
    # levels written before targets were recorded: the default, unless clearly smaller
    assert lod_target(2, {"2": {"n_faces": 19_998, "method": "fast_simplification"}}) == 20_000
    assert lod_target(2, {"2": {"n_faces": 432, "method": "copy"}}) == 20_000
    assert lod_target(2, {"2": {"n_faces": 3_000, "method": "fast_simplification"}}) == 3_000
    # a decimation that got stuck above the default keeps the default as its target
    assert lod_target(2, {"2": {"n_faces": 174_342, "method": "fast_simplification"}}) == 20_000
    assert lod_target(5, {}, 700) == 700 and lod_target(5, {}) is None


def test_transient_surface_levels_keep_time_dependent_fields(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    w = DatasetWriter(root, name="transient-lods", domain="cfd")
    m = _box(seed=3)
    geo = SurfaceMesh(points=m.points, faces=m.faces)
    rng = np.random.default_rng(0)
    w.write_surface_sample(
        "t_0",
        geo,
        times=0.1 * np.arange(5),
        fields={"temperature": 300 + rng.random((5, geo.n_points, 1)).astype(np.float32)},
        cell_fields={"pressure": np.stack([m.cell_fields["pressure"] * (k + 1) for k in range(5)])},
        field_units={"temperature": "K", "pressure": "Pa"},
        meta=_meta("t_0"),
        forces=_forces(m),
    )
    w.finalize()
    _run(root, "prep.mesh.decimate", LODS)
    s = Dataset(root).sample("t_0")
    assert s.array("temperature", "point", 2).shape[0] == 5
    assert s.array("pressure", "cell", 2).shape == (5, s.group["lod"]["2"]["faces"].shape[0], 1)
    _assert_consistent(root, [1, 2])
    r = _run(root, "prep.units.convert", {"conversions": {"pressure": "kPa"}})
    assert _lod_changes(r) == ["samples/t_0/lod/1", "samples/t_0/lod/2"]
    s = Dataset(root).sample("t_0")
    lod, full = s.field("pressure", lod=2), s.field("pressure")
    assert np.abs(lod).max() <= np.abs(full).max() * (1 + 1e-6)
    assert np.abs(lod[4]).mean() == pytest.approx(5 * np.abs(lod[0]).mean(), rel=1e-4)
    _assert_consistent(root, [1, 2])


def test_mesh_levels_are_rebuilt_only_when_the_geometry_changes(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    n = 70
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    pts = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (i * n + j).ravel()
    cells = np.concatenate(
        [np.stack([a, a + n, a + 1], 1), np.stack([a + 1, a + n, a + n + 1], 1)]
    ).astype(np.int32)
    w = DatasetWriter(root, name="mesh-lods", domain="cfd")
    w.write_mesh_sample(
        "m_0",
        points=pts,
        cells=cells,
        fields={"pressure": pts[:, :1] * 1000.0},
        field_units={"pressure": "Pa"},
    )
    w.finalize()
    levels = _levels(root, "m_0")
    assert levels  # 9.5k triangles: level 3 (5k) is a vertex map
    r = _run(root, "prep.units.convert", {"conversions": {"pressure": "kPa"}})
    assert not _lod_changes(r) and r.tolerance is None  # fields are gathered through the map
    r = _run(root, "prep.mesh.transform", {"scale": 2.0})
    assert _lod_changes(r) == [f"samples/m_0/lod/{lvl}" for lvl in levels]
    s = Dataset(root).sample("m_0")
    for lvl in levels:
        g: Any = s.group["lod"][str(lvl)]
        vm = np.asarray(g["vertex_map"][...])
        assert np.allclose(np.asarray(g["points"][...]), s.points()[vm])
        assert float(np.asarray(g["points"][...]).max()) == pytest.approx(2.0)
    assert VersionStore(root).status().clean


def test_gate_fails_closed_when_an_lod_rebuild_invalidates_a_failing_closure(tmp_path: Path):
    """The audit bug: force_closure fails on a sample, a later op changes the sample (here an
    LOD rebuild), `effective_assertions` drops the failing record, and the gate used to pass
    on the remaining evidence. Evidence that lapsed is "not evaluated", never a pass."""
    from frontier_data.versions import VersionStore
    from frontier_prep.benchmarks import make_case
    from frontier_prep.op import get_op

    root = tmp_path / "seam"
    make_case("seam_soup", root)
    store = VersionStore(root)
    store.init()
    failing = store.head
    for sid in ("case_0", "case_1"):
        store.append_provenance(
            {
                "kind": "assertion",
                "version": failing,
                "check": "force_closure",
                "status": "fail" if sid == "case_0" else "pass",
                "target": f"samples/{sid}",
                "detail": {"rel_err": 0.4 if sid == "case_0" else 1e-5},
            }
        )
    gate = get_op("prep.quality.gate")
    res = gate.run(root, {})
    assert res.stats["gate"]["failing_checks"] == {"force_closure": 1}
    assert "train-ready" not in VersionStore(root).tags()
    # the sample changes: new LOD groups (any level-0 rewrite does the same through a rebuild)
    get_op("prep.mesh.decimate").run(root, {"levels": [1], "targets": {"1": 100}})
    store = VersionStore(root)
    effective = {(a["check"], a["target"]) for a in store.effective_assertions()}
    assert ("force_closure", "samples/case_0") not in effective  # the record lapsed
    get_op("prep.quality.check").run(root, {"write_report": False})
    res = gate.run(root, {})
    g = res.stats["gate"]
    assert g["ok"] is False and "train-ready" not in VersionStore(root).tags()
    # both samples: the failing one, and the passing one whose closure is no longer current
    assert g["not_evaluated"]["force_closure"] == ["samples/case_0", "samples/case_1"]
    assert "force_closure is not evaluated" in res.summary and "run the check again" in res.summary
    assert res.assertions[0].status == "fail"
    # once the closure is evaluated again on the current content, nothing is "not evaluated"
    store = VersionStore(root)
    for sid in ("case_0", "case_1"):
        store.append_provenance(
            {
                "kind": "assertion",
                "version": store.head,
                "check": "force_closure",
                "status": "pass",
                "target": f"samples/{sid}",
                "detail": {"rel_err": 1e-5},
            }
        )
    assert "force_closure" not in gate.run(root, {}).stats["gate"]["not_evaluated"]


def test_plan_lists_the_lod_rebuilds_and_derive_drops_a_stale_point_copy(tmp_path: Path) -> None:
    root = _with_lods(tmp_path / "ds")
    sid = Dataset(root).sample_ids[0]
    levels = _levels(root, sid)
    assert levels
    # a rename needs no rebuild: the plan lists none
    plan = get_op("prep.fields.rename").plan(DatasetRef(root), {"map": {"yPlusMean": "y_plus"}})
    assert not [c for c in plan.changes if "/lod/" in c.target]
    _run(root, "prep.fields.map")
    convert = {
        "conversions": {"kinematic_pressure": "Pa", "kinematic_wall_shear_stress": "Pa"},
        "multiply_by": {
            "kinematic_pressure": {"rho": RHO},
            "kinematic_wall_shear_stress": {"rho": RHO},
        },
        "rename": {
            "kinematic_pressure": "pressure",
            "kinematic_wall_shear_stress": "wall_shear_stress",
        },
    }
    # a value change does: one planned change per stored level, before anything runs
    plan = get_op("prep.units.convert").plan(DatasetRef(root), convert)
    planned = sorted(c.target for c in plan.changes if "/lod/" in c.target)
    assert planned == [f"samples/{sid}/lod/{lvl}" for lvl in levels]
    assert all(c.detail.get("planned") for c in plan.changes if "/lod/" in c.target)
    _run(root, "prep.units.convert", convert)
    # a point copy of Cp, then derive rewrites the cell array with other values
    _run(root, "prep.fields.cast_location", {"to": "point", "fields": ["pressure_coefficient"]})
    assert "pressure_coefficient" in Dataset(root).sample(sid).field_names("point")
    before = Dataset(root).sample(sid).field("pressure_coefficient", location="cell").copy()
    res = _run(root, "prep.scalars.derive", {"derive": ["cd", "cp"], "p_inf": 7.5})
    s = Dataset(root).sample(sid)
    assert not np.allclose(s.field("pressure_coefficient", location="cell"), before)
    assert "pressure_coefficient" not in s.field_names("point")  # the stale copy is gone
    removed = [c for c in res.changes if c.action == "remove"]
    assert removed and removed[0].target == f"samples/{sid}/fields/pressure_coefficient"
    _assert_consistent(root, levels)


def test_gate_requires_the_checks_of_the_dataset_kind(tmp_path: Path) -> None:
    """A force closure is asked of surfaces with forces, a time continuity of transient data;
    a transient grid is not refused for lacking surface checks it cannot have."""
    from types import SimpleNamespace

    from frontier_prep.ops.quality import QualityGate

    def ref(kind: str, temporal: str, forces: dict[str, float]) -> Any:
        manifest = SimpleNamespace(
            sample_kind=kind, temporal=temporal, samples=[SimpleNamespace(forces=forces)]
        )
        return SimpleNamespace(manifest=manifest)

    req = QualityGate.required_for
    common = ["predictable_fields", "nan_inf", "split_leakage"]
    assert req(ref("surface", "steady", {"cd": 0.3})) == [
        "force_consistency",
        "force_closure",
        "area_sum",
        *common,
    ]
    assert req(ref("surface", "steady", {})) == ["area_sum", *common]
    assert req(ref("grid", "transient", {})) == ["temporal_continuity", *common]
    assert req(ref("mesh", "steady", {})) == ["mesh_cells", *common]
