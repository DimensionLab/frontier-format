# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frontier_data.dataset import (
    Dataset,
    DatasetWriter,
    MeshStats,
    Provenance,
    SampleMeta,
    read_surface,
    write_surface,
)
from frontier_data.lod import build_lods
from frontier_data.mesh import SurfaceMesh, make_box_mesh
from frontier_data.quality import (
    check_force_consistency,
    compute_sample_stats,
    integrate_drag,
    robust_z,
    run_quality,
    series_points_for,
)
from frontier_data.splits import find_leakage, parameter_holdout_split, random_split


def _meta(sid: str, mesh: SurfaceMesh, slant: float, cd: float) -> SampleMeta:
    return SampleMeta(
        id=sid,
        dataset="synthetic",
        solver="OpenFOAM",
        mesh=MeshStats(
            n_points=mesh.n_points,
            n_faces=mesh.n_faces,
            bounds=list(mesh.bounds),
            area=mesh.area,
            stl_area=mesh.area,
        ),
        geometry_params={"slant-angle-degrees": slant, "body-length": 1000.0 + slant},
        conditions={"u_inf": 1.0, "rho": 1.0, "reference_area": 0.112032},
        forces={"cd": cd, "cl": 0.0},
        field_roles={"pressure": "p", "wall_shear_stress": "wallShearStress"},
        provenance=Provenance(source="synthetic"),
    )


def _cd(mesh: SurfaceMesh) -> float:
    cd, _, _ = integrate_drag(
        mesh,
        mesh.cell_fields["p"],
        mesh.cell_fields["wallShearStress"],
        drag_dir=np.array([1.0, 0, 0]),
        rho=1.0,
        u_inf=1.0,
        a_ref=0.112032,
    )
    return cd


def build_dataset(root: Path, n: int = 6) -> Dataset:
    writer = DatasetWriter(root, name="synthetic", source={"kind": "test"})
    for i in range(n):
        mesh = make_box_mesh(subdivisions=10)
        mesh.cell_fields["p"] = mesh.cell_fields["p"] * (1 + 0.1 * i)
        mesh.point_fields["p"] = mesh.cell_to_point(mesh.cell_fields["p"])
        lods, _ = build_lods(mesh, (0, 2), targets={0: None, 1: 600, 2: 300, 3: 100})
        meta = _meta(f"box_{i}", mesh, slant=10.0 + 5 * i, cd=_cd(mesh))
        writer.write_sample(f"box_{i}", lods, meta, meta.forces)
    writer.finalize()
    return Dataset(root)


def test_surface_zarr_roundtrip(tmp_path: Path, box: SurfaceMesh) -> None:
    lods, _ = build_lods(box, (0, 2), targets={0: None, 1: 300, 2: 200, 3: 50})
    write_surface(tmp_path / "surface.zarr", lods)
    back = read_surface(tmp_path / "surface.zarr", 0)
    assert np.array_equal(back.points, box.points)
    assert np.array_equal(back.faces, box.faces)
    assert np.allclose(back.areas(), box.areas())
    assert set(back.point_fields) == set(box.point_fields)
    assert np.allclose(back.cell_fields["wallShearStress"], box.cell_fields["wallShearStress"])
    lod2 = read_surface(tmp_path / "surface.zarr", 2)
    assert lod2.n_faces == lods[2].n_faces
    # zarr v3 layout on disk
    assert (tmp_path / "surface.zarr" / "zarr.json").exists()
    meta = json.loads((tmp_path / "surface.zarr" / "zarr.json").read_text())
    assert meta["zarr_format"] == 3


def test_dataset_layout_and_reader(tmp_path: Path) -> None:
    ds = build_dataset(tmp_path / "ds", n=3)
    assert len(ds) == 3
    assert (tmp_path / "ds" / "manifest.json").exists()
    assert ds.manifest.format == "frontier-dataset" and ds.manifest.version == 1
    assert ds.sample_ids == ["box_0", "box_1", "box_2"]
    s = ds.surface("box_1", lod=2)
    assert s.n_faces <= 300
    meta = ds.meta("box_2")
    assert meta.solver == "OpenFOAM" and meta.lod_levels == [0, 2]
    assert ds.forces("box_0")["cd"] == pytest.approx(ds.entry("box_0").forces["cd"])
    assert ds.manifest.geometry_params == ["slant-angle-degrees", "body-length"]
    assert [f.name for f in ds.manifest.fields] == ["p", "wallShearStress"]


def test_force_consistency_check(box: SurfaceMesh) -> None:
    meta = _meta("x", box, 20.0, _cd(box))
    res = check_force_consistency(box, meta)
    assert res.status == "pass" and res.detail["rel_err"] < 1e-9
    meta.forces["cd"] = _cd(box) * 1.2
    assert check_force_consistency(box, meta).status == "fail"
    meta.field_roles = {}
    assert check_force_consistency(box, meta).status == "skip"


def test_robust_z() -> None:
    z = robust_z(np.array([1.0, 1.1, 0.9, 1.0, 10.0]))
    assert abs(z[-1]) > 6 and abs(z[:-1]).max() < 1.5


def test_quality_report(tmp_path: Path) -> None:
    ds = build_dataset(tmp_path / "ds", n=6)
    # inject a NaN into one sample's pressure to make a failure
    import zarr

    g: Any = zarr.open_group(str(ds.sample_dir("box_3") / "surface.zarr"), mode="r+")
    arr: Any = g["fields"]["p"]
    vals = arr[...]
    vals[0, 0] = np.nan
    arr[...] = vals
    ds.manifest.splits = {
        "train": ["box_0", "box_1", "box_2"],
        "val": ["box_3"],
        "test": ["box_4", "box_3"],
    }
    ds.save_manifest()
    report = run_quality(ds)
    assert (tmp_path / "ds" / "quality" / "report.json").exists()
    html = (tmp_path / "ds" / "quality" / "report.html").read_text()
    assert "<svg" in html and "box_3" in html
    by_id = {r["id"]: r for r in report["samples"]}
    assert by_id["box_3"]["status"] == "fail"
    checks = {c["check"]: c for c in by_id["box_3"]["checks"]}
    assert checks["nan_inf"]["status"] == "fail"
    assert checks["force_consistency"]["status"] == "pass"
    assert checks["area_sum"]["status"] == "pass"
    assert by_id["box_0"]["status"] in ("pass", "warn")
    dchecks = {c["check"]: c for c in report["dataset_checks"]}
    assert dchecks["split_leakage"]["status"] == "fail"
    assert dchecks["split_leakage"]["detail"]["id_overlap"] == {"val&test": ["box_3"]}
    assert dchecks["parameter_coverage"]["status"] in ("pass", "warn")
    assert "p" in report["recommended_normalization"]["fields"]
    assert Dataset(tmp_path / "ds").manifest.quality == "quality/report.json"


def test_splits() -> None:
    ids = [f"run_{i}" for i in range(1, 41)]
    r = random_split(ids, (0.8, 0.1, 0.1), seed=1)
    assert sorted(sum(r.splits.values(), [])) == sorted(ids)
    assert len(r.splits["val"]) == 4 and len(r.splits["test"]) == 4
    assert r.strategy["name"] == "random"
    assert random_split(ids, seed=1).splits == r.splits
    params = {i: {"slant-angle-degrees": float(k)} for k, i in enumerate(ids)}
    h = parameter_holdout_split(
        ids, params, "slant-angle-degrees", holdout_fraction=0.15, side="top"
    )
    assert len(h.splits["test"]) == 6
    assert all(params[i]["slant-angle-degrees"] >= 34 for i in h.splits["test"])
    assert all(params[i]["slant-angle-degrees"] < 34 for i in h.splits["train"] + h.splits["val"])
    assert h.strategy["threshold"] == 34.0 and h.strategy["parameter"] == "slant-angle-degrees"
    assert not find_leakage(h.splits)
    assert find_leakage({"a": ["x", "y"], "b": ["y"]}) == {"a&b": ["y"]}
    with pytest.raises(ValueError):
        parameter_holdout_split(ids, params, "missing")


def test_sample_paths_are_confined_to_the_dataset(tmp_path: Path) -> None:
    """AI-06 / SRV-05: a manifest's sample path is data and must not leave the dataset."""
    import json as _json

    from frontier_data import Dataset
    from frontier_data.synthetic import write_synthetic_dataset

    root = tmp_path / "ds"
    write_synthetic_dataset(root, kind="grid", n_samples=2, n_steps=3, size=8)
    outside = tmp_path / "outside"
    (root / "samples" / "traj_000").rename(outside)
    m = _json.loads((root / "manifest.json").read_text())
    for bad in ("../outside", str(outside)):
        m["samples"][0]["path"] = bad
        (root / "manifest.json").write_text(_json.dumps(m))
        with pytest.raises(ValueError, match="leaves the dataset directory"):
            Dataset(root).sample_dir("traj_000")
    m["samples"][0]["path"] = "samples/traj_000"
    (root / "manifest.json").write_text(_json.dumps(m))
    (root / "samples" / "traj_000").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the dataset"):
        Dataset(root).sample_dir("traj_000")
    assert Dataset(root).sample_dir("traj_001").is_dir()


def _stable(report: dict[str, Any]) -> str:
    """The report without its wall-clock fields."""
    return json.dumps(
        {k: v for k, v in report.items() if k not in ("generated_at", "elapsed_s")},
        sort_keys=True,
        default=str,
    )


def test_run_quality_accepts_precomputed_sample_stats(tmp_path: Path) -> None:
    """Per-sample statistics computed elsewhere (a worker pool) and handed in give the report
    of a serial run: all of them, a subset (the rest is computed in place), in any dict
    order; the caller's objects are not modified, so they can be used again."""
    ds = build_dataset(tmp_path / "ds", n=6)
    serial = _stable(run_quality(ds, out_dir=tmp_path / "serial"))
    ids = ds.sample_ids
    sp = series_points_for(ds)
    # computed in reverse order, as a pool would complete them in any order
    pre = {sid: compute_sample_stats(ds, sid, series_points=sp) for sid in reversed(ids)}
    n_checks = {sid: len(st.checks) for sid, st in pre.items()}
    keys = {sid: {n: sorted(v) for n, v in st.fields.items()} for sid, st in pre.items()}
    assert _stable(run_quality(ds, out_dir=tmp_path / "pre", precomputed=pre)) == serial
    # run_quality adds dataset-level checks and z-scores to its own copies only
    assert {sid: len(st.checks) for sid, st in pre.items()} == n_checks
    assert {sid: {n: sorted(v) for n, v in st.fields.items()} for sid, st in pre.items()} == keys
    assert _stable(run_quality(ds, out_dir=tmp_path / "again", precomputed=pre)) == serial
    part = {sid: pre[sid] for sid in ids[1::2]}
    assert _stable(run_quality(ds, out_dir=tmp_path / "part", precomputed=part)) == serial
    # an entry filed under another sample's id is refused, not silently reported
    with pytest.raises(ValueError, match="belong to sample"):
        run_quality(ds, out_dir=tmp_path / "bad", precomputed={ids[0]: pre[ids[1]]})


def test_field_bounds_closure() -> None:
    from frontier_data.quality import check_field_bounds

    class Sample:
        def __init__(self, t: np.ndarray) -> None:
            self.t = t

        def field_names(self) -> list[str]:
            return ["temperature"]

        def array(self, _name: str) -> np.ndarray:
            return self.t

    inside = Sample(np.linspace(300.0, 350.0, 1000))
    assert check_field_bounds(inside, {"temperature": [300.0, 350.0]}).status == "pass"
    hot = Sample(np.concatenate([np.full(990, 320.0), np.full(10, 400.0)]))
    res = check_field_bounds(hot, {"temperature": [300.0, 350.0]})
    assert res.status == "fail" and res.detail["temperature"]["outside_frac"] == 0.01
    assert check_field_bounds(inside, {"pressure": [0, 1]}).status == "skip"
