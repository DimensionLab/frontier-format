# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frontier_data.lod import LOD_TARGET_FACES, build_lods, decimate
from frontier_data.mesh import SurfaceMesh, make_box_mesh
from frontier_data.readers import vtk as rd

pv = pytest.importorskip("pyvista")


def test_vtp_roundtrip_with_quads(tmp_path: Path) -> None:
    # a quad-based plane with cell data -> triangulated with parent map, fields converted
    plane = pv.Plane(i_resolution=4, j_resolution=3)  # 12 quads
    plane.cell_data["p"] = np.arange(plane.n_cells, dtype=np.float32)
    plane.cell_data["wss"] = np.ones((plane.n_cells, 3), np.float32)
    path = tmp_path / "plane.vtp"
    plane.save(str(path))
    mesh = rd.read_vtp(path)
    assert mesh.n_faces == 24
    assert mesh.face_parent is not None and mesh.face_parent.max() == 11
    assert mesh.cell_fields["p"].shape == (24, 1)
    assert mesh.cell_fields["wss"].shape == (24, 3)
    assert "p" in mesh.point_fields and mesh.point_fields["p"].shape == (mesh.n_points, 1)
    # distribute parent areas: each quad's area split across its two triangles
    native = np.full(12, 2.0, np.float32)
    per_tri = rd.distribute_parent_areas(mesh, native)
    assert np.isclose(per_tri.sum(), 24.0)
    assert np.allclose(np.bincount(mesh.face_parent, weights=per_tri), 2.0)


def test_stl_and_csv(tmp_path: Path) -> None:
    box = make_box_mesh(subdivisions=2)
    box.to_pyvista().save(str(tmp_path / "geo.stl"))
    stl = rd.read_stl(tmp_path / "geo.stl")
    assert np.isclose(stl.area, box.area, rtol=1e-5)
    csv = tmp_path / "f.csv"
    csv.write_text("cd, cl\n 2.3e-01, -9.4e-02\n")
    assert rd.read_scalar_csv(csv) == {"cd": 0.23, "cl": -0.094}
    np.save(tmp_path / "a.npy", np.ones(5, np.float32))
    assert rd.read_cell_area_npy(tmp_path / "a.npy").shape == (5,)


def _check_lod(src: SurfaceMesh, out: SurfaceMesh, target: int) -> None:
    assert out.n_faces <= target * 1.05
    assert out.points.dtype == np.float32 and out.faces.dtype == np.int32
    assert set(out.point_fields) == set(src.point_fields)
    assert set(out.cell_fields) == set(src.cell_fields)
    assert out.point_fields["p"].shape == (out.n_points, 1)
    assert out.cell_fields["wallShearStress"].shape == (out.n_faces, 3)
    assert not out.degenerate_faces().any()
    # the field range is preserved by averaging (no new extrema)
    assert out.point_fields["p"].min() >= src.point_fields["p"].min() - 1e-6
    assert out.point_fields["p"].max() <= src.point_fields["p"].max() + 1e-6
    # bounding box roughly preserved
    assert np.allclose(out.bounds, src.bounds, atol=0.05)


@pytest.mark.parametrize("method", ["fast", "decimate_pro"])
def test_decimate_methods(method: str) -> None:
    src = make_box_mesh(subdivisions=20)  # 4800 faces
    if method == "fast":
        pytest.importorskip("fast_simplification")
        out, used = decimate(src, 800, method="fast")
        assert used == "fast_simplification"
    else:
        from frontier_data.lod import decimate_pyvista

        out = decimate_pyvista(src, 800)
    _check_lod(src, out, 800)


def test_build_lods_cascade() -> None:
    src = make_box_mesh(subdivisions=24)  # 6912 faces
    targets = {0: None, 1: 3000, 2: 1000, 3: 300}
    lods, timings = build_lods(src, (0, 2, 3), targets=targets)
    assert set(lods) == {0, 2, 3}
    assert lods[0] is src
    assert lods[2].n_faces <= 1000 and lods[3].n_faces <= 300
    assert [t.level for t in timings] == [0, 2, 3]
    # copy when target >= n_faces
    same, used = decimate(src, 10**6)
    assert used == "copy" and same.n_faces == src.n_faces
    assert LOD_TARGET_FACES[1] == 100_000


def test_validator_reports_stale_lod_levels_and_native_location(tmp_path):
    """LOD levels that do not describe level 0 are validation errors; `native_location`
    names where a field's data lives (meta.fields[].location) next to a derived point copy."""
    import json
    import shutil

    import numpy as np

    from frontier_data import Dataset, DatasetWriter, validate_dataset
    from frontier_data.mesh import make_box_mesh

    root = tmp_path / "ds"
    w = DatasetWriter(root, name="lodcheck", domain="cfd")
    m = make_box_mesh(size=(1.0, 0.4, 0.3), subdivisions=6)
    m.cell_fields["pMean"] = np.linspace(0, 1, m.n_faces, dtype=np.float32)
    m.with_point_fields_from_cells()  # a derived point copy next to the native cell field
    lods, _ = build_lods(m, (0, 1), targets={1: 100})
    w.write_surface_sample("s0", lods)
    w.finalize()
    assert validate_dataset(root).ok
    s = Dataset(root).sample("s0")
    assert next(f.location for f in s.meta.fields if f.name == "pMean") == "cell"
    # the format's rule is unchanged (no location: the point array first); the native array
    # is addressed through native_location, which agrees with the metadata
    assert s.native_location("pMean") == "cell"
    assert s.array("pMean", s.native_location("pMean")).shape[0] == m.n_faces
    assert s.dims("pMean", s.native_location("pMean"))[0] == "cell"
    assert s.array("pMean").shape[0] == m.n_points
    # level 0 renamed, level 1 left behind (what a level-0-only edit does)
    store = root / "samples" / "s0" / "surface.zarr"
    for parent in ("fields", "cell_fields"):
        shutil.move(str(store / parent / "pMean"), str(store / parent / "pressure"))
    meta = json.loads((root / "samples" / "s0" / "meta.json").read_text())
    for f in meta["fields"]:
        f["name"] = "pressure" if f["name"] == "pMean" else f["name"]
    (root / "samples" / "s0" / "meta.json").write_text(json.dumps(meta))
    man = json.loads((root / "manifest.json").read_text())
    for f in man["fields"]:
        f["name"] = "pressure" if f["name"] == "pMean" else f["name"]
    (root / "manifest.json").write_text(json.dumps(man))
    rep = validate_dataset(root)
    assert not rep.ok
    msgs = [i.message for i in rep.errors if "/lod/1" in i.where]
    assert msgs and "out of date" in msgs[0] and "pMean" in msgs[0]
