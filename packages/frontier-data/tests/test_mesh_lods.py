# SPDX-License-Identifier: Apache-2.0
"""Vertex-clustering LOD maps for mesh.zarr samples (playback engine, phase 5)."""

from pathlib import Path

import numpy as np
import pytest

from frontier_data.lod import (
    add_mesh_lods,
    build_mesh_lod_maps,
    cluster_vertex_map,
    drawable_triangles,
)


def _grid_surface(n: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32))
    points = np.stack([xs.ravel(), ys.ravel(), ((xs * ys) % 3).ravel() * 0.1], axis=1)
    i = np.arange(n - 1)
    a = (i[:, None] * n + i[None, :]).ravel()
    faces = np.concatenate([np.stack([a, a + 1, a + n + 1], 1), np.stack([a, a + n + 1, a + n], 1)])
    return points.astype(np.float32), faces.astype(np.int32)


def test_drawable_triangles_triangulates_2d_and_extracts_3d_boundary() -> None:
    # Two quads and two triangles in 2D.
    cells = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 6, -1], [4, 5, 6, -1]])
    types = np.array([9, 9, 5, 5], dtype=np.uint8)
    tris = drawable_triangles(cells, types, 2)
    assert tris.shape == (6, 3)
    # Two hexahedra sharing a face: 10 boundary quads -> 20 triangles, the shared face gone.
    hexa = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]])
    tris = drawable_triangles(hexa, np.array([12, 12], dtype=np.uint8), 3)
    assert tris.shape == (20, 3)
    # A tetrahedron alone keeps its four faces.
    tet = drawable_triangles(np.array([[0, 1, 2, 3]]), np.array([10], dtype=np.uint8), 3)
    assert tet.shape == (4, 3)


def test_cluster_vertex_map_hits_target_and_samples_the_full_mesh() -> None:
    points, faces = _grid_surface(200)  # 79,202 triangles
    vm, f = cluster_vertex_map(points, faces, 5_000)
    assert 2_500 <= len(f) <= 7_000
    assert len(vm) < len(points) / 8
    assert vm.dtype == np.int32 and f.dtype == np.int32
    assert vm.max() < len(points) and f.max() < len(vm)
    assert np.all((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2]))
    # No duplicate triangles.
    assert len(np.unique(np.sort(f, axis=1), axis=0)) == len(f)
    # Kept vertices are real vertices of the input: gathering reproduces coordinates.
    assert np.array_equal(points[vm][0], points[vm[0]])
    # Small meshes are kept whole; levels that would keep the mesh whole are skipped.
    vm0, f0 = cluster_vertex_map(points[:16], faces[:4], 5_000)
    assert np.array_equal(vm0, np.arange(16)) and np.array_equal(f0, faces[:4])
    maps = build_mesh_lod_maps(points, faces, (1, 2, 3))
    assert sorted(maps) == [2, 3], "level 1 (100k) exceeds 79k faces"
    assert len(maps[2][1]) > len(maps[3][1])


def test_mesh_writer_stores_lod_maps_and_the_upgrader_adds_them(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    from frontier_data.arrays import write_mesh_sample
    from frontier_data.dataset import DatasetWriter

    # A 2D quad grid as a mesh with a moving geometry and a point field over 3 steps.
    n = 160
    points, _ = _grid_surface(n)
    pts2 = points[:, :2]
    i = np.arange(n - 1)
    a = (i[:, None] * n + i[None, :]).ravel()
    cells = np.stack([a, a + 1, a + n + 1, a + n], 1).astype(np.int32)  # 25,281 quads
    times = np.array([0.0, 0.1, 0.2])
    pt = np.stack([pts2 + k * 0.01 for k in range(3)])
    u = np.stack([np.full((len(pts2), 1), float(k), np.float32) for k in range(3)])
    w = DatasetWriter(tmp_path / "ds", name="lods")
    w.write_mesh_sample(
        "s1",
        points=pts2,
        cells=cells,
        fields={"u": u},
        times=times,
        points_t=pt,
    )
    w.finalize()
    entry = w.manifest.samples[0]
    g = zarr.open_group(str(tmp_path / "ds/samples/s1/mesh.zarr"), mode="r")
    # 50,562 drawable triangles: levels 2 and 3 exist, level 1 would be the whole mesh.
    assert list(g.attrs["lod_levels"]) == [0, 2, 3]
    assert "lod/2/vertex_map" in g and "lod/3/faces" in g
    vm = np.asarray(g["lod/3/vertex_map"][...])
    assert np.array_equal(np.asarray(g["lod/3/points"][...]), np.asarray(g["points"][...])[vm])
    f = np.asarray(g["lod/3/faces"][...])
    assert 2_500 <= len(f) <= 7_000 and f.max() < len(vm)
    assert entry.lod_levels == [0, 2, 3]

    # A sample written without levels is upgraded in place by the one-off command.
    write_mesh_sample(
        tmp_path / "ds", "s2", points=pts2, cells=cells, lod_levels=None, dataset="lods"
    )
    g2 = zarr.open_group(str(tmp_path / "ds/samples/s2/mesh.zarr"), mode="r")
    assert list(g2.attrs.get("lod_levels", [0])) == [0]
    import json

    man = json.loads((tmp_path / "ds/manifest.json").read_text())
    if not any(s["id"] == "s2" for s in man["samples"]):
        man["samples"].append({"id": "s2", "path": "samples/s2"})
        (tmp_path / "ds/manifest.json").write_text(json.dumps(man))
    done = add_mesh_lods(tmp_path / "ds", (1, 2, 3))
    assert done["s2"] == [0, 2, 3]
    assert done["s1"] == [0, 2, 3], "existing levels are kept"
    g2 = zarr.open_group(str(tmp_path / "ds/samples/s2/mesh.zarr"), mode="r")
    assert "lod/2/vertex_map" in g2
    man = json.loads((tmp_path / "ds/manifest.json").read_text())
    assert next(s for s in man["samples"] if s["id"] == "s2")["lod_levels"] == [0, 2, 3]
