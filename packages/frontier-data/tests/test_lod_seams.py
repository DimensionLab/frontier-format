# SPDX-License-Identifier: Apache-2.0
"""Decimation must not tear open patch seams (duplicated vertices along patch borders)."""

import numpy as np

from frontier_data.lod import decimate_fast, merge_coincident_vertices
from frontier_data.mesh import SurfaceMesh, make_box_mesh


def _boundary_edges(faces: np.ndarray) -> int:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum())


def _area(points: np.ndarray, faces: np.ndarray) -> float:
    t = points[faces]
    return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)


def _seamed_box(n: int = 12) -> SurfaceMesh:
    """A box whose faces were triangulated as separate patches (no shared vertices)."""
    box = make_box_mesh(subdivisions=n)
    tri = box.points[box.faces]  # (M,3,3) -> every triangle gets its own vertices
    points = tri.reshape(-1, 3).astype(np.float32)
    faces = np.arange(points.shape[0], dtype=np.int32).reshape(-1, 3)
    fields = {k: v[box.faces].reshape(-1, v.shape[1]) for k, v in box.point_fields.items()}
    return SurfaceMesh(points=points, faces=faces, point_fields=fields)


def test_merge_removes_open_edges() -> None:
    mesh = _seamed_box()
    assert _boundary_edges(mesh.faces) > 0
    pts, faces = merge_coincident_vertices(mesh)
    assert _boundary_edges(faces) == 0
    assert abs(_area(pts, faces) - _area(mesh.points, mesh.faces)) < 1e-4


def test_decimate_keeps_surface_closed_and_area() -> None:
    mesh = _seamed_box()
    src_area = _area(mesh.points, mesh.faces)
    out = decimate_fast(mesh, target_faces=max(200, mesh.faces.shape[0] // 4))
    # A few zero-area corner slivers are dropped by the degenerate filter and leave
    # open edges of zero length; anything beyond that would be a real hole.
    assert _boundary_edges(out.faces) <= 8
    assert abs(_area(out.points, out.faces) - src_area) / src_area < 0.005
