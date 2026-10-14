# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from frontier_data.mesh import SurfaceMesh, make_box_mesh


def test_box_mesh_geometry(box: SurfaceMesh) -> None:
    sx, sy, sz = 1.0, 0.4, 0.3
    expected = 2 * (sx * sy + sx * sz + sy * sz)
    assert box.n_faces == 6 * 8 * 8 * 2
    assert np.isclose(box.area, expected, rtol=1e-6)
    assert box.points.dtype == np.float32 and box.faces.dtype == np.int32
    assert not box.degenerate_faces().any()
    # closed surface: outward normals integrate to zero
    n = box.face_normals() * box.face_areas()[:, None]
    assert np.allclose(n.sum(axis=0), 0, atol=1e-6)


def test_cell_point_roundtrip(box: SurfaceMesh) -> None:
    const = np.full(box.n_faces, 3.0, np.float32)
    pts = box.cell_to_point(const)
    assert pts.shape == (box.n_points, 1)
    assert np.allclose(pts, 3.0)
    back = box.point_to_cell(pts)
    assert back.shape == (box.n_faces, 1)
    assert np.allclose(back, 3.0)
    vec = np.tile(np.array([1.0, 2.0, 3.0], np.float32), (box.n_faces, 1))
    assert np.allclose(box.cell_to_point(vec), [1.0, 2.0, 3.0])


def test_field_validation() -> None:
    m = make_box_mesh(subdivisions=1)
    try:
        SurfaceMesh(points=m.points, faces=m.faces, point_fields={"bad": np.zeros(3)})
    except ValueError as exc:
        assert "bad" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_degenerate_detection() -> None:
    m = make_box_mesh(subdivisions=1)
    faces = m.faces.copy()
    faces[0, 1] = faces[0, 0]
    bad = SurfaceMesh(points=m.points, faces=faces)
    assert bad.degenerate_faces().sum() == 1
