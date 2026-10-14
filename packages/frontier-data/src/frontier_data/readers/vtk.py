# SPDX-License-Identifier: Apache-2.0
"""VTK XML PolyData (.vtp), STL and scalar-CSV readers built on PyVista.

``read_vtp`` triangulates polygons and keeps a ``face_parent`` map so native per-polygon data
(for example AhmedML's ``boundary_cell_area_<i>.npy``) can be related to the triangles.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from frontier_data.mesh import SurfaceMesh

log = logging.getLogger(__name__)

_PARENT_KEY = "__frontier_parent_cell"


def polydata_to_mesh(poly: Any, *, triangulate: bool = True) -> SurfaceMesh:
    """Convert a ``pyvista.PolyData`` into a :class:`SurfaceMesh` (triangulating if needed)."""
    import pyvista as pv

    poly = pv.PolyData(poly) if not isinstance(poly, pv.PolyData) else poly
    face_parent: np.ndarray | None = None
    if poly.n_cells and not poly.is_all_triangles:
        if not triangulate:
            raise ValueError("mesh has non-triangle polygons; pass triangulate=True")
        poly = poly.copy(deep=False)
        poly.cell_data[_PARENT_KEY] = np.arange(poly.n_cells, dtype=np.int64)
        poly = poly.triangulate()
        face_parent = np.asarray(poly.cell_data.pop(_PARENT_KEY), dtype=np.int64)
    if poly.n_cells and not poly.is_all_triangles:
        raise ValueError("triangulation failed to produce an all-triangle mesh")
    faces = (
        np.asarray(poly.regular_faces, dtype=np.int32)
        if poly.n_cells
        else np.zeros((0, 3), np.int32)
    )
    points = np.asarray(poly.points, dtype=np.float32)
    point_fields = {
        str(k): np.asarray(poly.point_data[k])
        for k in poly.point_data.keys()
        if not str(k).startswith("vtk")
    }
    cell_fields = {
        str(k): np.asarray(poly.cell_data[k])
        for k in poly.cell_data.keys()
        if not str(k).startswith("vtk")
    }
    return SurfaceMesh(
        points=points,
        faces=faces,
        point_fields=point_fields,
        cell_fields=cell_fields,
        face_parent=face_parent,
    )


def read_vtp(path: str | Path, *, point_fields_from_cells: bool = True) -> SurfaceMesh:
    """Read a VTK XML PolyData file (ascii, binary or appended) into a :class:`SurfaceMesh`.

    Cell fields are kept as-is; when ``point_fields_from_cells`` is true, every cell field without
    a point-data counterpart is also converted to an area-weighted point field.
    """
    import pyvista as pv

    t0 = time.perf_counter()
    poly = pv.read(str(path))
    if not isinstance(poly, pv.PolyData):
        poly = poly.extract_surface()
    mesh = polydata_to_mesh(poly)
    if point_fields_from_cells:
        mesh.with_point_fields_from_cells()
    log.info(
        "read %s: %d points, %d faces, fields=%s in %.2fs",
        path,
        mesh.n_points,
        mesh.n_faces,
        sorted(mesh.cell_fields),
        time.perf_counter() - t0,
    )
    return mesh


def read_stl(path: str | Path) -> SurfaceMesh:
    import pyvista as pv

    poly = pv.read(str(path))
    if not isinstance(poly, pv.PolyData):
        poly = poly.extract_surface()
    return polydata_to_mesh(poly)


def read_scalar_csv(path: str | Path) -> dict[str, float]:
    """Read a one-row CSV (``name1, name2\\n v1, v2``) into ``{name: float}``.

    Headers and values are stripped; non-numeric values are kept as NaN.
    """
    with Path(path).open(newline="") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError(f"{path}: expected a header row and a value row")
    header = [h.strip() for h in rows[0]]
    values = [v.strip() for v in rows[1]]
    out: dict[str, float] = {}
    for name, val in zip(header, values, strict=False):
        try:
            out[name] = float(val)
        except ValueError:
            out[name] = float("nan")
    return out


def read_cell_area_npy(path: str | Path) -> np.ndarray:
    arr = np.load(str(path))
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def distribute_parent_areas(mesh: SurfaceMesh, parent_area: np.ndarray) -> np.ndarray:
    """Split native polygon areas over their triangles proportionally to geometric area.

    Returns a per-triangle float32 array whose per-parent sums equal ``parent_area``.
    """
    if mesh.face_parent is None:
        if parent_area.shape[0] != mesh.n_faces:
            raise ValueError("parent_area length does not match faces and no parent map exists")
        return parent_area.astype(np.float32)
    geo = mesh.face_areas()
    parent_geo = np.bincount(mesh.face_parent, weights=geo, minlength=parent_area.shape[0])
    parent_geo[parent_geo == 0] = 1.0
    frac = geo / parent_geo[mesh.face_parent]
    return (frac * parent_area[mesh.face_parent]).astype(np.float32)
