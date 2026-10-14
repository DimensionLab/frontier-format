# SPDX-License-Identifier: Apache-2.0
"""In-memory triangle surface mesh with point and cell fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

FloatArray = np.ndarray


def as_field(values: np.ndarray) -> np.ndarray:
    """Return a float32 (N, C) view of a field array (1-D arrays become (N, 1))."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(f"field must be 1-D or 2-D, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


class FieldDict(dict[str, np.ndarray]):
    """``dict`` that normalises every assigned array to float32 ``(N, C)``."""

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        super().__setitem__(key, as_field(value))

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


@dataclass
class SurfaceMesh:
    """Triangulated surface with float32 points, int32 faces and float32 fields.

    ``point_fields`` are (n_points, C), ``cell_fields`` are (n_faces, C). ``cell_area`` is the
    per-triangle area; it is computed lazily from the geometry when not provided.
    ``face_parent`` maps each triangle to the polygon it came from in the source file (or
    ``None`` when the source was already triangulated).
    """

    points: np.ndarray
    faces: np.ndarray
    point_fields: dict[str, np.ndarray] = field(default_factory=dict)
    cell_fields: dict[str, np.ndarray] = field(default_factory=dict)
    cell_area: np.ndarray | None = None
    face_parent: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(np.asarray(self.points, dtype=np.float32))
        self.faces = np.ascontiguousarray(np.asarray(self.faces, dtype=np.int32))
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {self.points.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3) triangles, got {self.faces.shape}")
        pf, cf = FieldDict(), FieldDict()
        pf.update(self.point_fields)
        cf.update(self.cell_fields)
        self.point_fields, self.cell_fields = pf, cf
        for name, arr in self.point_fields.items():
            if arr.shape[0] != self.n_points:
                raise ValueError(f"point field {name!r} has {arr.shape[0]} rows != {self.n_points}")
        for name, arr in self.cell_fields.items():
            if arr.shape[0] != self.n_faces:
                raise ValueError(f"cell field {name!r} has {arr.shape[0]} rows != {self.n_faces}")
        if self.cell_area is not None:
            self.cell_area = np.asarray(self.cell_area, dtype=np.float32).reshape(-1)
            if self.cell_area.shape[0] != self.n_faces:
                raise ValueError("cell_area length does not match number of faces")

    # -- basic geometry -----------------------------------------------------------------

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def triangle_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.points.astype(np.float64)
        a = p[self.faces[:, 0]]
        return p[self.faces[:, 1]] - a, p[self.faces[:, 2]] - a

    def face_normals(self, unit: bool = True) -> np.ndarray:
        """Per-face normals following the face winding (float64, (M, 3))."""
        e1, e2 = self.triangle_vectors()
        n = np.cross(e1, e2)
        if unit:
            length = np.linalg.norm(n, axis=1)
            length[length == 0] = 1.0
            n = n / length[:, None]
        return n

    def face_areas(self) -> np.ndarray:
        """Per-face areas from geometry (float64, (M,))."""
        e1, e2 = self.triangle_vectors()
        return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    def areas(self) -> np.ndarray:
        """``cell_area`` if present, else computed face areas (float32)."""
        if self.cell_area is None:
            self.cell_area = self.face_areas().astype(np.float32)
        return self.cell_area

    @property
    def area(self) -> float:
        return float(np.sum(self.areas(), dtype=np.float64))

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        if self.n_points == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        lo = self.points.min(axis=0)
        hi = self.points.max(axis=0)
        return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))

    def degenerate_faces(self) -> np.ndarray:
        """Boolean mask of faces with repeated vertex indices or zero area."""
        f = self.faces
        repeated = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
        return repeated | (self.face_areas() <= 0.0)

    # -- field conversion ---------------------------------------------------------------

    def vertex_areas(self) -> np.ndarray:
        """Area lumped to vertices (one third of each incident face)."""
        a = self.face_areas() / 3.0
        out = np.zeros(self.n_points, dtype=np.float64)
        for k in range(3):
            out += np.bincount(self.faces[:, k], weights=a, minlength=self.n_points)
        return out

    def cell_to_point(self, values: np.ndarray) -> np.ndarray:
        """Area-weighted average of a cell field onto vertices, (N, C) float32."""
        v = as_field(values).astype(np.float64)
        a = self.face_areas()
        wsum = np.zeros(self.n_points, dtype=np.float64)
        out = np.zeros((self.n_points, v.shape[1]), dtype=np.float64)
        for k in range(3):
            idx = self.faces[:, k]
            wsum += np.bincount(idx, weights=a, minlength=self.n_points)
            for c in range(v.shape[1]):
                out[:, c] += np.bincount(idx, weights=a * v[:, c], minlength=self.n_points)
        wsum[wsum == 0] = 1.0
        return (out / wsum[:, None]).astype(np.float32)

    def point_to_cell(self, values: np.ndarray) -> np.ndarray:
        """Mean of the three vertex values per face, (M, C) float32."""
        v = as_field(values)
        return ((v[self.faces[:, 0]] + v[self.faces[:, 1]] + v[self.faces[:, 2]]) / 3.0).astype(
            np.float32
        )

    def with_point_fields_from_cells(self, names: list[str] | None = None) -> SurfaceMesh:
        """Add point fields converted from cell fields where no point field exists yet."""
        for name in names or list(self.cell_fields):
            if name not in self.point_fields:
                self.point_fields[name] = self.cell_to_point(self.cell_fields[name])
        return self

    # -- interop ------------------------------------------------------------------------

    def to_pyvista(self) -> pv.PolyData:
        import pyvista as pv

        poly = pv.PolyData.from_regular_faces(self.points, self.faces)
        for name, arr in self.point_fields.items():
            poly.point_data[name] = arr if arr.shape[1] > 1 else arr[:, 0]
        for name, arr in self.cell_fields.items():
            poly.cell_data[name] = arr if arr.shape[1] > 1 else arr[:, 0]
        return poly

    def copy(self) -> SurfaceMesh:
        return SurfaceMesh(
            points=self.points.copy(),
            faces=self.faces.copy(),
            point_fields={k: v.copy() for k, v in self.point_fields.items()},
            cell_fields={k: v.copy() for k, v in self.cell_fields.items()},
            cell_area=None if self.cell_area is None else self.cell_area.copy(),
            face_parent=None if self.face_parent is None else self.face_parent.copy(),
        )


def make_box_mesh(
    size: tuple[float, float, float] = (1.0, 0.4, 0.3), subdivisions: int = 2
) -> SurfaceMesh:
    """Small synthetic closed box surface for tests and demos (no I/O, no VTK needed)."""
    sx, sy, sz = size
    n = max(1, subdivisions)
    pts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def add_face(origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
        base = len(pts)
        for i in range(n + 1):
            for j in range(n + 1):
                p = origin + u * (i / n) + v * (j / n)
                pts.append((float(p[0]), float(p[1]), float(p[2])))
        for i in range(n):
            for j in range(n):
                a = base + i * (n + 1) + j
                b = a + 1
                c = a + (n + 1)
                d = c + 1
                faces.append((a, c, b))
                faces.append((b, c, d))

    o = np.array([-sx / 2, -sy / 2, 0.0])
    ex, ey, ez = np.array([sx, 0, 0.0]), np.array([0, sy, 0.0]), np.array([0, 0, sz])
    add_face(o, ey, ex)  # bottom (normal -z)
    add_face(o + ez, ex, ey)  # top (+z)
    add_face(o, ex, ez)  # front y=-sy/2 (normal -y)
    add_face(o + ey, ez, ex)  # back (+y)
    add_face(o, ez, ey)  # x = -sx/2 (normal -x)
    add_face(o + ex, ey, ez)  # x = +sx/2 (+x)
    mesh = SurfaceMesh(points=np.array(pts, dtype=np.float32), faces=np.array(faces, np.int32))
    # Synthetic "pressure": stagnation on the front face (x min), suction behind.
    cx = mesh.points[mesh.faces].mean(axis=1)[:, 0]
    p = (0.5 - (cx - cx.min()) / max(sx, 1e-9)).astype(np.float32)
    mesh.cell_fields["p"] = p
    mesh.cell_fields["wallShearStress"] = np.tile(
        np.array([-1e-3, 0.0, 0.0], np.float32), (mesh.n_faces, 1)
    )
    mesh.with_point_fields_from_cells()
    return mesh
