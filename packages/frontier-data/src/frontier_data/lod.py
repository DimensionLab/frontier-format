# SPDX-License-Identifier: Apache-2.0
"""Level-of-detail generation for surface meshes.

Levels follow ``docs/spec/dataset-format.md``: 0 = full resolution, 1 ~ 100k faces, 2 ~ 20k, 3 ~ 5k.
Decimation uses ``fast_simplification`` (quadric edge collapse, ~2 s for 2M faces) and transfers
point fields through the vertex-collapse map (area-weighted mean of the collapsed vertices).
Levels are built as a cascade (0 -> 1 -> 2 -> 3) which is both faster and smoother than
decimating the full mesh for every level. When ``fast_simplification`` is unavailable the code
falls back to PyVista's ``decimate_pro`` which preserves retained vertices and their point data.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from frontier_data.mesh import SurfaceMesh

log = logging.getLogger(__name__)

LOD_TARGET_FACES: dict[int, int | None] = {0: None, 1: 100_000, 2: 20_000, 3: 5_000}
LogFn = Callable[[str], None]


@dataclass(frozen=True)
class LodTiming:
    level: int
    n_faces: int
    n_points: int
    seconds: float
    method: str


def _transfer_point_fields(
    src: SurfaceMesh, new_points: np.ndarray, k: int = 4
) -> dict[str, np.ndarray]:
    """Sample each source point field at the new vertex positions.

    Inverse-distance weighting over the ``k`` nearest source vertices. This is robust to
    quadric-collapse vertex placement and never mixes values across far-apart regions,
    which the collapse-map averaging did (visible as star artifacts on decimated LODs).
    """
    from scipy.spatial import KDTree

    tree = KDTree(src.points.astype(np.float64))
    dist, idx = tree.query(new_points.astype(np.float64), k=k)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    w = 1.0 / np.maximum(dist, 1e-12)
    exact = dist[:, 0] <= 1e-9
    w[exact] = 0.0
    w[exact, 0] = 1.0
    w /= w.sum(axis=1, keepdims=True)
    out: dict[str, np.ndarray] = {}
    for name, arr in src.point_fields.items():
        vals = arr.astype(np.float64)[idx]  # (n_new, k, C)
        out[name] = np.einsum("nk,nkc->nc", w, vals).astype(np.float32)
    return out


def _finish(
    src: SurfaceMesh, points: np.ndarray, faces: np.ndarray, pf: dict[str, np.ndarray]
) -> SurfaceMesh:
    out = SurfaceMesh(points=points, faces=faces, point_fields=pf)
    out.cell_fields = {name: out.point_to_cell(pf[name]) for name in src.cell_fields if name in pf}
    out.areas()
    return out


def merge_coincident_vertices(mesh: SurfaceMesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (points, faces) with exactly coincident vertices merged.

    Solver exports (OpenFOAM patches, Fluent zones, ...) write each patch with its own
    copy of the seam vertices, so the triangle soup has thousands of open boundary edges.
    Edge-collapse decimators treat those seams as holes and tear the surface open; merging
    first keeps the decimated LODs watertight and area-preserving.
    """
    pts, inverse = np.unique(mesh.points, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[mesh.faces].astype(np.int32)
    # Drop triangles that collapsed to a line/point after merging.
    keep = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    return pts.astype(np.float32), faces[keep]


def _drop_degenerate(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Remove triangles with repeated indices or (numerically) zero area."""
    distinct = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    tri = points[faces].astype(np.float64)
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return faces[distinct & (area2 > 1e-18)]


def decimate_fast(mesh: SurfaceMesh, target_faces: int, aggressiveness: int = 7) -> SurfaceMesh:
    import fast_simplification as fs

    points, faces = merge_coincident_vertices(mesh)
    result: Any = fs.simplify(
        points,
        faces,
        target_count=int(target_faces),
        agg=aggressiveness,
    )
    pts, faces = result[0], result[1]
    pts = np.asarray(pts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    faces = _drop_degenerate(pts, faces)
    pf = _transfer_point_fields(mesh, pts)
    return _finish(mesh, pts, faces, pf)


def decimate_pyvista(mesh: SurfaceMesh, target_faces: int) -> SurfaceMesh:
    """Fallback: ``decimate_pro`` keeps a subset of the original vertices and their point data."""
    from frontier_data.readers.vtk import polydata_to_mesh

    reduction = 1.0 - float(target_faces) / max(mesh.n_faces, 1)
    poly = mesh.to_pyvista()
    for name in list(poly.cell_data.keys()):
        del poly.cell_data[name]
    dec = poly.decimate_pro(reduction, preserve_topology=False)
    out = polydata_to_mesh(dec)
    pf = {k: v for k, v in out.point_fields.items() if k in mesh.point_fields}
    return _finish(mesh, out.points, out.faces, pf)


def decimate(mesh: SurfaceMesh, target_faces: int, method: str = "auto") -> tuple[SurfaceMesh, str]:
    """Decimate ``mesh`` to roughly ``target_faces`` triangles, returning (mesh, method_used)."""
    if target_faces >= mesh.n_faces:
        return mesh.copy(), "copy"
    if method in ("auto", "fast"):
        try:
            return decimate_fast(mesh, target_faces), "fast_simplification"
        except ImportError:
            if method == "fast":
                raise
            log.warning("fast_simplification not installed, falling back to decimate_pro")
    return decimate_pyvista(mesh, target_faces), "decimate_pro"


def build_lods(
    mesh: SurfaceMesh,
    levels: Iterable[int] = (0, 1, 2, 3),
    *,
    targets: dict[int, int | None] | None = None,
    method: str = "auto",
    log_fn: LogFn | None = None,
) -> tuple[dict[int, SurfaceMesh], list[LodTiming]]:
    """Build the requested LOD levels as a cascade. Level 0 is the input mesh itself."""
    targets = targets or LOD_TARGET_FACES
    wanted = sorted(set(int(level) for level in levels))
    out: dict[int, SurfaceMesh] = {}
    timings: list[LodTiming] = []
    mesh.areas()
    if 0 in wanted:
        out[0] = mesh
        timings.append(LodTiming(0, mesh.n_faces, mesh.n_points, 0.0, "source"))
    current = mesh
    # cascade through every level up to the highest requested, keeping only the wanted ones;
    # the recorded time of a wanted level includes the skipped intermediate levels
    pending = 0.0
    for level in range(1, max(wanted, default=0) + 1):
        target = targets.get(level)
        if target is None:
            continue
        t0 = time.perf_counter()
        current, used = decimate(current, target, method=method)
        dt = time.perf_counter() - t0
        pending += dt
        if level in wanted:
            out[level] = current
            timings.append(LodTiming(level, current.n_faces, current.n_points, pending, used))
            pending = 0.0
        msg = (
            f"lod {level}: {current.n_faces} faces / {current.n_points} points "
            f"in {dt:.2f}s ({used})"
        )
        log.info(msg)
        if log_fn:
            log_fn(msg)
    return out, timings


# -- mesh kind: vertex-clustering levels as vertex maps ---------------------------------------
#
# A ``mesh.zarr`` keeps its fields at full resolution (time first, one step per chunk); a
# level of detail for it is a *vertex map*: for every kept vertex the index of the full
# resolution vertex that represents it, plus triangles over the kept vertices. A viewer
# gathers any step of any field (and ``points_t`` of a moving mesh) through the map, so
# nothing per step is stored twice. Written as ``lod/<level>/{points, faces, vertex_map}``;
# the server builds the same thing on the fly when a store has none.

MESH_LOD_TARGETS: dict[int, int] = {1: 100_000, 2: 20_000, 3: 5_000}

_VTK_TRIANGLE, _VTK_QUAD, _VTK_TETRA, _VTK_HEX, _VTK_WEDGE, _VTK_PYRAMID = 5, 9, 10, 12, 13, 14
_VTK_PIXEL, _VTK_VOXEL = 8, 11
# Face templates of the 3D cells (outward-ish order is irrelevant: boundary detection sorts).
_CELL_FACES: dict[int, list[list[int]]] = {
    _VTK_TETRA: [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
    _VTK_HEX: [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]],
    _VTK_VOXEL: [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [1, 3, 7, 5],
        [3, 2, 6, 7],
        [2, 0, 4, 6],
    ],
    _VTK_WEDGE: [[0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
    _VTK_PYRAMID: [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
}


def drawable_triangles(cells: np.ndarray, cell_types: np.ndarray, point_dim: int) -> np.ndarray:
    """The triangles a viewer draws for a volume mesh: 2D cells triangulated, 3D cells
    reduced to their boundary faces (faces owned by exactly one cell), quads split in two.
    Same rule as ``frontier_data::mesh::VolumeMesh::surface_faces`` in Rust."""
    cells = np.asarray(cells, dtype=np.int64)
    cell_types = np.asarray(cell_types).reshape(-1)
    tris: list[np.ndarray] = []
    polys: list[np.ndarray] = []  # candidate 3D faces, padded to 4 with -1
    for ty in np.unique(cell_types):
        rows = cells[cell_types == ty]
        if rows.size == 0:
            continue
        if ty == _VTK_TRIANGLE:
            tris.append(rows[:, :3])
        elif ty in (_VTK_QUAD, _VTK_PIXEL):
            q = rows[:, :4] if ty == _VTK_QUAD else rows[:, [0, 1, 3, 2]]
            tris.append(q[:, [0, 1, 2]])
            tris.append(q[:, [0, 2, 3]])
        elif ty in _CELL_FACES:
            for tpl in _CELL_FACES[ty]:
                f = rows[:, tpl]
                if f.shape[1] == 3:
                    f = np.concatenate([f, np.full((len(f), 1), -1, dtype=np.int64)], axis=1)
                polys.append(f)
        elif point_dim == 2:
            # Unknown 2D polygon: fan-triangulate the valid nodes.
            for row in rows:
                nodes = row[row >= 0]
                for k in range(1, len(nodes) - 1):
                    tris.append(np.array([[nodes[0], nodes[k], nodes[k + 1]]], dtype=np.int64))
    if polys:
        allp = np.concatenate(polys, axis=0)
        key = np.sort(allp, axis=1)
        _, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
        boundary = allp[first[counts == 1]]
        quad = boundary[:, 3] >= 0
        tris.append(boundary[~quad][:, :3])
        q = boundary[quad]
        tris.append(q[:, [0, 1, 2]])
        tris.append(q[:, [0, 2, 3]])
    if not tris:
        return np.zeros((0, 3), dtype=np.int32)
    out = np.concatenate(tris, axis=0).astype(np.int32)
    ok = (out[:, 0] != out[:, 1]) & (out[:, 1] != out[:, 2]) & (out[:, 0] != out[:, 2])
    return out[ok]


def _cluster_once(
    points: np.ndarray, faces: np.ndarray, cell: float
) -> tuple[np.ndarray, np.ndarray]:
    lo = points.min(axis=0)
    ijk = np.floor((points - lo) / cell).astype(np.int64)
    _, inv = np.unique(ijk, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n_clusters = int(inv.max()) + 1 if inv.size else 0
    # Representative: the vertex nearest to its cluster's centroid.
    sums = np.zeros((n_clusters, 3), dtype=np.float64)
    np.add.at(sums, inv, points.astype(np.float64))
    counts = np.bincount(inv, minlength=n_clusters).astype(np.float64)
    centroid = sums / np.maximum(counts, 1)[:, None]
    d = ((points.astype(np.float64) - centroid[inv]) ** 2).sum(axis=1)
    order = np.lexsort((d, inv))
    first = np.searchsorted(inv[order], np.arange(n_clusters))
    vertex_map = order[first].astype(np.int32)
    f = inv[faces]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    if len(f):
        _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
        f = f[np.sort(keep)]
    return vertex_map, f.astype(np.int32)


def cluster_vertex_map(
    points: np.ndarray, faces: np.ndarray, target_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering decimation of a triangle surface to about ``target_faces``.

    Returns ``(vertex_map, faces)``: ``vertex_map[i]`` is the full-resolution index of kept
    vertex ``i`` (the vertex nearest to its cluster's centroid, so positions and fields are
    sampled, never averaged across far-apart regions), ``faces`` are triangles over the
    kept vertices. A mesh at or below the target is returned whole (identity map).
    """
    points = np.asarray(points, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if points.ndim == 2 and points.shape[1] == 2:
        points = np.concatenate([points, np.zeros((len(points), 1), np.float32)], axis=1)
    if len(faces) <= target_faces or len(faces) == 0:
        return np.arange(len(points), dtype=np.int32), faces
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) or 1.0
    cell = extent / np.sqrt(target_faces)
    best = _cluster_once(points, faces, cell)
    for _ in range(6):
        ratio = len(best[1]) / target_faces
        if 0.7 <= ratio <= 1.25:
            break
        cell *= float(np.clip(np.sqrt(ratio), 0.5, 2.0))
        nxt = _cluster_once(points, faces, cell)
        if len(nxt[1]) == 0:
            break
        best = nxt
    return best


def build_mesh_lod_maps(
    points: np.ndarray,
    faces: np.ndarray,
    levels: Iterable[int] = (1, 2, 3),
    *,
    targets: dict[int, int] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{level: (vertex_map, faces)}`` for every requested level above 0 whose target is
    below the surface's face count (a level that would keep the mesh whole is skipped)."""
    targets = targets or MESH_LOD_TARGETS
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for level in sorted(set(int(level) for level in levels)):
        target = targets.get(level)
        if level == 0 or target is None or len(faces) <= target:
            continue
        t0 = time.perf_counter()
        vm, f = cluster_vertex_map(points, faces, target)
        log.info(
            "mesh lod %d: %d faces / %d points in %.2fs (vertex clustering)",
            level,
            len(f),
            len(vm),
            time.perf_counter() - t0,
        )
        out[level] = (vm, f)
    return out


def write_mesh_lods(
    root: Any, points: np.ndarray, maps: dict[int, tuple[np.ndarray, np.ndarray]]
) -> list[int]:
    """Write ``lod/<level>/{points, faces, vertex_map}`` into an open ``mesh.zarr`` group
    (``points`` are the full-resolution step-0 points). Returns the levels present."""
    from frontier_data.dataset import _write_array

    if "lod" in root:
        del root["lod"]
    if not maps:
        root.attrs["lod_levels"] = [0]
        return [0]
    lod = root.create_group("lod")
    pts = np.asarray(points, dtype=np.float32)
    for level, (vm, f) in sorted(maps.items()):
        g = lod.create_group(str(level))
        _write_array(g, "points", pts[vm], dims=["point", "space"])
        _write_array(g, "faces", np.asarray(f, dtype=np.int32), dims=["face", "corner"])
        _write_array(g, "vertex_map", np.asarray(vm, dtype=np.int32), dims=["point"])
    levels = [0, *sorted(maps)]
    root.attrs["lod_levels"] = levels
    return levels


def add_mesh_lods(
    dataset_root: str | Path,
    levels: Iterable[int] = (1, 2, 3),
    *,
    sample_ids: Iterable[str] | None = None,
    force: bool = False,
    log_fn: LogFn | None = None,
) -> dict[str, list[int]]:
    """Add vertex-clustering LOD maps to the ``mesh.zarr`` of every (or the named) sample
    of an existing dataset; ``python -m frontier_data mesh-lods <dataset>``. Samples that
    already have levels are skipped unless ``force``. Returns ``{sample_id: levels}``."""
    import zarr

    root = Path(dataset_root)
    manifest = json.loads((root / "manifest.json").read_text())
    wanted = set(sample_ids) if sample_ids is not None else None
    out: dict[str, list[int]] = {}
    for entry in manifest.get("samples", []):
        sid = entry["id"]
        if wanted is not None and sid not in wanted:
            continue
        store_path = root / entry["path"] / "mesh.zarr"
        if not store_path.is_dir():
            continue
        g: Any = zarr.open_group(str(store_path), mode="r+")
        existing = list(g.attrs.get("lod_levels", [0]))
        if len(existing) > 1 and not force:
            out[sid] = existing
            continue
        points = np.asarray(g["points"][...], dtype=np.float32)
        cells = np.asarray(g["cells"][...])
        ctypes = np.asarray(g["cell_types"][...]) if "cell_types" in g else None
        if ctypes is None:
            from frontier_data.arrays import default_cell_types

            ctypes = default_cell_types(cells, points.shape[1])
        faces = drawable_triangles(cells, ctypes, points.shape[1])
        maps = build_mesh_lod_maps(points, faces, levels)
        present = write_mesh_lods(g, points, maps)
        entry["lod_levels"] = present
        # the sample's own metadata follows (atomic: the file may be a versioned hardlink)
        meta_path = root / entry["path"] / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            if meta.get("lod_levels") != present:
                meta["lod_levels"] = present
                tmp = meta_path.with_name(meta_path.name + ".tmp")
                tmp.write_text(json.dumps(meta, indent=1))
                tmp.replace(meta_path)
        out[sid] = present
        msg = f"{sid}: {len(faces)} triangles -> levels {present}" + "".join(
            f", lod {lvl}: {len(f)} faces" for lvl, (_, f) in sorted(maps.items())
        )
        log.info(msg)
        if log_fn:
            log_fn(msg)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out
