# SPDX-License-Identifier: Apache-2.0
"""Geometry ops: topology audit, seam merge, repair, orientation, transform/units, LOD
cascade, point resampling, part ids. All pure numpy/scipy over ``points``/``faces``; every
op reports geometric invariants (area, bounds, open edges, components) before and after."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from frontier_data.lod import LOD_TARGET_FACES, build_lods
from frontier_data.mesh import SurfaceMesh
from frontier_prep import kernels
from frontier_prep import vocabulary as vocab
from frontier_prep.base import BaseOp, DatasetRef, SampleOutcome
from frontier_prep.io import SampleEditor
from frontier_prep.op import Assertion, Change, OpWarning, Params, register

# -- topology helpers ----------------------------------------------------------------------------
#
# Every kernel below is pure numpy/scipy over int64 face indices and is bitwise deterministic:
# edges are sorted by a packed integer key, ties by slot index (stable), components are
# relabelled by first appearance, orientation follows a BFS from the lowest face index of each
# component with neighbours in ascending face order. The Rust port spec
# describes the same rules; the regression test compares against the pre-optimisation code.
#
# The ``*_py`` functions are the reference. The public names (``topology``,
# ``merge_vertices``, ``orient_consistently``, ``duplicate_faces``, ``face_areas``,
# ``TopologyTables.build``) run the Rust port of the same rule when ``frontier_prep.kernels``
# selects it and return the same bytes (``tests/test_kernels_native.py``); input the native
# kernel refuses goes to the reference, which also decides the error.


def _native_faces(faces: Any) -> np.ndarray | None:
    """``faces`` as the C-contiguous int64 ``(M, 3)`` array the native kernels take, or
    ``None`` when it is not an integer ``(M, 3)`` array."""
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu":
        return None
    return np.ascontiguousarray(f, dtype=np.int64)


def _native_points(points: Any) -> np.ndarray | None:
    """``points`` as a C-contiguous float32/float64 ``(N, 3)`` array, or ``None``."""
    if not isinstance(points, np.ndarray):
        return None
    if points.ndim != 2 or points.shape[1] != 3 or not points.dtype.isnative:
        return None
    if points.dtype not in (np.float32, np.float64):
        return None
    return np.ascontiguousarray(points)


@dataclass(frozen=True)
class TopologyTables:
    """Sorted-edge tables of a triangle mesh, built once per face array and shared by the
    topology audit, component labelling, orientation and boundary-loop kernels.

    Slot ``k`` of the ``3M`` directed edge slots belongs to face ``k % M`` and edge slot
    ``k // M`` (slot 0 is ``(f0, f1)``, 1 is ``(f1, f2)``, 2 is ``(f2, f0)``). ``order`` sorts
    the slots by the packed undirected key ``min(u, v) * n + max(u, v)`` and, within one edge,
    by slot index (stable), so the incidence of edge ``e`` is the contiguous run
    ``order[start[e] : start[e] + counts[e]]`` and consecutive entries of a run are faces
    that share that edge.
    """

    faces: np.ndarray
    """(M, 3) int64 face indices."""
    n: int
    """Packing base: ``max(faces) + 1``."""
    order: np.ndarray
    """(3M,) int64: slot indices sorted by (edge key, slot index)."""
    inv: np.ndarray
    """(3M,) int64: edge id of each slot (``inv[k]``), ids are ranks of the sorted keys."""
    start: np.ndarray
    """(E,) int64: first position of each edge's run in ``order``."""
    counts: np.ndarray
    """(E,) int64: faces incident to each edge (1 open, 2 manifold, >2 non-manifold)."""
    uniq: np.ndarray
    """(E, 2) int64: undirected edges ``(min, max)`` in key order."""

    @classmethod
    def build(cls, faces: np.ndarray) -> TopologyTables:
        nk = kernels.native()
        f = _native_faces(faces) if nk is not None else None
        if nk is not None and f is not None and len(f):
            try:
                order, inv, start, counts, uniq = nk.edge_tables(f)
            except ValueError:  # negative indices: the reference decides
                pass
            else:
                return cls(f, int(f.max()) + 1, order, inv, start, counts, uniq)
        return cls.build_py(faces)

    @classmethod
    def build_py(cls, faces: np.ndarray) -> TopologyTables:
        """The numpy reference of :meth:`build`."""
        f = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
        m = len(f)
        if m == 0:
            z = np.zeros(0, dtype=np.int64)
            return cls(f, 1, z, z, z, z, np.zeros((0, 2), dtype=np.int64))
        n = int(f.max()) + 1
        e0 = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        e1 = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
        key = np.minimum(e0, e1) * n + np.maximum(e0, e1)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        new = np.empty(3 * m, dtype=bool)
        new[0] = True
        np.not_equal(ks[1:], ks[:-1], out=new[1:])
        start = np.flatnonzero(new)
        counts = np.diff(np.append(start, 3 * m))
        eid_sorted = np.cumsum(new) - 1
        inv = np.empty(3 * m, dtype=np.int64)
        inv[order] = eid_sorted
        ukey = ks[start]
        uniq = np.column_stack([ukey // n, ukey % n])
        return cls(f, n, order, inv, start, counts, uniq)

    @property
    def n_faces(self) -> int:
        return int(len(self.faces))

    @property
    def n_edges(self) -> int:
        return int(len(self.uniq))

    @property
    def boundary(self) -> np.ndarray:
        """(E,) bool: edges with exactly one incident face."""
        return self.counts == 1

    @cached_property
    def face_s(self) -> np.ndarray:
        """(3M,) face index of each sorted slot."""
        return self.order % self.n_faces

    @cached_property
    def slot_s(self) -> np.ndarray:
        """(3M,) edge slot (0..2) of each sorted slot."""
        return self.order // self.n_faces

    @cached_property
    def eid_s(self) -> np.ndarray:
        """(3M,) edge id of each sorted slot (non-decreasing)."""
        return self.inv[self.order]

    def forward_s(self) -> np.ndarray:
        """(3M,) bool per sorted slot: the face traverses the edge from ``min`` to ``max``."""
        return self.faces[self.face_s, self.slot_s] == self.uniq[self.eid_s, 0]

    def inconsistent_edges(self) -> int:
        """Number of directed edges used more than once (two faces traverse an edge the same
        way); equals the count of duplicated rows of the directed edge list."""
        fwd = self.forward_s()
        n_fwd = np.bincount(self.eid_s, weights=fwd, minlength=self.n_edges)
        n_bwd = self.counts - n_fwd
        return int((n_fwd > 1).sum() + (n_bwd > 1).sum())

    def used_points(self) -> int:
        """Number of distinct vertex indices referenced by the faces."""
        return int(np.count_nonzero(np.bincount(self.faces.ravel(), minlength=self.n)))

    def components(self) -> np.ndarray:
        """Face components (faces sharing an edge), labels (M,) by first appearance."""
        return _components_from_slots(self.n_faces, self.inv)

    def manifold_pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Faces ``(a, b)`` across each manifold edge (count 2) and their edge slots
        ``(sa, sb)``; ``a`` is the slot with the lower slot index."""
        eid_s = self.eid_s
        same = (eid_s[1:] == eid_s[:-1]) & (self.counts[eid_s[1:]] == 2)
        face_s, slot_s = self.face_s, self.slot_s
        return face_s[:-1][same], face_s[1:][same], slot_s[:-1][same], slot_s[1:][same]


def _edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(undirected unique edges (E,2), inverse (3M,), counts (E,)) of a triangle mesh."""
    t = TopologyTables.build(faces)
    return t.uniq, t.inv, t.counts


def _components_from_slots(m: int, inv: np.ndarray) -> np.ndarray:
    """Connected components of faces through shared slot ids, labels by first appearance.
    ``inv`` may carry any non-negative ids (cut edges get fresh ones)."""
    nk = kernels.native()
    if nk is not None and m:
        try:
            labels = nk.components_from_slots(m, np.ascontiguousarray(inv, dtype=np.int64))
        except ValueError:
            pass
        else:
            return labels.astype(np.int32)  # scipy's label dtype, which the reference returns
    return components_from_slots_py(m, inv)


def components_from_slots_py(m: int, inv: np.ndarray) -> np.ndarray:
    """The reference of ``_components_from_slots``: a bipartite face/edge graph in CSR form
    needs no sort (face rows have exactly three entries)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    if m == 0:
        return np.zeros(0, dtype=np.int64)
    ne = int(inv.max()) + 1
    indptr = np.concatenate([np.arange(0, 3 * m + 1, 3), np.full(ne, 3 * m)])
    indices = m + inv.reshape(3, m).T.reshape(-1)
    g = sparse.csr_matrix((np.ones(3 * m, dtype=np.int8), indices, indptr), shape=(m + ne, m + ne))
    _, labels = connected_components(g, directed=False)
    labels = labels[:m]
    _, first = np.unique(labels, return_index=True)
    remap = np.empty_like(labels)
    remap[labels[np.sort(first)]] = np.arange(len(first))
    return remap[labels]


def _components(n_points: int, faces: np.ndarray, edges: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """Connected components over faces (faces sharing an edge), labels (M,)."""
    return _components_from_slots(len(faces), inv)


def topology(
    points: np.ndarray, faces: np.ndarray, tables: TopologyTables | None = None
) -> dict[str, Any]:
    """Open edges, non-manifold edges, components, Euler characteristic, orientation
    consistency, degenerate faces, area, bounds. ``tables`` (built from the same faces)
    is reused by the reference when given; the native kernel builds its own."""
    nk = kernels.native()
    if nk is not None:
        p, f = _native_points(points), _native_faces(faces)
        if p is not None and f is not None:
            try:
                return nk.topology(p, f)
            except ValueError:  # indices outside the point array: the reference decides
                pass
    return topology_py(points, faces, tables)


def topology_py(
    points: np.ndarray, faces: np.ndarray, tables: TopologyTables | None = None
) -> dict[str, Any]:
    """The numpy reference of :func:`topology`."""
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return {"n_points": int(len(points)), "n_faces": 0}
    t = tables if tables is not None else TopologyTables.build_py(f)
    f = t.faces
    counts, uniq = t.counts, t.uniq
    open_edges = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    inconsistent = t.inconsistent_edges()
    mesh = SurfaceMesh(points=points, faces=f.astype(np.int32))
    areas = mesh.face_areas()
    repeated = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    degenerate = int((repeated | (areas <= 0.0)).sum())
    comp = t.components()
    n_comp = int(comp.max() + 1) if comp.size else 0
    euler = int(t.used_points() - len(uniq) + len(f))
    boundary_len = (
        float(
            np.linalg.norm(
                points[uniq[counts == 1, 0]] - points[uniq[counts == 1, 1]], axis=1
            ).sum()
        )
        if open_edges
        else 0.0
    )
    return {
        "n_points": int(len(points)),
        "n_faces": int(len(f)),
        "n_edges": int(len(uniq)),
        "open_edges": open_edges,
        "nonmanifold_edges": nonmanifold,
        "inconsistent_edges": inconsistent,
        "degenerate_faces": degenerate,
        "components": n_comp,
        "euler": euler,
        "genus_estimate": (2 * n_comp - euler) / 2 if open_edges == 0 else None,
        "watertight": open_edges == 0 and nonmanifold == 0,
        "oriented": inconsistent == 0,
        "area": float(areas.sum()),
        "boundary_length": boundary_len,
        "bounds": [float(x) for x in mesh.bounds],
        "signed_volume": float(_signed_volume(points, f)),
    }


def _signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    """``sum_i dot(p0, cross(p1, p2)) / 6`` in float64. The dot product is written as three
    products added left to right: ``np.einsum("ij,ij->i", ...)`` accumulates in a
    SIMD-width-dependent order with fused multiply-add (NEON gives
    ``fma(a2, x2, a0 * x0) + a1 * x1``), so it is not the same number on every machine."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape[1] != 3:
        return 0.0
    a, x = p[faces[:, 0]], np.cross(p[faces[:, 1]], p[faces[:, 2]])
    d = a[:, 0] * x[:, 0]
    d += a[:, 1] * x[:, 1]
    d += a[:, 2] * x[:, 2]
    return float(np.sum(d) / 6.0)


def orient_consistently(
    faces: np.ndarray, tables: TopologyTables | None = None
) -> tuple[np.ndarray, int]:
    """Flip faces so neighbours traverse shared edges in opposite directions (per component,
    BFS from the lowest face index). Returns (faces, n_flipped). ``tables`` is reused by the
    reference; the native kernel builds its own."""
    nk = kernels.native()
    f = _native_faces(faces) if nk is not None else None
    if nk is not None and f is not None:
        try:
            out, n_flipped = nk.orient_consistently(f)
        except ValueError:  # negative indices: the reference decides
            pass
        else:
            return out, int(n_flipped)
    return orient_consistently_py(faces, tables)


def orient_consistently_py(
    faces: np.ndarray, tables: TopologyTables | None = None
) -> tuple[np.ndarray, int]:
    """The numpy reference of :func:`orient_consistently`.

    Rules (the Rust port must match them): adjacency is over manifold edges only (count 2);
    the relation value of a face pair is the sum over their shared manifold edges of 1
    (opposite traversal) or 2 (same traversal) and the pair "disagrees" iff that sum is 2;
    the root of every component is its lowest face index and keeps its winding; every other
    face takes ``flip = flip[pred] ^ disagree(pred, face)`` where ``pred`` is its BFS
    predecessor with neighbours visited in ascending face index. All components are walked
    level-synchronously, which gives the same predecessors as one BFS per component.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(faces, dtype=np.int64).copy()
    m = len(f)
    if m == 0:
        return f.astype(np.int32), 0
    t = tables if tables is not None else TopologyTables.build_py(f)
    a, b, sa, sb = t.manifold_pairs()
    # directed edge of face x at slot s: (f[x, s], f[x, (s+1)%3]); same direction -> disagree
    val = np.where(f[a, sa] == f[b, sb], 2.0, 1.0)
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    key = src * m + dst
    ukey, kinv = np.unique(key, return_inverse=True)
    rel = np.bincount(kinv.reshape(-1), weights=np.concatenate([val, val]))
    rows, cols = ukey // m, ukey % m
    indptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=m), out=indptr[1:])
    deg = np.diff(indptr)
    g = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), cols, indptr), shape=(m, m))
    _, comp = connected_components(g, directed=False)
    _, roots = np.unique(comp, return_index=True)
    frontier = np.sort(roots)
    visited = np.zeros(m, dtype=bool)
    flip = np.zeros(m, dtype=bool)
    first_pos = np.empty(m, dtype=np.int64)  # scratch: first discovery position per face
    visited[frontier] = True
    while len(frontier):
        cnt = deg[frontier]
        tot = int(cnt.sum())
        if tot == 0:
            break
        # expand the CSR rows in frontier (queue) order, neighbours ascending
        rep = np.repeat(np.arange(len(frontier)), cnt)
        offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        idx = indptr[frontier][rep] + offs
        nbr = cols[idx]
        keep = ~visited[nbr]
        nbr, idx, rep = nbr[keep], idx[keep], rep[keep]
        if len(nbr) == 0:
            break
        # first discovery wins (the earliest frontier node, then the lowest neighbour):
        # minimum position per neighbour, no sort (ufunc.at is defined for repeated indices)
        pos = np.arange(len(nbr))
        first_pos[nbr] = tot
        np.minimum.at(first_pos, nbr, pos)
        firsts = first_pos[nbr] == pos
        nbr, idx, rep = nbr[firsts], idx[firsts], rep[firsts]
        pred = frontier[rep]
        visited[nbr] = True
        flip[nbr] = flip[pred] ^ (rel[idx] == 2.0)
        frontier = nbr
    n_flipped = int(flip.sum())
    f[flip] = f[flip][:, [0, 2, 1]]
    return f.astype(np.int32), n_flipped


def _group_rows(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lexicographic grouping of the rows of ``key`` (N, C): ``(first, inverse)`` where the
    groups are numbered in sorted row order, ``first[g]`` is the lowest row index of group
    ``g`` and ``inverse[i]`` is the group of row ``i``. Same result as
    ``np.unique(key, axis=0, return_index=True, return_inverse=True)`` without the
    structured-dtype sort."""
    k = np.asarray(key)
    n = len(k)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = np.lexsort(tuple(k[:, c] for c in range(k.shape[1] - 1, -1, -1)))
    ks = k[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = np.any(ks[1:] != ks[:-1], axis=1)
    first = order[np.flatnonzero(new)]
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.cumsum(new) - 1
    return first, inverse


def merge_vertices(
    points: np.ndarray, faces: np.ndarray, tol: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge coincident vertices (exactly, or on a grid of size ``tol``). Returns
    (points, faces, old->new index map). Degenerate faces are dropped.

    Merge order rule: groups are numbered in lexicographic order of the (quantised) coordinate
    rows, and the representative of a group is its first occurrence, i.e. the lowest original
    vertex index; its coordinates are kept verbatim (no averaging).
    """
    nk = kernels.native()
    if nk is not None:
        p, f = _native_points(np.asarray(points)), _native_faces(faces)
        if p is not None and f is not None:
            try:
                # None: keys not representable as int64 (non-finite coordinates with a
                # tolerance), where numpy's conversion is platform-defined
                merged = nk.merge_vertices(p, f, float(tol))
            except ValueError:  # indices outside the point array: the reference decides
                merged = None
            if merged is not None:
                return merged
    return merge_vertices_py(points, faces, tol)


def merge_vertices_py(
    points: np.ndarray, faces: np.ndarray, tol: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The numpy reference of :func:`merge_vertices`."""
    p = np.asarray(points)
    key = p if tol <= 0 else np.round(p / tol).astype(np.int64)
    first, inverse = _group_rows(key)
    new_points = p[first]
    f = inverse[np.asarray(faces, dtype=np.int64)]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return new_points.astype(np.float32), f[keep].astype(np.int32), inverse


def duplicate_faces(faces: np.ndarray) -> np.ndarray:
    """Mask of faces whose vertex set repeats an earlier face's (first occurrence kept)."""
    nk = kernels.native()
    f = _native_faces(faces) if nk is not None else None
    if nk is not None and f is not None:
        return nk.duplicate_faces(f)
    return duplicate_faces_py(faces)


def duplicate_faces_py(faces: np.ndarray) -> np.ndarray:
    """The numpy reference of :func:`duplicate_faces`."""
    key = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
    first, _ = _group_rows(key)
    dup = np.ones(len(faces), dtype=bool)
    dup[first] = False
    return dup


def face_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face areas (float64, (M,)): ``SurfaceMesh.face_areas()``, which is the reference."""
    nk = kernels.native()
    if nk is not None:
        p, f = _native_points(points), _native_faces(faces)
        if p is not None and f is not None:
            try:
                return nk.face_areas(p, f)
            except ValueError:
                pass
    return SurfaceMesh(points=points, faces=faces).face_areas()


def _rewrite_surface(
    ed: SampleEditor,
    points: np.ndarray,
    faces: np.ndarray,
    point_map: np.ndarray | None,
    face_keep: np.ndarray | None,
    *,
    new_point_values: dict[str, np.ndarray] | None = None,
    new_face_values: dict[str, np.ndarray] | None = None,
) -> None:
    """Replace geometry at LOD 0 and carry fields: point fields through ``point_map``
    (first occurrence wins, deterministic), cell fields through ``face_keep``."""
    s = ed.sample
    g: Any = s.group
    ed.write_root_array("points", points.astype(np.float32), ["point", "space"])
    ed.write_root_array("faces", faces.astype(np.int32), ["cell", "cell_node"])
    mesh = SurfaceMesh(points=points, faces=faces)
    areas = face_areas(mesh.points, mesh.faces)
    area = float(areas.sum())
    ed.write_root_array("cell_area", areas.astype(np.float32), ["cell"])
    # first old vertex mapping to each new vertex (computed once, shared by every field)
    first = np.unique(point_map, return_index=True)[1] if point_map is not None else None
    for name in s.field_names("point"):
        info = ed.field_info(name)
        data = ed.field(name, location="point")
        if first is not None:
            data = data[..., first, :] if data.ndim == 3 else data[first]
        if new_point_values and name in new_point_values:
            data = np.concatenate([data, new_point_values[name]], axis=-2)
        ed.write_field(name, data, location="point", units=info.units if info else None)
    for name in s.field_names("cell"):
        info = ed.field_info(name)
        data = ed.field(name, location="cell")
        if face_keep is not None:
            data = data[..., face_keep, :] if data.ndim == 3 else data[face_keep]
        if new_face_values and name in new_face_values:
            data = np.concatenate([data, new_face_values[name]], axis=-2)
        ed.write_field(name, data, location="cell", units=info.units if info else None)
    if "points_t" in g and first is not None:
        ed.write_root_array(
            "points_t", np.asarray(g["points_t"][...])[:, first], ["time", "point", "space"]
        )
    bounds = [float(x) for x in mesh.bounds]
    ed.set_attrs(n_points=int(len(points)), n_faces=int(len(faces)), area=area, bounds=bounds)

    def _m(m: Any) -> None:
        m.mesh.n_points = int(len(points))
        m.mesh.n_faces = int(len(faces))
        m.mesh.area = area
        m.mesh.bounds = bounds

    ed.update_meta(_m)


# -- LOD levels follow level 0 ---------------------------------------------------------------
#
# ``surface.zarr`` stores derived levels as full copies (``lod/<n>/{points, faces, cell_area,
# fields/<name>, cell_fields/<name>}``). Ops edit level 0 only; ``rebuild_lods`` is the one
# function that brings the stored levels back in line, used by ``prep.mesh.decimate`` and by
# the engine after any op that left ``SampleEditor.lods_stale`` set.

#: tolerance class of a rebuilt level (the one ``prep.mesh.decimate`` declares)
LOD_TOLERANCE: dict[str, float] = {"rel_area": 1e-3, "face_count_rel": 0.05}
_SLAB_BYTES = 64 << 20


def lod_target(level: int, stats: dict[str, Any], n_stored: int | None = None) -> int | None:
    """Target face count of a stored level: the recorded ``target``. For a level written
    before targets were recorded, the default of the level (``LOD_TARGET_FACES``) unless the
    stored level is clearly smaller than it and was decimated (a custom, smaller target: the
    stored face count is used). A level that ended above the default (a decimation that
    could not go further on a non-manifold surface) or that is a copy of a small mesh keeps
    the default, so a later repair of level 0 lets it reach the intended size."""
    st = stats.get(str(level)) or {}
    if isinstance(st.get("target"), int) and st["target"] > 0:
        return int(st["target"])
    default = LOD_TARGET_FACES.get(level)
    n = st.get("n_faces") if isinstance(st.get("n_faces"), int) else n_stored
    if default is None:
        return int(n) if n else None
    if n and st.get("method") != "copy" and n < 0.95 * default:
        return int(n)
    return default


def lod_mismatches(ed: SampleEditor) -> dict[int, str]:
    """Stored surface levels that visibly do not describe level 0, with the reason: field
    names that differ at a location, or an array whose length is not the level's own point /
    face count. Metadata only (no array is read). A level with the right names and stale
    *values* cannot be told apart this way; the engine prevents that case, and
    ``prep.mesh.decimate`` with the stored levels repairs it."""
    s = ed.reopen()
    out: dict[int, str] = {}
    if s.kind != "surface":
        return out
    for lvl in ed.lod_levels_on_disk():
        g: Any = s.group["lod"][str(lvl)]
        if "points" not in g or "faces" not in g:
            out[lvl] = "no geometry"
            continue
        sizes = {"point": int(g["points"].shape[0]), "cell": int(g["faces"].shape[0])}
        for loc, n in sizes.items():
            names = s.field_names(loc, lod=lvl)
            if names != s.field_names(loc):
                missing = sorted(set(s.field_names(loc)) - set(names))
                extra = sorted(set(names) - set(s.field_names(loc)))
                out[lvl] = f"{loc} fields differ from level 0 (missing {missing}, extra {extra})"
                break
            bad = [x for x in names if int(s.array(x, loc, lvl).shape[-2]) != n]
            if bad:
                out[lvl] = f"{loc} fields {bad} do not have {n} rows"
                break
    return out


def _as_2d(a: np.ndarray) -> np.ndarray:
    return a[:, None] if a.ndim == 1 else a


def _gather_into(group: Any, name: str, src: Any, idx: np.ndarray, dims: list[str]) -> int:
    """Write ``lod/<n>/<group>/<name>`` as ``src`` gathered through ``idx`` along the entity
    axis (the nearest level-0 vertex or face), keeping dtype and dims; a time-dependent array
    is copied one slab of steps at a time. Returns the bytes written."""
    from frontier_data.dataset import _compressors, _write_array

    if dims[:1] != ["time"]:
        data = np.asarray(src[...])[idx]
        _write_array(group, name, data, dims=dims)
        return int(data.nbytes)
    shape = (int(src.shape[0]), len(idx), *[int(x) for x in src.shape[2:]])
    proto = np.broadcast_to(np.zeros((), dtype=src.dtype), shape)
    chunks = SampleEditor._chunks(proto, dims)  # noqa: SLF001  (the write_field layout rule)
    arr = group.create_array(
        name, shape=shape, dtype=src.dtype, chunks=chunks, compressors=_compressors()
    )
    step_bytes = max(int(np.prod(src.shape[1:])) * np.dtype(src.dtype).itemsize, 1)
    rows = max(int(src.chunks[0]), _SLAB_BYTES // step_bytes, 1)
    rows += (-rows) % max(int(chunks[0]), 1)
    for t0 in range(0, shape[0], rows):
        block = np.asarray(src[t0 : min(t0 + rows, shape[0])])[:, idx]
        arr[t0 : t0 + len(block)] = block
    arr.attrs["dims"] = list(dims)
    return int(proto.nbytes)


def _cell_average(
    values: np.ndarray, area: np.ndarray, assign: np.ndarray, n: int, nearest: np.ndarray
) -> np.ndarray:
    """Area-weighted mean of the level-0 faces assigned to each coarse face (``assign`` is
    the nearest coarse centroid of every level-0 face); a coarse face no level-0 face maps
    to takes its nearest level-0 face. Conservative for the covered faces, summed in
    level-0 face order (deterministic)."""
    v = _as_2d(values).astype(np.float64)
    w = area.astype(np.float64)
    wsum = np.bincount(assign, weights=w, minlength=n)
    out = np.empty((n, v.shape[1]), dtype=np.float64)
    for c in range(v.shape[1]):
        out[:, c] = np.bincount(assign, weights=w * v[:, c], minlength=n)
    covered = wsum > 0
    out[covered] /= wsum[covered, None]
    out[~covered] = v[nearest[~covered]]
    return out.astype(np.float32)


def rebuild_lods(
    ed: SampleEditor,
    levels: list[int] | None = None,
    targets: dict[int, int] | None = None,
    *,
    merge_seams_first: bool = True,
) -> dict[int, dict[str, Any]]:
    """(Re)build derived LOD levels of a sample from its current level 0 and clear
    ``ed.lods_stale``.

    ``levels`` defaults to the levels stored under ``lod/`` (same level numbers); the target
    face count of a level is ``targets[level]``, else the one recorded in ``meta.lod_stats``
    (:func:`lod_target`), else ``LOD_TARGET_FACES``. Every stored level ends up with exactly
    the field names of level 0 at each location:

    - geometry and steady float point fields: the ``frontier_data.lod.build_lods`` cascade
      (quadric decimation after merging coincident vertices, inverse-distance transfer);
    - a steady float cell field with a point copy: the transferred point field averaged to
      the faces (what the converters store);
    - a steady float cell field without a point copy: area-weighted mean of the level-0
      faces nearest to each coarse face;
    - integer fields and time-dependent fields: the nearest level-0 vertex / face.

    ``meta.lod_levels``, ``meta.lod_stats`` (``n_faces``, ``n_points``, ``method``,
    ``target``) and the root ``lod_levels`` attribute are updated; the manifest entry follows
    in the engine. Level 0 is never touched. A ``mesh.zarr`` level is a vertex map and is
    rebuilt with the vertex-clustering of ``frontier_data.lod``. Returns
    ``{level: {n_faces, n_points, method, target, rel_area, seconds}}``; ``seconds`` is not
    stored (timings are not content).
    """
    s = ed.reopen()
    wanted = sorted({int(x) for x in (levels if levels is not None else ed.lod_levels_on_disk())})
    wanted = [x for x in wanted if x > 0]
    if not wanted or s.kind not in ("surface", "mesh"):
        ed.lods_stale = ed.geometry_stale = False
        return {}
    if s.kind == "mesh":
        built = _rebuild_mesh_lods(ed, wanted, targets)
    else:
        built = _rebuild_surface_lods(ed, wanted, targets, merge_seams_first)
    ed.lods_stale = ed.geometry_stale = False
    ed.reopen()
    return built


def _rebuild_surface_lods(
    ed: SampleEditor, wanted: list[int], targets: dict[int, int] | None, merge_first: bool
) -> dict[int, dict[str, Any]]:
    import shutil
    import time

    import zarr
    from scipy.spatial import KDTree

    from frontier_data.dataset import _write_mesh_group
    from frontier_data.lod import merge_coincident_vertices

    s = ed.sample
    points, faces = np.asarray(s.group["points"][...]), s.cells
    point_names, cell_names = s.field_names("point"), s.field_names("cell")

    def _steady_float(name: str, loc: str) -> bool:
        a = s.array(name, loc)
        return a.dtype.kind == "f" and a.ndim <= 2 and not s.is_time_dependent(name, loc)

    cascade_pf = {
        n: _as_2d(ed.field(n, location="point")) for n in point_names if _steady_float(n, "point")
    }
    # cell fields the cascade derives from their point copy; the rest is transferred below
    cascade_cf = {
        n: _as_2d(ed.field(n, location="cell"))
        for n in cell_names
        if n in cascade_pf and _steady_float(n, "cell")
    }
    full = SurfaceMesh(points=points, faces=faces, point_fields=cascade_pf, cell_fields=cascade_cf)
    full_area = full.face_areas()
    area0 = float(full_area.sum())
    mesh = full
    if merge_first:
        p, f = merge_coincident_vertices(full)
        if len(p) != full.n_points:
            first, inv = _group_rows(full.points)  # point fields by first occurrence
            keep = (
                (inv[faces[:, 0]] != inv[faces[:, 1]])
                & (inv[faces[:, 1]] != inv[faces[:, 2]])
                & (inv[faces[:, 0]] != inv[faces[:, 2]])
            )
            mesh = SurfaceMesh(
                points=p,
                faces=f,
                point_fields={k: v[first] for k, v in cascade_pf.items()},
                cell_fields={k: v[keep] for k, v in cascade_cf.items()},
            )
    stats = ed.meta.lod_stats
    disk = set(ed.lod_levels_on_disk())
    tg: dict[int, int | None] = dict(LOD_TARGET_FACES)
    for lvl in wanted:
        n_stored = int(s.group["lod"][str(lvl)]["faces"].shape[0]) if lvl in disk else None
        tg[lvl] = lod_target(lvl, stats, n_stored)
    tg.update({int(k): int(v) for k, v in (targets or {}).items()})
    missing = [lvl for lvl in wanted if tg.get(lvl) is None]
    if missing:
        raise ValueError(f"{ed.id}: no target face count for LOD level(s) {missing}")
    lods, timings = build_lods(mesh, [0, *wanted], targets=tg)
    by_level = {t.level: t for t in timings}

    extra_point = [n for n in point_names if n not in cascade_pf]
    extra_cell = [n for n in cell_names if n not in cascade_cf]
    centroids0 = points[faces].astype(np.float64).mean(axis=1) if extra_cell else None
    root: Any = zarr.open_group(str(s.store_path), mode="r+")
    lod_group = root["lod"] if "lod" in root else root.create_group("lod")
    built: dict[int, dict[str, Any]] = {}
    for lvl in wanted:
        t0 = time.perf_counter()
        lod = lods[lvl]
        if (s.store_path / "lod" / str(lvl)).exists():
            shutil.rmtree(s.store_path / "lod" / str(lvl))
        g = lod_group.create_group(str(lvl))
        _write_mesh_group(g, lod)
        written = lod.points.nbytes + lod.faces.nbytes + 4 * lod.n_faces
        written += sum(4 * v.size for v in (*lod.point_fields.values(), *lod.cell_fields.values()))
        if extra_point:
            idx = np.asarray(
                KDTree(points.astype(np.float64)).query(lod.points.astype(np.float64))[1]
            )
            for n in extra_point:
                written += _gather_into(
                    g["fields"], n, s.array(n, "point"), idx, s.dims(n, "point")
                )
        if extra_cell and centroids0 is not None:
            cg = g["cell_fields"] if "cell_fields" in g else g.create_group("cell_fields")
            cent = lod.points[lod.faces].astype(np.float64).mean(axis=1)
            nearest = np.asarray(KDTree(centroids0).query(cent)[1])
            assign: np.ndarray | None = None  # nearest coarse face of every level-0 face
            for n in extra_cell:
                dims = s.dims(n, "cell")
                if _steady_float(n, "cell"):
                    from frontier_data.dataset import _write_array

                    if assign is None:
                        assign = np.asarray(KDTree(cent).query(centroids0)[1])
                    data = _cell_average(
                        ed.field(n, location="cell"), full_area, assign, lod.n_faces, nearest
                    )
                    _write_array(cg, n, data, dims=dims)
                    written += data.nbytes
                else:
                    written += _gather_into(cg, n, s.array(n, "cell"), nearest, dims)
        ed.bytes_written += int(written)
        t = by_level[lvl]
        built[lvl] = {
            "n_faces": lod.n_faces,
            "n_points": lod.n_points,
            "method": t.method,
            "target": int(tg[lvl] or 0),
            "rel_area": abs(lod.area - area0) / max(area0, 1e-30),
            "seconds": t.seconds + time.perf_counter() - t0,
        }
    present = sorted({0, *ed.lod_levels_on_disk()})
    root.attrs["lod_levels"] = present

    def _m(m: Any) -> None:
        m.lod_levels = present
        m.lod_stats = {k: v for k, v in m.lod_stats.items() if k.isdigit() and int(k) in present}
        zero = m.lod_stats.setdefault("0", {"method": "source"})
        zero.update({"n_faces": int(len(faces)), "n_points": int(len(points))})
        for lvl, st in built.items():
            m.lod_stats[str(lvl)] = {k: st[k] for k in ("n_faces", "n_points", "method", "target")}

    ed.update_meta(_m)
    return built


def _rebuild_mesh_lods(
    ed: SampleEditor, wanted: list[int], targets: dict[int, int] | None
) -> dict[int, dict[str, Any]]:
    import time

    import zarr

    from frontier_data.arrays import default_cell_types
    from frontier_data.lod import (
        MESH_LOD_TARGETS,
        build_mesh_lod_maps,
        drawable_triangles,
        write_mesh_lods,
    )

    t0 = time.perf_counter()
    g: Any = zarr.open_group(str(ed.store_path), mode="r+")
    points = np.asarray(g["points"][...], dtype=np.float32)
    cells = np.asarray(g["cells"][...])
    ctypes = (
        np.asarray(g["cell_types"][...])
        if "cell_types" in g
        else default_cell_types(cells, points.shape[1])
    )
    tris = drawable_triangles(cells, ctypes, points.shape[1])
    tg = {**MESH_LOD_TARGETS, **{int(k): int(v) for k, v in (targets or {}).items()}}
    maps = build_mesh_lod_maps(points, tris, wanted, targets=tg)
    present = write_mesh_lods(g, points, maps)
    ed.update_meta(lambda m: setattr(m, "lod_levels", present))
    dt = time.perf_counter() - t0
    return {
        lvl: {
            "n_faces": int(len(f)),
            "n_points": int(len(vm)),
            "method": "vertex_clustering",
            "target": int(tg[lvl]),
            "rel_area": 0.0,
            "seconds": dt / max(len(maps), 1),
        }
        for lvl, (vm, f) in sorted(maps.items())
    }


def _surface_ids(ref: DatasetRef) -> list[str]:
    return [e.id for e in ref.manifest.samples if e.sample_kind == "surface"]


# -- mesh.watertight_check -------------------------------------------------------------------


class TopologyParams(Params):
    lod: int = Field(default=0, description="LOD level to audit (0 = full).")
    fail_on_open: bool = Field(default=False, description="Fail (not warn) when open edges exist.")


@register
class WatertightCheck(BaseOp):
    id = "prep.mesh.watertight_check"
    version = "1.0.0"
    category = "mesh"
    summary = (
        "Topology audit: open and non-manifold edges, components, Euler characteristic, "
        "orientation, degenerate faces, signed volume."
    )
    when_to_use = (
        "Before any topology op (decimation, hole filling) and before integrating forces (lift "
        "needs a closed surface or gauge pressure)."
    )
    read_only = True
    produces_assertions = True
    Params = TopologyParams

    def plan_samples(self, ref: DatasetRef, params: TopologyParams) -> list[str]:
        return _surface_ids(ref)

    def apply_sample(self, ed: SampleEditor, params: TopologyParams, *, seed: int) -> SampleOutcome:
        s = ed.sample
        pts = s.points(lod=params.lod)
        faces = (
            s.cells
            if params.lod == 0
            else np.asarray(s.group["lod"][str(params.lod)]["faces"][...])
        )  # type: ignore[index]
        t = topology(pts, faces)
        status = (
            "pass"
            if t["watertight"] and t["oriented"]
            else (
                "fail"
                if (params.fail_on_open and t["open_edges"]) or t["nonmanifold_edges"]
                else "warn"
            )
        )
        ed.update_meta(lambda m: m.extra.setdefault("topology", {}).update({str(params.lod): t}))
        return SampleOutcome(
            sample_id=ed.id,
            assertions=[
                Assertion(check="watertight", status=status, target=f"samples/{ed.id}", detail=t)
            ],
            stats=t,
            changes=[
                Change(
                    target=f"samples/{ed.id}/meta.json", action="modify", detail={"topology": True}
                )
            ],
        )


# -- mesh.merge_seams ----------------------------------------------------------------------


class MergeSeamsParams(Params):
    tolerance: float = Field(
        default=0.0, ge=0, description="Merge radius (0 = exactly coincident)."
    )
    max_area_change: float = Field(
        default=1e-3, description="Fail when the relative area changes more than this."
    )


@register
class MergeSeams(BaseOp):
    id = "prep.mesh.merge_seams"
    version = "1.0.0"
    category = "mesh"
    summary = (
        "Merge coincident vertices at patch seams (triangle soup from per-patch exports) and drop "
        "collapsed faces; reports open edges and area before/after."
    )
    when_to_use = (
        "Solver exports (OpenFOAM patches, Fluent zones) where each patch carries its own copy of "
        "seam vertices: thousands of open edges, decimation tears the surface."
    )
    changes_geometry = True
    produces_assertions = True
    Params = MergeSeamsParams

    def plan_samples(self, ref: DatasetRef, params: MergeSeamsParams) -> list[str]:
        return _surface_ids(ref)

    def plan_sample(self, ed: SampleEditor, params: MergeSeamsParams) -> list[Change]:
        return [
            Change(
                target=f"samples/{ed.id}/points",
                action="modify",
                detail={"tolerance": params.tolerance},
            )
        ]

    def apply_sample(
        self, ed: SampleEditor, params: MergeSeamsParams, *, seed: int
    ) -> SampleOutcome:
        s = ed.sample
        pts, faces = s.points(), s.cells
        before = topology(pts, faces)
        new_pts, new_faces, pmap = merge_vertices(pts, faces, params.tolerance)
        if len(new_pts) == len(pts) and len(new_faces) == len(faces):
            return SampleOutcome(
                sample_id=ed.id, skipped=True, stats={"merged": 0, "before": before}
            )
        keep = (
            (pmap[faces[:, 0]] != pmap[faces[:, 1]])
            & (pmap[faces[:, 1]] != pmap[faces[:, 2]])
            & (pmap[faces[:, 0]] != pmap[faces[:, 2]])
        )
        after = topology(new_pts, new_faces)
        rel_area = abs(after["area"] - before["area"]) / max(before["area"], 1e-30)
        if rel_area > params.max_area_change:
            return SampleOutcome(
                sample_id=ed.id,
                skipped=True,
                warnings=[
                    OpWarning(
                        severity="fail",
                        code="area-change",
                        message=f"{ed.id}: merging changed area by {rel_area:.2e}",
                        target=ed.id,
                    )
                ],
            )
        _rewrite_surface(ed, new_pts, new_faces, pmap, keep)
        detail = {
            "merged_vertices": int(len(pts) - len(new_pts)),
            "dropped_faces": int(len(faces) - len(new_faces)),
            "open_edges_before": before["open_edges"],
            "open_edges_after": after["open_edges"],
            "area_rel_change": rel_area,
        }
        ed.update_meta(lambda m: m.extra.setdefault("geometry", {}).update({"merge_seams": detail}))
        return SampleOutcome(
            sample_id=ed.id,
            changes=[
                Change(
                    target=f"samples/{ed.id}/points",
                    action="modify",
                    before={"n_points": before["n_points"], "open_edges": before["open_edges"]},
                    after={"n_points": after["n_points"], "open_edges": after["open_edges"]},
                )
            ],
            assertions=[
                Assertion(
                    check="area_conservation",
                    status="pass",
                    target=f"samples/{ed.id}",
                    detail={"rel_change": rel_area},
                )
            ],
            stats=detail,
        )


# -- mesh.repair -----------------------------------------------------------------------------


class RepairParams(Params):
    drop_degenerate: bool = True
    drop_duplicate_faces: bool = True
    orient: bool = Field(
        default=True, description="Make face orientation consistent per component."
    )
    fill_holes_max_edges: int = Field(
        default=0,
        ge=0,
        description=(
            "Fill boundary loops with at most this many edges (0 = off). Changes geometry; area "
            "delta recorded."
        ),
    )
    max_area_change: float = 1e-2


@register
class RepairMesh(BaseOp):
    id = "prep.mesh.repair"
    version = "1.0.0"
    category = "mesh"
    summary = (
        "Drop degenerate and duplicate faces, make orientation consistent, fill small holes "
        "(opt-in); invariant diff recorded."
    )
    when_to_use = (
        "After watertight_check reports defects. Hole filling is opt-in because it invents "
        "geometry; keep it to small loops and check the area delta."
    )
    changes_geometry = True
    produces_assertions = True
    Params = RepairParams

    def plan_samples(self, ref: DatasetRef, params: RepairParams) -> list[str]:
        return _surface_ids(ref)

    def plan_sample(self, ed: SampleEditor, params: RepairParams) -> list[Change]:
        return [
            Change(target=f"samples/{ed.id}/faces", action="modify", detail=params.model_dump())
        ]

    def apply_sample(self, ed: SampleEditor, params: RepairParams, *, seed: int) -> SampleOutcome:
        s = ed.sample
        pts, faces = s.points(), s.cells.astype(np.int64)
        tables = TopologyTables.build(faces)
        before = topology(pts, faces, tables)
        keep = np.ones(len(faces), dtype=bool)
        detail: dict[str, Any] = {}
        if params.drop_degenerate:
            deg = SurfaceMesh(points=pts, faces=faces.astype(np.int32)).degenerate_faces()
            keep &= ~deg
            detail["degenerate_dropped"] = int(deg.sum())
        if params.drop_duplicate_faces:
            dup = duplicate_faces(faces)
            keep &= ~dup
            detail["duplicate_dropped"] = int(dup.sum())
        f = faces[keep]
        n_flipped = 0
        if params.orient:
            # the edge tables of the audit are valid as long as no face was dropped
            f, n_flipped = orient_consistently(f, tables if keep.all() else None)
            detail["flipped"] = n_flipped
        new_pts, new_pf, new_cf = pts, None, None
        if params.fill_holes_max_edges > 0:
            new_pts, f, new_pf, new_cf, n_filled = _fill_holes(
                ed, pts, f, params.fill_holes_max_edges
            )
            detail["holes_filled"] = n_filled
        changed = (not keep.all()) or n_flipped > 0 or len(new_pts) != len(pts)
        if not changed:
            return SampleOutcome(sample_id=ed.id, skipped=True, stats=detail)
        after = topology(new_pts, f)
        rel_area = abs(after["area"] - before["area"]) / max(before["area"], 1e-30)
        status = "pass" if rel_area <= params.max_area_change else "fail"
        if status == "fail":
            return SampleOutcome(
                sample_id=ed.id,
                skipped=True,
                warnings=[
                    OpWarning(
                        severity="fail",
                        code="area-change",
                        message=f"{ed.id}: repair changed area by {rel_area:.2e}",
                        target=ed.id,
                    )
                ],
            )
        # cell fields: dropped faces removed; flipped faces keep values; new faces get the
        # mean of their neighbours
        _rewrite_surface(
            ed,
            new_pts,
            f,
            None if len(new_pts) == len(pts) else np.arange(len(pts)),
            keep,
            new_point_values=new_pf,
            new_face_values=new_cf,
        )
        detail.update(
            {
                "open_edges_before": before["open_edges"],
                "open_edges_after": after["open_edges"],
                "area_rel_change": rel_area,
            }
        )
        ed.update_meta(lambda m: m.extra.setdefault("geometry", {}).update({"repair": detail}))
        return SampleOutcome(
            sample_id=ed.id,
            changes=[
                Change(
                    target=f"samples/{ed.id}/faces",
                    action="modify",
                    before={
                        "n_faces": before["n_faces"],
                        "open_edges": before["open_edges"],
                        "inconsistent_edges": before["inconsistent_edges"],
                    },
                    after={
                        "n_faces": after["n_faces"],
                        "open_edges": after["open_edges"],
                        "inconsistent_edges": after["inconsistent_edges"],
                    },
                )
            ],
            assertions=[
                Assertion(
                    check="area_conservation",
                    status=status,
                    target=f"samples/{ed.id}",
                    detail={"rel_change": rel_area},
                )
            ],
            stats=detail,
        )


def _boundary_loops(faces: np.ndarray, tables: TopologyTables | None = None) -> list[list[int]]:
    t = tables if tables is not None else TopologyTables.build(faces)
    b = t.uniq[t.boundary]
    if len(b) == 0:
        return []
    nxt: dict[int, list[int]] = {}
    for a, c in b.tolist():
        nxt.setdefault(a, []).append(c)
        nxt.setdefault(c, []).append(a)
    seen: set[int] = set()
    loops: list[list[int]] = []
    for start in sorted(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur, prev = start, -1
        while True:
            cands = [v for v in nxt[cur] if v != prev and v not in seen]
            if not cands:
                break
            prev, cur = cur, cands[0]
            seen.add(cur)
            loop.append(cur)
        loops.append(loop)
    return loops


def _fill_holes(
    ed: SampleEditor, pts: np.ndarray, faces: np.ndarray, max_edges: int
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Fan-triangulate small boundary loops around a new centroid vertex."""
    loops = [lp for lp in _boundary_loops(faces) if 3 <= len(lp) <= max_edges]
    if not loops:
        return pts, faces, {}, {}, 0
    s = ed.sample
    new_pts = [pts]
    new_faces = [faces]
    pf: dict[str, list[np.ndarray]] = {n: [] for n in s.field_names("point")}
    cf: dict[str, list[np.ndarray]] = {n: [] for n in s.field_names("cell")}
    pdata = {n: ed.field(n, location="point") for n in pf}
    cdata = {n: ed.field(n, location="cell") for n in cf}
    # faces touching each boundary vertex, for cell-value transfer
    n = len(pts)
    for lp in loops:
        c = pts[lp].mean(axis=0, keepdims=True)
        new_pts.append(c.astype(np.float32))
        ring = np.array(lp)
        tri = np.column_stack([ring, np.roll(ring, -1), np.full(len(ring), n)])
        new_faces.append(tri.astype(np.int32))
        for name in pf:
            d = pdata[name]
            pf[name].append(
                d[..., lp, :].mean(axis=-2, keepdims=True)
                if d.ndim == 3
                else d[lp].mean(axis=0, keepdims=True)
            )
        touching = np.nonzero(np.isin(faces, ring).any(axis=1))[0]
        for name in cf:
            d = cdata[name]
            v = (
                d[..., touching, :].mean(axis=-2, keepdims=True)
                if d.ndim == 3
                else d[touching].mean(axis=0, keepdims=True)
            )
            cf[name].append(np.repeat(v, len(ring), axis=-2))
        n += 1
    out_pf = {k: np.concatenate(v, axis=-2) for k, v in pf.items() if v}
    out_cf = {k: np.concatenate(v, axis=-2) for k, v in cf.items() if v}
    f = np.concatenate(new_faces)
    f, _ = orient_consistently(f)
    return np.concatenate(new_pts), f, out_pf, out_cf, len(loops)


# -- mesh.orient_outward ---------------------------------------------------------------------


class OrientParams(Params):
    method: Literal["signed_volume", "centroid"] = Field(
        default="signed_volume",
        description=(
            "signed_volume for closed shells; centroid: majority of normals away from the centroid "
            "(open shells)."
        ),
    )


@register
class OrientOutward(BaseOp):
    id = "prep.mesh.orient_outward"
    version = "1.0.0"
    category = "mesh"
    summary = (
        "Make faces consistently oriented and pointing outward (signed volume or centroid test); "
        "records flips."
    )
    when_to_use = (
        "Force integration and normals-based features need outward normals; run after repair or "
        "seam merge, and after any decimation."
    )
    changes_geometry = True
    Params = OrientParams

    def plan_samples(self, ref: DatasetRef, params: OrientParams) -> list[str]:
        return _surface_ids(ref)

    def plan_sample(self, ed: SampleEditor, params: OrientParams) -> list[Change]:
        return [
            Change(
                target=f"samples/{ed.id}/faces", action="modify", detail={"method": params.method}
            )
        ]

    def apply_sample(self, ed: SampleEditor, params: OrientParams, *, seed: int) -> SampleOutcome:
        s = ed.sample
        pts, faces = s.points(), s.cells
        f, n_flipped = orient_consistently(faces)
        flip_all = False
        if params.method == "signed_volume":
            flip_all = _signed_volume(pts, f) < 0
        else:
            m = SurfaceMesh(points=pts, faces=f)
            c = pts.mean(axis=0)
            centers = pts[f].mean(axis=1)
            flip_all = (
                float(np.mean(np.einsum("ij,ij->i", m.face_normals(), centers - c) < 0)) > 0.5
            )
        if flip_all:
            f = f[:, [0, 2, 1]]
            n_flipped = len(f) - n_flipped
        if n_flipped == 0:
            return SampleOutcome(sample_id=ed.id, skipped=True)
        ed.write_root_array("faces", f.astype(np.int32), ["cell", "cell_node"])
        detail = {"flipped": int(n_flipped), "flip_all": bool(flip_all), "method": params.method}
        ed.update_meta(lambda m: m.extra.setdefault("geometry", {}).update({"orient": detail}))
        return SampleOutcome(
            sample_id=ed.id,
            changes=[Change(target=f"samples/{ed.id}/faces", action="modify", detail=detail)],
            stats=detail,
        )


# -- mesh.transform --------------------------------------------------------------------------


class TransformParams(Params):
    scale: float | None = Field(
        default=None, description="Uniform scale factor (alternative to from_unit/to_unit)."
    )
    from_unit: str | None = Field(
        default=None, description="Length unit of the stored points, e.g. mm."
    )
    to_unit: str | None = Field(default=None, description="Target length unit, e.g. m.")
    translate: list[float] | None = None
    axes: list[str] | None = Field(
        default=None,
        description="Axis permutation with signs, e.g. ['x','z','-y'] for y-up -> z-up.",
    )
    allow_reflection: bool = Field(
        default=False, description="Permit a handedness change (determinant -1)."
    )
    vector_fields: list[str] | None = Field(
        default=None,
        description="Vector fields to rotate with the frame (default: every 3-component field).",
    )


@register
class TransformMesh(BaseOp):
    id = "prep.mesh.transform"
    version = "1.0.0"
    category = "mesh"
    summary = (
        "Scale (unit conversion), translate and permute axes of the geometry and its vector "
        "fields, with a handedness check."
    )
    when_to_use = (
        "mm vs m, quarter-scale models, y-up CAD exports. The op refuses reflections unless "
        "allowed because they flip normals and chirality silently."
    )
    changes_geometry = True
    changes_fields = True
    Params = TransformParams

    def _matrix(self, params: TransformParams) -> tuple[np.ndarray, float, str]:
        scale = params.scale if params.scale is not None else 1.0
        desc = []
        if params.from_unit and params.to_unit:
            reg = vocab.ureg()
            scale *= float(
                reg.Quantity(1.0, vocab.normalize_unit(params.from_unit))
                .to(vocab.normalize_unit(params.to_unit))
                .magnitude
            )
            desc.append(f"{params.from_unit}->{params.to_unit}")
        r = np.eye(3)
        if params.axes:
            r = np.zeros((3, 3))
            for i, ax in enumerate(params.axes):
                sign = -1.0 if ax.startswith("-") else 1.0
                j = "xyz".index(ax.lstrip("+-"))
                r[i, j] = sign
            desc.append(f"axes={params.axes}")
        det = float(np.linalg.det(r))
        if det < 0 and not params.allow_reflection:
            raise ValueError(
                f"axes {params.axes} is a reflection (det={det:.0f}); "
                "set allow_reflection to confirm"
            )
        return r, scale, ", ".join(desc) or f"scale={scale}"

    def plan_samples(self, ref: DatasetRef, params: TransformParams) -> list[str]:
        return [e.id for e in ref.manifest.samples if e.sample_kind != "grid"]

    def plan_extra(
        self, ref: DatasetRef, params: TransformParams
    ) -> tuple[list[OpWarning], list[Any]]:
        try:
            self._matrix(params)
        except ValueError as exc:
            return [OpWarning(severity="fail", code="reflection", message=str(exc))], []
        return [], []

    def plan_sample(self, ed: SampleEditor, params: TransformParams) -> list[Change]:
        _, scale, desc = self._matrix(params)
        return [
            Change(
                target=f"samples/{ed.id}/points",
                action="modify",
                detail={"transform": desc, "scale": scale},
            )
        ]

    def apply_sample(
        self, ed: SampleEditor, params: TransformParams, *, seed: int
    ) -> SampleOutcome:
        r, scale, desc = self._matrix(params)
        s = ed.sample
        g: Any = s.group
        d = s.points().shape[1]
        rr = r[:d, :d]
        tr = np.asarray(params.translate or [0.0] * d, dtype=np.float64)[:d]

        def tf(p: np.ndarray) -> np.ndarray:
            return ((p.astype(np.float64) * scale) @ rr.T + tr).astype(np.float32)

        pts = tf(s.points())
        ed.write_root_array("points", pts, ["point", "space"])
        if "points_t" in g:
            ed.write_root_array(
                "points_t",
                np.stack([tf(x) for x in np.asarray(g["points_t"][...])]),
                ["time", "point", "space"],
            )
        if "cell_area" in g:
            ed.write_root_array(
                "cell_area",
                (np.asarray(g["cell_area"][...]) * scale * scale).astype(np.float32),
                ["cell"],
            )
        if s.kind == "surface" and np.linalg.det(rr) < 0:
            ed.write_root_array("faces", s.cells[:, [0, 2, 1]], ["cell", "cell_node"])
        changes = [
            Change(target=f"samples/{ed.id}/points", action="modify", detail={"transform": desc})
        ]
        if params.axes:
            names = params.vector_fields or [
                n for n in s.field_names() if s.array(n).shape[-1] == 3
            ]
            for name in names:
                if name not in s.field_names():
                    continue
                info = ed.field_info(name)
                # the native array and a derived copy at the other location rotate together
                locs = [x for x in ("cell", "point") if name in s.field_names(x)]
                for loc in locs or [ed.location_of(name)]:
                    data = ed.field(name, location=loc)
                    ed.write_field(
                        name,
                        (data.astype(np.float64) @ r.T).astype(np.float32),
                        location=loc,
                        units=info.units if info else None,
                        meta_extra={"frame": desc},
                    )
                    changes.append(
                        Change(
                            target=ed.target(name, loc), action="modify", detail={"rotated": desc}
                        )
                    )
        mesh = SurfaceMesh(points=pts, faces=s.cells) if s.kind == "surface" else None
        bounds = [float(x) for x in (mesh.bounds if mesh else _bounds(pts))]
        ed.set_attrs(bounds=bounds)

        def _m(m: Any) -> None:
            m.mesh.bounds = bounds
            if mesh is not None:
                m.mesh.area = float(mesh.face_areas().sum())
            m.extra.setdefault("geometry", {})["transform"] = {
                "desc": desc,
                "scale": scale,
                "translate": tr.tolist(),
                "axes": params.axes,
            }

        ed.update_meta(_m)
        return SampleOutcome(
            sample_id=ed.id, changes=changes, stats={"scale": scale, "transform": desc}
        )


def _bounds(points: np.ndarray) -> list[float]:
    out = [0.0] * 6
    if points.size:
        lo, hi = points.min(axis=0), points.max(axis=0)
        for i in range(min(points.shape[1], 3)):
            out[2 * i], out[2 * i + 1] = float(lo[i]), float(hi[i])
    return out


# -- mesh.decimate ---------------------------------------------------------------------------


class DecimateParams(Params):
    levels: list[int] = Field(
        default_factory=lambda: [2, 3], description="LOD levels to (re)build."
    )
    targets: dict[int, int] | None = Field(
        default=None, description="Level -> target face count (defaults: 1=100k, 2=20k, 3=5k)."
    )
    merge_seams_first: bool = True
    max_area_change: float = Field(
        default=1e-3, description="Tolerance on relative area per level."
    )


@register
class Decimate(BaseOp):
    """Build the listed LOD levels from the current level 0 (:func:`rebuild_lods`); every
    level gets the field names of level 0 at each location. Also the repair for a dataset
    whose stored levels no longer match level 0: run it with the stored levels.

    Determinism register: ``tolerance`` (quadric collapse order depends on the library's heap
    tie-breaking; ``rel_area`` 1e-3, ``face_count_rel`` 0.05; identical run to run on one
    machine and library version). The same class applies to the LOD groups the engine
    rebuilds after any other op changed level 0 of a sample that stores levels; level 0 keeps
    that op's own class and such a Result carries ``tolerance: {lod_rel_area,
    lod_face_count_rel}``."""

    id = "prep.mesh.decimate"
    version = "1.0.0"
    category = "mesh"
    summary = (
        "(Re)build the LOD cascade with quadric decimation after seam merge; area conservation "
        "checked per level."
    )
    when_to_use = (
        "Samples without LOD levels, or after geometry repair; viewers need level 2/3, coarse "
        "models train on level 1/2."
    )
    changes_geometry = True
    produces_assertions = True
    determinism = "tolerance"
    tolerance: ClassVar[dict[str, float] | None] = dict(LOD_TOLERANCE)
    Params = DecimateParams

    def plan_samples(self, ref: DatasetRef, params: DecimateParams) -> list[str]:
        return _surface_ids(ref)

    def plan_sample(self, ed: SampleEditor, params: DecimateParams) -> list[Change]:
        return [
            Change(
                target=f"samples/{ed.id}/lod/{lvl}",
                action="add" if lvl not in ed.meta.lod_levels else "modify",
            )
            for lvl in params.levels
        ]

    def apply_sample(self, ed: SampleEditor, params: DecimateParams, *, seed: int) -> SampleOutcome:
        built = rebuild_lods(
            ed,
            levels=params.levels,
            targets=params.targets,
            merge_seams_first=params.merge_seams_first,
        )
        assertions = [
            Assertion(
                check="lod_area_conservation",
                status="pass" if st["rel_area"] <= params.max_area_change else "warn",
                target=f"samples/{ed.id}/lod/{lvl}",
                detail={"rel_area": st["rel_area"], "n_faces": st["n_faces"]},
            )
            for lvl, st in built.items()
        ]
        return SampleOutcome(
            sample_id=ed.id,
            changes=[
                Change(
                    target=f"samples/{ed.id}/lod/{lvl}",
                    action="modify",
                    detail={"n_faces": st["n_faces"]},
                )
                for lvl, st in built.items()
            ],
            assertions=assertions,
            stats={"levels": {lvl: st["n_faces"] for lvl, st in built.items()}},
        )

    def apply_manifest(
        self, ref: DatasetRef, manifest: Any, params: DecimateParams, outcomes: list[SampleOutcome]
    ) -> tuple[list[Change], dict[str, Any]]:
        done = {o.sample_id for o in outcomes if not o.skipped}
        for e in manifest.samples:
            if e.id in done:
                e.lod_levels = sorted({0, *ref.editor(e.id).lod_levels_on_disk()})
        return [], {}


# -- mesh.resample ---------------------------------------------------------------------------


class ResampleParams(Params):
    n_points: int = Field(ge=8, description="Target point budget.")
    method: Literal["random", "farthest_point", "area_weighted"] = "farthest_point"
    name: str = Field(default="resample_index", description="Name of the static index array.")


@register
class ResamplePoints(BaseOp):
    id = "prep.mesh.resample"
    version = "1.0.0"
    category = "mesh"
    summary = (
        "Select a fixed point budget per sample (random, area-weighted or farthest-point) and "
        "store the indices as a static array; seeded."
    )
    when_to_use = (
        "Point-based models (PointNet, Transolver) that need the same number of points per sample "
        "without decimating the geometry."
    )
    changes_fields = True
    determinism = "seeded"
    Params = ResampleParams

    def plan_samples(self, ref: DatasetRef, params: ResampleParams) -> list[str]:
        return [e.id for e in ref.manifest.samples if e.sample_kind != "grid"]

    def plan_sample(self, ed: SampleEditor, params: ResampleParams) -> list[Change]:
        return [
            Change(
                target=f"samples/{ed.id}/static/{params.name}",
                action="add",
                detail={"n": params.n_points, "method": params.method},
            )
        ]

    def apply_sample(self, ed: SampleEditor, params: ResampleParams, *, seed: int) -> SampleOutcome:
        s = ed.sample
        pts = s.points().astype(np.float64)
        n = min(params.n_points, len(pts))
        rng = np.random.default_rng(seed)
        if params.method == "random":
            idx = np.sort(rng.choice(len(pts), n, replace=False))
        elif params.method == "area_weighted" and s.kind == "surface":
            m = SurfaceMesh(points=pts.astype(np.float32), faces=s.cells)
            w = m.vertex_areas()
            idx = np.sort(rng.choice(len(pts), n, replace=False, p=w / w.sum()))
        else:
            idx = _farthest_point(pts, n, int(rng.integers(len(pts))))
        ed.write_field(
            params.name,
            idx.astype(np.int32)[:, None],
            location="static",
            dims=["sample", "component"],
            units="-",
            meta_extra={"method": params.method, "seed": seed, "n": int(n)},
        )
        return SampleOutcome(
            sample_id=ed.id,
            changes=[
                Change(
                    target=f"samples/{ed.id}/static/{params.name}",
                    action="add",
                    detail={"n": int(n)},
                )
            ],
            stats={"n": int(n)},
        )


def _farthest_point(pts: np.ndarray, n: int, start: int) -> np.ndarray:
    """Farthest-point sampling on a seeded subset (at most 200k candidates) for speed."""
    cand = np.arange(len(pts))
    if len(pts) > 200_000:
        cand = np.linspace(0, len(pts) - 1, 200_000).astype(np.int64)
    p = pts[cand]
    sel = np.empty(n, dtype=np.int64)
    sel[0] = np.searchsorted(cand, start) if start in cand else 0
    d = np.linalg.norm(p - p[sel[0]], axis=1)
    for i in range(1, n):
        j = int(np.argmax(d))
        sel[i] = j
        d = np.minimum(d, np.linalg.norm(p - p[j], axis=1))
    return np.sort(cand[sel])


# -- mesh.part_ids ---------------------------------------------------------------------------


class PartIdsParams(Params):
    name: str = "part_id"
    overwrite: bool = False
    split_by_normals: bool = Field(
        default=False,
        description=(
            "Also split components where the dihedral angle exceeds `angle_deg` (patch-like parts)."
        ),
    )
    angle_deg: float = 60.0
    min_faces: int = Field(
        default=1,
        description="Merge parts smaller than this into their largest neighbour (0 = keep all).",
    )


@register
class PartIds(BaseOp):
    id = "prep.mesh.part_ids"
    version = "1.0.0"
    category = "mesh"
    summary = (
        "Recover part ids from connectivity (and optionally sharp edges) when the export merged "
        "all patches; stored as cell_fields/part_id, marked inferred."
    )
    when_to_use = (
        "merged_surfaces exports where torque/force per part cannot be verified; naming the parts "
        "is a judgment, this op only numbers them deterministically by size."
    )
    changes_fields = True
    Params = PartIdsParams

    def plan_samples(self, ref: DatasetRef, params: PartIdsParams) -> list[str]:
        return _surface_ids(ref)

    def plan_sample(self, ed: SampleEditor, params: PartIdsParams) -> list[Change]:
        if params.name in ed.field_names("cell") and not params.overwrite:
            return []
        return [Change(target=f"samples/{ed.id}/cell_fields/{params.name}", action="add")]

    def apply_sample(self, ed: SampleEditor, params: PartIdsParams, *, seed: int) -> SampleOutcome:
        if not self.plan_sample(ed, params):
            return SampleOutcome(sample_id=ed.id, skipped=True)
        s = ed.sample
        pts, faces = s.points(), s.cells.astype(np.int64)
        t = TopologyTables.build(faces)
        if params.split_by_normals:
            m = SurfaceMesh(points=pts, faces=faces.astype(np.int32))
            n = m.face_normals()
            # drop adjacency across sharp edges by marking those edges as boundary
            inv_s, face_s = t.eid_s, t.face_s
            same = inv_s[1:] == inv_s[:-1]
            a, b = face_s[:-1][same], face_s[1:][same]
            cosang = np.einsum("ij,ij->i", n[a], n[b])
            sharp = cosang < np.cos(np.deg2rad(params.angle_deg))
            keep_edge = np.ones(t.n_edges, dtype=bool)
            keep_edge[inv_s[1:][same][sharp]] = False
            # cut slots get fresh ids above the edge range so they connect nothing
            inv2 = np.where(keep_edge[t.inv], t.inv, t.n_edges + np.arange(len(t.inv)))
            labels = _components_from_slots(len(faces), inv2)
        else:
            labels = t.components()
        # order parts by size (largest first), ties by first face index
        ids, first, sizes = np.unique(labels, return_index=True, return_counts=True)
        order = np.lexsort((first, -sizes))
        remap = np.empty(len(ids), dtype=np.int64)
        remap[ids[order]] = np.arange(len(ids))
        labels = remap[labels]
        if params.min_faces > 1:
            small = np.nonzero(sizes[order] < params.min_faces)[0]
            if len(small):
                labels = np.where(np.isin(labels, small), 0, labels)
        ed.write_field(
            params.name,
            labels.astype(np.int32)[:, None],
            location="cell",
            units="-",
            meta_extra={
                "inferred": True,
                "method": "connectivity" + ("+normals" if params.split_by_normals else ""),
                "n_parts": int(labels.max() + 1),
            },
        )
        return SampleOutcome(
            sample_id=ed.id,
            changes=[
                Change(
                    target=f"samples/{ed.id}/cell_fields/{params.name}",
                    action="add",
                    detail={"n_parts": int(labels.max() + 1)},
                )
            ],
            stats={"n_parts": int(labels.max() + 1), "sizes": np.bincount(labels).tolist()[:50]},
        )
