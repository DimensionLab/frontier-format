# SPDX-License-Identifier: Apache-2.0
"""Reference (pre-optimisation) topology kernels, copied verbatim from
``frontier_prep.ops.mesh`` before the performance pass. ``test_kernels_regression.py``
compares the current kernels against these so the comparison is honest: nothing here is
imported by the package."""

from __future__ import annotations

from typing import Any

import numpy as np

from frontier_data.mesh import SurfaceMesh


def _edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    n = int(f.max()) + 1 if f.size else 1
    key = e[:, 0] * n + e[:, 1]
    ukey, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    uniq = np.column_stack([ukey // n, ukey % n])
    return uniq, inv.reshape(-1), counts


def _signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    if p.shape[1] != 3:
        return 0.0
    # The one line that is not the verbatim pre-optimisation code: the original
    # ``np.einsum("ij,ij->i", a, np.cross(b, c))`` accumulates with fused multiply-add in a
    # SIMD-width-dependent order, so its last bits differ between machines. The dot product
    # is spelled out (three products added left to right), as in ``frontier_prep.ops.mesh``.
    a, x = p[faces[:, 0]], np.cross(p[faces[:, 1]], p[faces[:, 2]])
    return float(np.sum((a[:, 0] * x[:, 0] + a[:, 1] * x[:, 1]) + a[:, 2] * x[:, 2]) / 6.0)


def _components(n_points: int, faces: np.ndarray, edges: np.ndarray, inv: np.ndarray) -> np.ndarray:
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    m = len(faces)
    face_of = np.tile(np.arange(m), 3)
    order = np.argsort(inv, kind="stable")
    inv_s, face_s = inv[order], face_of[order]
    same = inv_s[1:] == inv_s[:-1]
    a, b = face_s[:-1][same], face_s[1:][same]
    g = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m)).tocsr()
    _, labels = connected_components(g, directed=False)
    _, first = np.unique(labels, return_index=True)
    remap = np.empty_like(labels)
    remap[labels[np.sort(first)]] = np.arange(len(first))
    return remap[labels]


def topology(points: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return {"n_points": int(len(points)), "n_faces": 0}
    uniq, inv, counts = _edges(f)
    open_edges = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    d = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    n_pts = int(f.max()) + 1
    _, dcounts = np.unique(d[:, 0] * n_pts + d[:, 1], return_counts=True)
    inconsistent = int((dcounts > 1).sum())
    mesh = SurfaceMesh(points=points, faces=f.astype(np.int32))
    degenerate = int(mesh.degenerate_faces().sum())
    comp = _components(len(points), f, uniq, inv)
    n_comp = int(comp.max() + 1) if comp.size else 0
    used = np.unique(f)
    euler = int(len(used) - len(uniq) + len(f))
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
        "area": float(mesh.face_areas().sum()),
        "boundary_length": boundary_len,
        "bounds": [float(x) for x in mesh.bounds],
        "signed_volume": float(_signed_volume(points, f)),
    }


def orient_consistently(faces: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import sparse
    from scipy.sparse.csgraph import breadth_first_order

    f = np.asarray(faces, dtype=np.int64).copy()
    m = len(f)
    if m == 0:
        return f.astype(np.int32), 0
    uniq, inv, counts = _edges(f)
    face_of = np.tile(np.arange(m), 3)
    slot = np.repeat(np.arange(3), m)
    order = np.argsort(inv, kind="stable")
    inv_s, face_s, slot_s = inv[order], face_of[order], slot[order]
    same = (inv_s[1:] == inv_s[:-1]) & (counts[inv_s[1:]] == 2)
    a, b = face_s[:-1][same], face_s[1:][same]
    sa, sb = slot_s[:-1][same], slot_s[1:][same]
    da = f[a, sa] == f[b, sb]
    g = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m)).tocsr()
    g = g + g.T
    val = np.where(da, 2.0, 1.0)
    rel = sparse.coo_matrix((val, (a, b)), shape=(m, m)).tocsr()
    rel = rel + rel.T
    rel_lil = rel.tolil()
    flip = np.zeros(m, dtype=bool)
    visited = np.zeros(m, dtype=bool)
    for start in range(m):
        if visited[start]:
            continue
        bfs, pred = breadth_first_order(g, start, directed=False, return_predecessors=True)
        visited[bfs] = True
        for node in bfs[1:]:
            p = pred[node]
            disagree = rel_lil[p, node] == 2.0
            flip[node] = flip[p] ^ bool(disagree)
    n_flipped = int(flip.sum())
    f[flip] = f[flip][:, [0, 2, 1]]
    return f.astype(np.int32), n_flipped


def merge_vertices(
    points: np.ndarray, faces: np.ndarray, tol: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(points)
    key = p if tol <= 0 else np.round(p / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    new_points = p[first]
    f = inverse[np.asarray(faces, dtype=np.int64)]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return new_points.astype(np.float32), f[keep].astype(np.int32), inverse


def duplicate_faces(faces: np.ndarray) -> np.ndarray:
    """Mask of faces that repeat an earlier face's vertex set (``RepairMesh`` inline code)."""
    key = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    dup = np.ones(len(faces), dtype=bool)
    dup[first] = False
    return dup


def _boundary_loops(faces: np.ndarray) -> list[list[int]]:
    uniq, _, counts = _edges(faces)
    b = uniq[counts == 1]
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
