# SPDX-License-Identifier: Apache-2.0
"""The optimised topology kernels are bitwise identical to the pre-optimisation reference
(``_reference_kernels.py``) on meshes with seams, non-manifold edges, degenerate faces,
duplicate faces, flipped faces, a non-orientable strip, many components, and on one real
AhmedML mesh when the dataset copy is present. The explicit test vectors are the ones quoted
in the Rust port specification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frontier_prep.ops import mesh as mk


def _load_reference() -> Any:
    """The pre-optimisation kernels, loaded by path (pytest runs with --import-mode=importlib,
    so sibling modules are not importable by name)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_reference_kernels", Path(__file__).with_name("_reference_kernels.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load_reference()

REPO = Path(__file__).resolve().parents[3]
AHMED = REPO / "datasets" / "prep-work" / "ahmedml-40-perf" / "samples" / "run_1"


# -- synthetic meshes --------------------------------------------------------------------------


def _grid_soup(
    nx: int, ny: int, *, seams: bool, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A planar grid of quads split into triangles; with ``seams`` every patch of 2x2 quads
    carries its own copy of its vertices (triangle soup with coincident vertices)."""
    pts: list[np.ndarray] = []
    faces: list[list[int]] = []
    if not seams:
        xs, ys = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
        p = np.stack([xs.ravel(), ys.ravel(), 0.1 * np.sin(xs.ravel())], axis=1)
        idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
        for i in range(nx):
            for j in range(ny):
                a, b, c, d = idx[i, j], idx[i + 1, j], idx[i, j + 1], idx[i + 1, j + 1]
                faces += [[a, b, d], [a, d, c]]
        return p.astype(np.float32), np.asarray(faces, dtype=np.int32)
    for i0 in range(0, nx, 2):
        for j0 in range(0, ny, 2):
            base = sum(len(q) for q in pts)
            ii, jj = np.meshgrid(np.arange(i0, i0 + 3), np.arange(j0, j0 + 3), indexing="ij")
            p = np.stack([ii.ravel(), jj.ravel(), 0.1 * np.sin(ii.ravel())], axis=1)
            pts.append(p)
            loc = np.arange(9).reshape(3, 3) + base
            for i in range(2):
                for j in range(2):
                    a, b, c, d = loc[i, j], loc[i + 1, j], loc[i, j + 1], loc[i + 1, j + 1]
                    faces += [[a, b, d], [a, d, c]]
    p = np.concatenate(pts).astype(np.float32)
    perm = rng.permutation(len(faces))
    return p, np.asarray(faces, dtype=np.int32)[perm]


def _tetra() -> tuple[np.ndarray, np.ndarray]:
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    f = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], dtype=np.int32)
    return p, f


def _mobius(n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """A non-orientable strip: the parity graph has an odd cycle, so the result depends on
    the BFS tree (the reference and the kernel must still agree)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inner = np.stack(
        [
            np.cos(t) * (1 - 0.3 * np.cos(t / 2)),
            np.sin(t) * (1 - 0.3 * np.cos(t / 2)),
            -0.3 * np.sin(t / 2),
        ],
        1,
    )
    outer = np.stack(
        [
            np.cos(t) * (1 + 0.3 * np.cos(t / 2)),
            np.sin(t) * (1 + 0.3 * np.cos(t / 2)),
            0.3 * np.sin(t / 2),
        ],
        1,
    )
    p = np.concatenate([inner, outer]).astype(np.float32)
    faces = []
    for k in range(n):
        a, b = k, (k + 1) % n
        if k == n - 1:  # the half twist: the last strip connects to the swapped edge
            faces += [[a, n + a, b], [n + a, n + b, b]]
            faces[-2] = [a, n + a, n + b]
            faces[-1] = [a, n + b, b]
        else:
            faces += [[a, n + a, n + b], [a, n + b, b]]
    return p, np.asarray(faces, dtype=np.int32)


def _cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(3)
    cases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cases["tetra"] = _tetra()
    cases["grid"] = _grid_soup(6, 5, seams=False, rng=rng)
    p, f = _grid_soup(8, 6, seams=True, rng=rng)
    cases["seams"] = (p, f)
    # flipped faces: a third of the seam soup reversed
    ff = f.copy()
    sel = rng.random(len(ff)) < 0.33
    ff[sel] = ff[sel][:, [0, 2, 1]]
    cases["seams_flipped"] = (p, ff)
    # non-manifold: a fin glued onto an interior edge of the plain grid
    p2, f2 = cases["grid"]
    apex = np.array([[2.5, 2.5, 3.0]], dtype=np.float32)
    fin = np.array([[f2[10, 0], f2[10, 1], len(p2)]], dtype=np.int32)
    cases["nonmanifold"] = (np.concatenate([p2, apex]), np.concatenate([f2, fin]))
    # degenerate: repeated-index faces and a zero-area sliver, plus duplicate faces
    f3 = np.concatenate([f2, [[3, 3, 7]], [[1, 2, 2]], f2[:4], f2[5:7][:, [0, 2, 1]]])
    p3 = p2.copy()
    cases["degenerate_dup"] = (p3, f3.astype(np.int32))
    cases["mobius"] = _mobius()
    # many components: disjoint tetrahedra, some flipped, in shuffled face order
    pts, fcs = [], []
    for k in range(7):
        tp, tf = _tetra()
        tf = tf + 4 * k
        if k % 2:
            tf = tf[:, [0, 2, 1]]
        pts.append(tp + np.array([3.0 * k, 0, 0], dtype=np.float32))
        fcs.append(tf)
    f4 = np.concatenate(fcs)[rng.permutation(28)]
    cases["components"] = (np.concatenate(pts), f4.astype(np.int32))
    return cases


CASES = _cases()


def _assert_same_topology(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k], (k, a[k], b[k])


@pytest.mark.parametrize("name", sorted(CASES))
def test_topology_matches_reference(name: str) -> None:
    p, f = CASES[name]
    _assert_same_topology(mk.topology(p, f), ref.topology(p, f))


@pytest.mark.parametrize("name", sorted(CASES))
def test_components_match_reference(name: str) -> None:
    p, f = CASES[name]
    uniq, inv, counts = ref._edges(f)
    t = mk.TopologyTables.build(f)
    assert np.array_equal(t.uniq, uniq)
    assert np.array_equal(t.inv, inv)
    assert np.array_equal(t.counts, counts)
    assert np.array_equal(t.components(), ref._components(len(p), f, uniq, inv))


@pytest.mark.parametrize("name", sorted(CASES))
def test_orient_matches_reference(name: str) -> None:
    p, f = CASES[name]
    new, n_new = mk.orient_consistently(f)
    old, n_old = ref.orient_consistently(f)
    assert n_new == n_old
    assert np.array_equal(new, old)
    assert new.dtype == old.dtype == np.int32


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("tol", [0.0, 0.5])
def test_merge_matches_reference(name: str, tol: float) -> None:
    p, f = CASES[name]
    a = mk.merge_vertices(p, f, tol)
    b = ref.merge_vertices(p, f, tol)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
        assert x.dtype == y.dtype


@pytest.mark.parametrize("name", sorted(CASES))
def test_duplicate_faces_and_loops_match_reference(name: str) -> None:
    p, f = CASES[name]
    assert np.array_equal(mk.duplicate_faces(f), ref.duplicate_faces(f))
    assert mk._boundary_loops(f) == ref._boundary_loops(f)


def test_tables_reused_across_kernels() -> None:
    p, f = CASES["seams_flipped"]
    t = mk.TopologyTables.build(f)
    assert mk.topology(p, f, t) == mk.topology(p, f)
    assert np.array_equal(mk.orient_consistently(f, t)[0], mk.orient_consistently(f)[0])
    assert mk._boundary_loops(f, t) == mk._boundary_loops(f)


def test_empty_mesh() -> None:
    p = np.zeros((0, 3), dtype=np.float32)
    f = np.zeros((0, 3), dtype=np.int32)
    assert mk.topology(p, f) == ref.topology(p, f)
    assert mk.orient_consistently(f)[1] == 0
    assert mk.TopologyTables.build(f).n_edges == 0


# -- explicit test vectors (quoted in the Rust port specification) ---------------------------------
# Every vector is also checked against the reference kernels, so the quoted numbers are the
# verified behaviour of the pre-optimisation code.


def _same(a: tuple[Any, ...], b: tuple[Any, ...]) -> None:
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            assert x.dtype == y.dtype


def _check_ref(points: np.ndarray, f: np.ndarray) -> None:
    _assert_same_topology(mk.topology(points, f), ref.topology(points, f))
    _same(mk.orient_consistently(f), ref.orient_consistently(f))
    uniq, inv, counts = ref._edges(f)
    t = mk.TopologyTables.build(f)
    assert np.array_equal(t.uniq, uniq) and np.array_equal(t.inv, inv)
    assert np.array_equal(t.counts, counts)


def test_vector_edge_tables_two_triangles() -> None:
    f = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    _check_ref(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32), f)
    t = mk.TopologyTables.build(f)
    assert t.n == 4
    assert t.uniq.tolist() == [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]
    assert t.counts.tolist() == [1, 1, 2, 1, 1]
    assert t.inv.tolist() == [0, 2, 2, 3, 1, 4]
    assert t.order.tolist() == [0, 4, 1, 2, 3, 5]
    assert t.inconsistent_edges() == 0
    assert t.components().tolist() == [0, 0]
    assert t.used_points() == 4
    top = mk.topology(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32), f)
    assert (top["open_edges"], top["nonmanifold_edges"], top["euler"]) == (4, 0, 1)
    assert top["boundary_length"] == 4.0


def test_vector_edge_tables_same_winding() -> None:
    f = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    _check_ref(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32), f)
    t = mk.TopologyTables.build(f)
    assert t.inconsistent_edges() == 1
    assert mk.orient_consistently(f)[0].tolist() == [[0, 1, 2], [1, 3, 2]]


def test_vector_edge_tables_nonmanifold_fan() -> None:
    f = np.array([[0, 1, 2], [0, 1, 3], [1, 0, 4]], dtype=np.int32)
    _check_ref(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]], np.float32), f)
    t = mk.TopologyTables.build(f)
    assert t.uniq.tolist() == [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]]
    assert t.counts.tolist() == [3, 1, 1, 1, 1, 1, 1]
    assert t.inconsistent_edges() == 1  # (0,1) traversed forward by faces 0 and 1
    assert t.components().tolist() == [0, 0, 0]
    # no manifold edge: nothing is flipped
    assert mk.orient_consistently(f)[1] == 0


def test_vector_merge_seam() -> None:
    p = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
    )
    f = np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int32)
    np_, nf, pmap = mk.merge_vertices(p, f)
    _same((np_, nf, pmap), ref.merge_vertices(p, f))
    assert np_.tolist() == [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    assert pmap.tolist() == [0, 2, 1, 2, 1, 3]
    assert nf.tolist() == [[0, 2, 1], [2, 3, 1]]
    assert mk.topology(np_, nf)["open_edges"] == 4


def test_vector_merge_tolerance_and_collapse() -> None:
    p = np.array([[0, 0, 0], [0.2, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    f = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    np_, nf, pmap = mk.merge_vertices(p, f, tol=0.5)
    # keys round(p/0.5): rows 0 and 1 both quantise to (0,0,0); groups sorted by key
    assert pmap.tolist() == [0, 0, 1, 2]
    assert np_.tolist() == [[0, 0, 0], [0, 1, 0], [1, 0, 0]]
    assert nf.tolist() == [[0, 2, 1]]  # face 0 collapsed
    _same(mk.merge_vertices(p, f, tol=0.5), ref.merge_vertices(p, f, tol=0.5))


def test_vector_orient_flipped_strip() -> None:
    f = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]], dtype=np.int32)
    ff = f.copy()
    ff[2] = ff[2][[0, 2, 1]]  # [2, 4, 3]
    out, n = mk.orient_consistently(ff)
    _same((out, np.int64(n)), ref.orient_consistently(ff))
    assert n == 1
    assert out.tolist() == f.tolist()
    # flipping the root flips everything else instead (root keeps its winding)
    ff2 = f.copy()
    ff2[0] = ff2[0][[0, 2, 1]]
    out2, n2 = mk.orient_consistently(ff2)
    _same((out2, np.int64(n2)), ref.orient_consistently(ff2))
    assert n2 == 3
    assert out2.tolist() == [[0, 2, 1], [1, 2, 3], [2, 4, 3], [3, 4, 5]]


def test_vector_orient_two_components() -> None:
    # faces 0 and 3 both traverse edge (4,5) as 5->4: face 3 (not the root) is flipped
    f = np.array([[4, 6, 5], [0, 1, 2], [1, 3, 2], [4, 7, 5]], dtype=np.int32)
    out, n = mk.orient_consistently(f)
    _same((out, np.int64(n)), ref.orient_consistently(f))
    assert n == 1
    assert out.tolist() == [[4, 6, 5], [0, 1, 2], [1, 3, 2], [4, 5, 7]]


def test_vector_duplicate_faces() -> None:
    f = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 3], [0, 2, 1]], dtype=np.int32)
    assert mk.duplicate_faces(f).tolist() == [False, True, False, True]
    assert np.array_equal(mk.duplicate_faces(f), ref.duplicate_faces(f))


# -- real data (optional) ----------------------------------------------------------------------


@pytest.mark.skipif(
    not AHMED.exists() or os.environ.get("FRONTIER_PERF_TESTS") != "1",
    reason="AhmedML copy missing or FRONTIER_PERF_TESTS != 1",
)
def test_real_ahmedml_mesh_matches_reference() -> None:
    from frontier_data.arrays import Sample

    s = Sample.open(AHMED, "run_1")
    p, f = s.points(), s.cells
    _assert_same_topology(mk.topology(p, f), ref.topology(p, f))
    a = mk.merge_vertices(p, f)
    b = ref.merge_vertices(p, f)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    rng = np.random.default_rng(0)
    ff = a[1].copy()
    sel = rng.random(len(ff)) < 0.3
    ff[sel] = ff[sel][:, [0, 2, 1]]
    new, n_new = mk.orient_consistently(ff)
    old, n_old = ref.orient_consistently(ff)
    assert n_new == n_old and np.array_equal(new, old)
