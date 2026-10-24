# SPDX-License-Identifier: Apache-2.0
"""The Rust kernels (``frontier_native.prep``) against the numpy reference kernels.

Every comparison is bitwise: arrays by dtype, shape and bytes, audit dicts by key order,
value type and IEEE bits (a NaN matches a NaN). Bounding-box signed zeros are
numerically equal: NumPy extrema choose their sign differently across CPU backends.
Inputs: the regression meshes (seams,
non-manifold fans, degenerate and duplicate faces, flipped faces, a Mobius strip, many
components), random triangle soups over a small coordinate grid (many coincident vertices,
repeated indices, non-manifold edges), and degenerate input: empty, a single triangle,
duplicated vertices, NaN / infinite / signed-zero coordinates, float64 points, int64 faces.

Also here: backend selection (``FRONTIER_PREP_KERNELS``), what a run records about the
kernel it used, and that an op produces the same version with either backend.
"""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frontier_prep import kernels
from frontier_prep.benchmarks import make_case
from frontier_prep.ops import mesh as mk
from frontier_prep.ops import units as un

REPO = Path(__file__).resolve().parents[3]
AHMED = REPO / "datasets" / "prep-work" / "ahmedml-40-perf" / "samples" / "run_1"

_NATIVE, _WHY = kernels._load()  # noqa: SLF001
needs_native = pytest.mark.skipif(
    _NATIVE is None and os.environ.get(kernels.ENV_VAR, "").lower() != "native",
    reason=f"native kernels unavailable: {_WHY}",
)


def _regression_cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The meshes of ``test_kernels_regression.py`` (loaded by path: pytest runs with
    --import-mode=importlib, so sibling modules are not importable by name)."""
    spec = importlib.util.spec_from_file_location(
        "_regression_meshes", Path(__file__).with_name("test_kernels_regression.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.CASES)


def _soup(rng: np.random.Generator, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Random faces over ``n`` vertices on a coarse grid: coincident vertices, repeated
    indices, duplicate faces and edges with many faces all occur."""
    pts = rng.integers(-3, 4, (n, 3)).astype(np.float32) * np.float32(0.25)
    pts[rng.random(n) < 0.1] *= np.float32(-0.0)  # signed zeros
    faces = rng.integers(0, n, (m, 3)).astype(np.int32)
    return pts, faces


def _cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cases = {f"reg_{k}": v for k, v in _regression_cases().items()}
    rng = np.random.default_rng(20261012)
    f0 = np.zeros((0, 3), np.int32)
    cases["empty"] = (np.zeros((0, 3), np.float32), f0)
    cases["points_no_faces"] = (rng.random((5, 3)).astype(np.float32), f0)
    cases["single_triangle"] = (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        np.array([[0, 1, 2]], np.int32),
    )
    cases["single_degenerate"] = (
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32),
        np.array([[0, 1, 2]], np.int32),
    )
    cases["all_vertices_coincide"] = (
        np.ones((6, 3), np.float32),
        np.array([[0, 1, 2], [3, 4, 5], [0, 4, 2]], np.int32),
    )
    # every vertex duplicated: a soup whose seams all merge
    p, f = cases["reg_grid"]
    cases["duplicated_vertices"] = (
        np.concatenate([p, p]),
        np.concatenate([f, f[::-1] + len(p)]).astype(np.int32),
    )
    # an edge shared by five faces, in both directions
    cases["nonmanifold_edge"] = (
        rng.random((7, 3)).astype(np.float32),
        np.array([[0, 1, 2], [0, 1, 3], [1, 0, 4], [0, 1, 5], [1, 0, 6]], np.int32),
    )
    for k, (m, n) in enumerate([(1, 3), (4, 3), (40, 12), (300, 60), (2500, 400), (9000, 9000)]):
        cases[f"soup_{k}"] = _soup(rng, m, n)
    # more than 128 open edges and faces: the pairwise sums recurse
    p, f = _soup(rng, 700, 900)
    cases["soup_irregular"] = (rng.standard_normal((900, 3)).astype(np.float32), f)
    # float64 points and int64 faces
    p, f = cases["soup_3"]
    cases["float64_points"] = (p.astype(np.float64) * 1.1 + 0.3, f.astype(np.int64))
    # non-finite coordinates
    for name, bad in (("nan", np.nan), ("inf", np.inf), ("neg_inf", -np.inf)):
        p, f = cases["soup_3"]
        q = p.copy()
        q[rng.random(q.shape) < 0.05] = bad
        cases[f"coords_{name}"] = (q, f)
    q = cases["reg_seams"][0].copy()
    q[3] = np.nan
    q[4] = np.nan  # two identical all-NaN rows: never merged
    q[7, 1] = np.nan
    cases["seams_nan"] = (q, cases["reg_seams"][1])
    # signed zeros in every position
    z = np.zeros((8, 3), np.float32)
    z[::2] = -0.0
    z[1, 0] = 1.0
    cases["signed_zeros"] = (z, np.array([[0, 1, 2], [3, 1, 4], [5, 6, 7]], np.int32))
    return cases


CASES = _cases()
NAMES = sorted(CASES)


def _bits(x: float) -> bytes:
    return b"nan" if math.isnan(x) else struct.pack("<d", x)


def _same_value(a: Any, b: Any, where: str) -> None:
    assert type(a) is type(b), (where, type(a), type(b))
    if isinstance(a, float):
        if where.startswith("bounds[") and a == b == 0.0:
            return
        assert _bits(a) == _bits(b), (where, a, b)
    elif isinstance(a, list):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same_value(x, y, f"{where}[{i}]")
    else:
        assert a == b, (where, a, b)


def _same_dict(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert list(a) == list(b)
    for k in a:
        _same_value(a[k], b[k], k)


def _same_array(a: np.ndarray, b: np.ndarray, where: str = "") -> None:
    assert a.dtype == b.dtype, (where, a.dtype, b.dtype)
    assert a.shape == b.shape, (where, a.shape, b.shape)
    assert a.tobytes() == b.tobytes(), where


def _both(fn: Any) -> tuple[Any, Any]:
    """``fn()`` with the reference and with the native kernels."""
    with kernels.forced("python"):
        ref = fn()
    with kernels.forced("native"):
        nat = fn()
    return ref, nat


# -- parity ------------------------------------------------------------------------------------


@needs_native
@pytest.mark.parametrize("name", NAMES)
def test_edge_tables_and_components(name: str) -> None:
    _, f = CASES[name]
    ref, nat = _both(lambda: mk.TopologyTables.build(f))
    assert ref.n == nat.n
    for attr in ("faces", "order", "inv", "start", "counts", "uniq"):
        _same_array(getattr(ref, attr), getattr(nat, attr), attr)
    rc, nc = _both(lambda: mk.TopologyTables.build(f).components())
    _same_array(rc, nc, "components")


@needs_native
@pytest.mark.parametrize("name", NAMES)
def test_topology(name: str) -> None:
    p, f = CASES[name]
    with np.errstate(invalid="ignore"):
        ref, nat = _both(lambda: mk.topology(p, f))
    _same_dict(ref, nat)


@needs_native
@pytest.mark.parametrize("name", NAMES)
def test_face_areas(name: str) -> None:
    p, f = CASES[name]
    with np.errstate(invalid="ignore"):
        ref, nat = _both(lambda: mk.face_areas(p, f))
    _same_array(ref, nat)


@needs_native
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tol", [0.0, 0.5, 0.1, 1e-3])
def test_merge_vertices(name: str, tol: float) -> None:
    p, f = CASES[name]
    with np.errstate(invalid="ignore"):
        ref, nat = _both(lambda: mk.merge_vertices(p, f, tol))
    for x, y, what in zip(ref, nat, ("points", "faces", "point_map"), strict=True):
        _same_array(x, y, what)


@needs_native
def test_merge_non_representable_keys_go_to_the_reference() -> None:
    """Non-finite coordinates on a tolerance grid: numpy's float-to-int64 conversion is
    platform-defined, so the native kernel declines and the reference runs."""
    assert _NATIVE is not None
    p = np.array([[np.nan, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int64)
    assert _NATIVE.merge_vertices(p, f, 0.5) is None
    assert _NATIVE.merge_vertices(np.full((3, 3), 1e30, np.float32), f, 1e-12) is None
    assert _NATIVE.merge_vertices(p, f, 0.0) is not None  # exact keys handle NaN


@needs_native
@pytest.mark.parametrize("name", NAMES)
def test_orientation_and_duplicates(name: str) -> None:
    _, f = CASES[name]
    (rf, rn), (nf, nn) = _both(lambda: mk.orient_consistently(f))
    assert rn == nn and type(rn) is type(nn)
    _same_array(rf, nf, "oriented faces")
    rd, nd = _both(lambda: mk.duplicate_faces(f))
    _same_array(rd, nd, "duplicate mask")
    # the output of a walk is a fixed point with both kernels
    (rf2, _), (nf2, _) = _both(lambda: mk.orient_consistently(nf))
    _same_array(rf2, nf2)


@needs_native
@pytest.mark.parametrize("factor", [1.225, 3.0, 1e-3, -47.880258888889, 1e30, 0.0])
def test_scale_block(factor: float) -> None:
    rng = np.random.default_rng(5)
    special = [0.0, -0.0, 1.0, 0.1, 3e-7, 1e30, -1e30, 1e-45, 3.4e38, np.nan, np.inf, -np.inf]
    blocks: list[np.ndarray] = [
        np.array(special, np.float32),
        np.array(special + [1e300, -1e-320], np.float64),
        (rng.standard_normal(4099) * 10.0 ** rng.integers(-20, 20, 4099)).astype(np.float32),
        rng.standard_normal((7, 33, 3)).astype(np.float32),
        rng.standard_normal((50, 2)),
        np.zeros((0, 3), np.float32),
        np.asfortranarray(rng.standard_normal((6, 5)).astype(np.float32)),
        rng.standard_normal((40, 3)).astype(np.float32)[::2],  # not contiguous
        np.arange(12, dtype=np.int32).reshape(4, 3),  # integers stay with the reference
    ]
    for x in blocks:
        with np.errstate(over="ignore", invalid="ignore"):
            ref, nat = _both(lambda x=x: un.scale_block(x, factor))
        _same_array(ref, nat, f"{x.dtype} {x.shape}")


@needs_native
def test_invalid_indices_behave_like_the_reference() -> None:
    """Indices the native kernels refuse (negative, beyond the points) reach the reference,
    which decides the result or the error."""
    p = np.random.default_rng(1).random((4, 3)).astype(np.float32)
    neg = np.array([[0, 1, -1], [1, 2, 3]], np.int32)  # numpy reads -1 as the last point
    ref, nat = _both(lambda: mk.merge_vertices(p, neg))
    for x, y in zip(ref, nat, strict=True):
        _same_array(x, y)
    beyond = np.array([[0, 1, 9]], np.int32)
    for backend in ("python", "native"):
        with kernels.forced(backend), pytest.raises(IndexError):
            mk.topology(p, beyond)


@needs_native
def test_native_results_do_not_depend_on_the_thread_count(tmp_path: Path) -> None:
    """The same digest from processes with one, three and the default number of kernel
    threads (``RAYON_NUM_THREADS`` is read once per process)."""
    script = tmp_path / "digest.py"
    script.write_text(
        "import hashlib, json, numpy as np\n"
        "from frontier_prep import kernels\n"
        "from frontier_prep.ops import mesh as mk, units as un\n"
        "assert kernels.backend() == 'native'\n"
        "rng = np.random.default_rng(3)\n"
        "p = rng.integers(-40, 41, (60000, 3)).astype(np.float32) * np.float32(0.125)\n"
        "f = rng.integers(0, 60000, (150000, 3)).astype(np.int32)\n"
        "h = hashlib.sha256()\n"
        "t = mk.TopologyTables.build(f)\n"
        "for a in (t.order, t.inv, t.start, t.counts, t.uniq, t.components()):\n"
        "    h.update(a.tobytes())\n"
        "h.update(json.dumps(mk.topology(p, f)).encode())\n"
        "for tol in (0.0, 0.3):\n"
        "    for a in mk.merge_vertices(p, f, tol):\n"
        "        h.update(a.tobytes())\n"
        "h.update(mk.orient_consistently(f)[0].tobytes())\n"
        "h.update(mk.duplicate_faces(f).tobytes())\n"
        "h.update(mk.face_areas(p, f).tobytes())\n"
        "h.update(un.scale_block(p, 1.225).tobytes())\n"
        "print(kernels.threads(), h.hexdigest())\n"
    )
    seen: dict[str, str] = {}
    for threads in ("1", "3", ""):
        env = {k: v for k, v in os.environ.items() if k != "RAYON_NUM_THREADS"}
        env[kernels.ENV_VAR] = "native"
        if threads:
            env["RAYON_NUM_THREADS"] = threads
        out = subprocess.run(
            [sys.executable, str(script)], env=env, capture_output=True, text=True, check=True
        ).stdout.split()
        if threads:
            assert out[0] == threads
        seen[threads or "default"] = out[1]
    assert len(set(seen.values())) == 1, seen


# -- selection and provenance ------------------------------------------------------------------


def test_environment_variable_selects_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(kernels.ENV_VAR, "python")
    assert kernels.native() is None
    assert kernels.describe()["backend"] == "python"
    assert kernels.describe()["requested"] == "python"
    assert kernels.threads() is None
    monkeypatch.setenv(kernels.ENV_VAR, "rust")
    with pytest.raises(ValueError, match="auto, native, python"):
        kernels.native()
    monkeypatch.delenv(kernels.ENV_VAR)
    assert kernels.requested() == "auto"
    assert kernels.backend() == ("native" if _NATIVE is not None else "python")


def test_missing_native_module_falls_back_or_fails_when_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernels, "_load", lambda: (None, "frontier_native is not importable"))
    monkeypatch.delenv(kernels.ENV_VAR, raising=False)
    assert kernels.backend() == "python"
    d = kernels.describe()
    assert d["backend"] == "python" and d["version"] and d["requested"] == "auto"
    # the ops still work
    f = np.array([[0, 1, 2], [1, 3, 2], [2, 4, 3]], np.int32)
    assert mk.orient_consistently(f)[1] == 1
    monkeypatch.setenv(kernels.ENV_VAR, "native")
    with pytest.raises(kernels.KernelUnavailableError, match="frontier-py:develop"):
        kernels.native()


def test_other_spec_revision_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    fake = types.ModuleType("frontier_native")
    fake.prep = types.SimpleNamespace(SPEC_REVISION=kernels.SPEC_REVISION + 1)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "frontier_native", fake)
    kernels._load.cache_clear()  # noqa: SLF001
    try:
        monkeypatch.delenv(kernels.ENV_VAR, raising=False)
        assert kernels.backend() == "python"
    finally:
        monkeypatch.undo()
        kernels._load.cache_clear()  # noqa: SLF001


@needs_native
def test_native_is_forced_and_described(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(kernels.ENV_VAR, "native")
    d = kernels.describe()
    assert d["backend"] == "native"
    assert _NATIVE is not None and d["version"] == _NATIVE.KERNELS_VERSION
    assert d["spec_revision"] == kernels.SPEC_REVISION == _NATIVE.SPEC_REVISION
    assert (kernels.threads() or 0) >= 1


def _provenance_of(result: Any, root: Path) -> dict[str, Any]:
    from frontier_data.versions import VersionStore

    recs = [r for r in VersionStore(root).provenance("op") if r["id"] == result.provenance_id]
    assert len(recs) == 1
    return recs[0]


def _run_seam_repair(root: Path, workers: int) -> list[Any]:
    from frontier_prep.base import DatasetRef
    from frontier_prep.op import Context, get_op

    out = []
    for op_id, params in (
        ("prep.mesh.merge_seams", {}),
        ("prep.mesh.repair", {}),
        ("prep.mesh.orient_outward", {}),
        ("prep.mesh.watertight_check", {}),
        ("prep.units.convert", {"conversions": {"p": "kPa"}, "from_units": {"p": "Pa"}}),
    ):
        out.append(get_op(op_id).run(DatasetRef(root), params, ctx=Context(workers=workers)))
    return out


@needs_native
def test_ops_give_the_same_versions_with_either_backend_and_record_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seam merge, repair, orientation, audit and a unit conversion on a generated triangle
    soup: the version chain is identical with the reference kernels, with the native kernels,
    and with the native kernels in worker processes; the Result and the provenance record say
    which kernel ran and its version."""
    src = tmp_path / "src"
    make_case("seam_soup", src)  # generated once: the copies start from the same bytes
    runs: dict[str, list[Any]] = {}
    roots: dict[str, Path] = {}
    for label, backend, workers in (("py", "python", 1), ("rs", "native", 1), ("rs2", "native", 2)):
        monkeypatch.setenv(kernels.ENV_VAR, backend)
        roots[label] = tmp_path / label
        shutil.copytree(src, roots[label])
        runs[label] = _run_seam_repair(roots[label], workers)
    versions = {k: [r.output_version for r in v] for k, v in runs.items()}
    assert versions["py"] == versions["rs"] == versions["rs2"], versions
    assert len({v for v in versions["py"] if v}) >= 2  # the chain did change the data

    assert _NATIVE is not None
    for label, backend in (("py", "python"), ("rs", "native")):
        res = runs[label][0]
        assert res.machine.prep_kernels == backend
        want = _NATIVE.KERNELS_VERSION if backend == "native" else res.machine.frontier_prep
        assert res.machine.prep_kernels_version == want
        rec = _provenance_of(res, roots[label])
        assert rec["kernels"]["backend"] == backend
        assert rec["kernels"]["version"] == want
        assert rec["kernels"]["spec_revision"] == kernels.SPEC_REVISION


# -- quality.check in the worker pool ----------------------------------------------------------


def _report(root: Path) -> str:
    import json

    rep = json.loads((root / "quality" / "report.json").read_text())
    return json.dumps(
        {k: v for k, v in rep.items() if k not in ("generated_at", "elapsed_s")}, sort_keys=True
    ).replace(str(root), "<root>")


def test_quality_check_uses_the_worker_pool_and_reports_the_same(tmp_path: Path) -> None:
    """``prep.quality.check`` computes the per-sample statistics in ``apply_sample`` (worker
    processes) and merges them in sample order: assertions and the report (minus its
    wall-clock fields) equal those of one process, and equal ``run_quality`` on its own."""
    from frontier_data.dataset import Dataset
    from frontier_data.quality import run_quality
    from frontier_prep.base import DatasetRef
    from frontier_prep.op import Context, get_op

    src = tmp_path / "src"
    make_case("seam_soup", src)
    op = get_op("prep.quality.check")
    results = {}
    for workers in (1, 3):
        root = tmp_path / f"w{workers}"
        shutil.copytree(src, root)
        assert op.plan_samples(DatasetRef(root), op.coerce({})) == Dataset(root).sample_ids
        results[workers] = op.run(DatasetRef(root), {"html": False}, ctx=Context(workers=workers))
    a, b = results[1], results[3]
    # (the versions differ: the report file carries a timestamp, the op's tolerance class)
    assert [x.model_dump() for x in a.assertions] == [x.model_dump() for x in b.assertions]
    assert len(a.assertions) > 3
    assert _report(tmp_path / "w1") == _report(tmp_path / "w3")
    # the transported statistics are taken out of the outcome: nothing of them is recorded
    for res in (a, b):
        assert "_quality_sample_stats" not in str(res.stats)
        res.model_dump_json()
    # a steady surface set has no transient sample: no frame-hash assertion, as before
    assert not [x for x in a.assertions if x.check == "initial_frame_duplicates"]
    plain = tmp_path / "plain"
    shutil.copytree(src, plain)
    run_quality(Dataset(plain), write_html=False, dataset_location=".")
    assert _report(plain) == _report(tmp_path / "w1")


# -- real data (optional) ----------------------------------------------------------------------


@needs_native
@pytest.mark.skipif(
    not AHMED.exists() or os.environ.get("FRONTIER_PERF_TESTS") != "1",
    reason="AhmedML copy missing or FRONTIER_PERF_TESTS != 1",
)
def test_real_ahmedml_mesh_matches_reference() -> None:
    from frontier_data.arrays import Sample

    s = Sample.open(AHMED, "run_1")
    p, f = np.asarray(s.points()), np.asarray(s.cells)
    ref, nat = _both(lambda: mk.topology(p, f))
    _same_dict(ref, nat)
    rt, nt = _both(lambda: mk.TopologyTables.build(f))
    for attr in ("order", "inv", "start", "counts", "uniq"):
        _same_array(getattr(rt, attr), getattr(nt, attr), attr)
    for tol in (0.0, 1e-4):
        rm, nm = _both(lambda tol=tol: mk.merge_vertices(p, f, tol))
        for x, y in zip(rm, nm, strict=True):
            _same_array(x, y)
    merged = mk.merge_vertices(p, f)
    ref, nat = _both(lambda: mk.topology(merged[0], merged[1]))
    _same_dict(ref, nat)
    rng = np.random.default_rng(0)
    ff = merged[1].copy()
    sel = rng.random(len(ff)) < 0.3
    ff[sel] = ff[sel][:, [0, 2, 1]]
    (rf, rn), (nf, nn) = _both(lambda: mk.orient_consistently(ff))
    assert rn == nn
    _same_array(rf, nf)
    rd, nd = _both(lambda: mk.duplicate_faces(f))
    _same_array(rd, nd)
    ra, na = _both(lambda: mk.face_areas(p, f))
    _same_array(ra, na)
    x = np.asarray(s.field(s.field_names("cell")[0], location="cell"))
    rs, ns = _both(lambda: un.scale_block(x, 1.225))
    _same_array(rs, ns)
