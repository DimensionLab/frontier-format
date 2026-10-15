# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.metadata
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from frontier_plugins import check_entry_point
from frontier_prep.base import DatasetRef
from frontier_prep.op import Context, get_op
from frontier_prep.ops import smoothing


@pytest.fixture
def triangle():
    return np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), np.array([[0, 1, 2]])


def test_known_triangle_two_passes_and_fixed_component_timing(triangle):
    vertices, faces = triangle
    fixed = np.zeros(vertices.shape, dtype=bool)
    fixed[0, 0] = True
    # A triangle's uniform operator has off-diagonal entries 1/2.
    operator = (np.ones((3, 3)) - np.eye(3)) / 2
    first = vertices + 0.75 * (operator @ vertices - vertices)
    expected = first - 0.76 * (operator @ first - first)
    expected[fixed] = vertices[fixed]
    result, metadata = smoothing.taubin_smooth(vertices, faces, fixed=fixed, iterations=1)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(vertices, [[0, 0, 0], [2, 0, 0], [0, 2, 0]])
    assert result.dtype == np.float64 and metadata["backend"] == "cpu"
    assert metadata["timing_s"]["total"] >= 0
    assert metadata["versions"]["scipy"]


def test_translation_equivariance_and_fixed_all_identity(triangle):
    v, f = triangle
    a, _ = smoothing.taubin_smooth(v, f, iterations=4)
    b, _ = smoothing.taubin_smooth(v + [4, 8, -2], f, iterations=4)
    np.testing.assert_allclose(b, a + [4, 8, -2], rtol=1e-13, atol=1e-13)
    held, _ = smoothing.taubin_smooth(v, f, fixed=np.ones(v.shape, bool), iterations=20)
    np.testing.assert_array_equal(held, v)


def test_empty_faces_and_zero_iterations_identity(triangle):
    v, f = triangle
    for faces, iterations in ((np.empty((0, 3), dtype=int), 20), (f, 0)):
        result, _ = smoothing.taubin_smooth(v, faces, iterations=iterations)
        np.testing.assert_array_equal(result, v)
        assert not np.shares_memory(result, v)


def test_cache_by_content_not_pointer_and_immutable(triangle):
    v, f = triangle
    a, key, _ = smoothing._operator(len(v), f)
    b, same_key, hit = smoothing._operator(len(v), f.copy())
    assert a is b and hit and key == same_key
    with pytest.raises(ValueError):
        a.data[0] = 0
    changed = f.copy()
    changed[0] = [0, 2, 1]
    _, different_key, _ = smoothing._operator(len(v), changed)
    assert different_key != key
    for n in range(5, 12):
        smoothing._operator(n, f)
    assert len(smoothing._CACHE) <= smoothing._CACHE_ENTRIES
    assert sum(map(smoothing._operator_bytes, smoothing._CACHE.values())) <= smoothing._CACHE_LIMIT


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": -1},
        {"iterations": 1.5},
        {"iterations": True},
        {"lam": 0},
        {"lam": float("nan")},
        {"mu": 0},
        {"mu": -1.1},
        {"backend": "gpu"},
    ],
)
def test_parameter_validation(triangle, kwargs):
    with pytest.raises(ValidationError):
        smoothing.taubin_smooth(*triangle, **kwargs)


@pytest.mark.parametrize("fixed", [np.zeros((3, 1), bool), np.zeros((3, 3)), np.zeros(3, bool)])
def test_reject_ambiguous_fixed_masks(triangle, fixed):
    with pytest.raises(ValueError, match="coordinate-component"):
        smoothing.taubin_smooth(*triangle, fixed=fixed)


def test_reject_invalid_mesh(triangle):
    v, _ = triangle
    for f in (np.array([[0, 0, 1]]), np.array([[0, 1, 3]]), np.array([[0.0, 1.0, 2.0]])):
        with pytest.raises(ValueError):
            smoothing.taubin_smooth(v, f)
    with pytest.raises(ValueError, match="finite"):
        smoothing.taubin_smooth(v * np.nan, np.array([[0, 1, 2]]))


def test_auto_fallback_is_observable_but_cuda_is_required(triangle, monkeypatch):
    def unavailable():
        raise smoothing.CudaUnavailableError("CUDA driver unavailable for test")

    monkeypatch.setattr(smoothing, "_cuda_modules", unavailable)
    _, metadata = smoothing.taubin_smooth(*triangle, backend="auto")
    assert metadata["backend"] == "cpu"
    assert "driver unavailable" in metadata["fallback_reason"]
    with pytest.raises(RuntimeError, match="driver unavailable"):
        smoothing.taubin_smooth(*triangle, backend="cuda")


def test_execution_failure_never_falls_back(triangle, monkeypatch):
    runtime = SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda _: {"name": b"test"},
        runtimeGetVersion=lambda: 12000,
        driverGetVersion=lambda: 12000,
    )

    def allocation_failure(_):
        raise MemoryError("GPU out of memory")

    cp = SimpleNamespace(
        __version__="test", cuda=SimpleNamespace(runtime=runtime), asarray=allocation_failure
    )
    monkeypatch.setattr(smoothing, "_cuda_modules", lambda: (cp, SimpleNamespace(csr_matrix=None)))
    with pytest.raises(MemoryError, match="out of memory"):
        smoothing.taubin_smooth(*triangle, backend="auto")


def test_unknown_preflight_error_never_falls_back(triangle, monkeypatch):
    def broken_context():
        raise RuntimeError("unrecognized CUDA context failure")

    monkeypatch.setattr(smoothing, "_cuda_modules", broken_context)
    with pytest.raises(RuntimeError, match="unrecognized"):
        smoothing.taubin_smooth(*triangle, backend="auto")


def test_real_cuda_parity(triangle):
    try:
        smoothing._cuda_modules()
    except smoothing.CudaUnavailableError as exc:
        pytest.skip(str(exc))
    v, f = triangle
    fixed = np.zeros(v.shape, bool)
    fixed[0, 0] = True
    cpu, _ = smoothing.taubin_smooth(v, f, fixed=fixed, iterations=200)
    gpu, metadata = smoothing.taubin_smooth(v, f, fixed=fixed, iterations=200, backend="cuda")
    np.testing.assert_allclose(gpu, cpu, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(gpu[fixed], v[fixed])
    assert metadata["backend"] == "cuda" and metadata["versions"]["cupy"]


def test_pipeline_plan_run_and_provenance(case):
    root = case("naming")
    ref = DatasetRef(root)
    ed = ref.editor(ref.sample_ids()[0])
    before_faces = ed.sample.cells.copy()
    field = ed.sample.field_names("cell")[0]
    before_field = ed.field(field, location="cell").copy()
    mask = np.ones(ed.sample.group["points"].shape, dtype=bool)
    mask[1:, 1:] = False
    ed.write_field("held", mask, location="point", dims=["point", "component"])
    op = get_op("prep.mesh.taubin")
    params = {"iterations": 2, "fixed_mask_field": "held"}
    plan = op.plan(root, params)
    assert plan.changes and plan.params["backend"] == "cpu"
    result = op.run(root, params, ctx=Context(workers=1))
    assert result.output_version != result.input_version
    after = DatasetRef(root).editor(ed.id)
    np.testing.assert_array_equal(after.sample.cells, before_faces)
    np.testing.assert_array_equal(after.field(field, location="cell"), before_field)
    assert after.sample.group["points"].dtype == np.float64
    provenance = after.meta.extra["geometry"]["taubin"]
    assert provenance["backend"] == "cpu" and "timing_s" not in provenance
    assert result.stats["samples"][ed.id]["timing_s"]["total"] >= 0
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_manifest_declares_compatible_independent_ops_api():
    dist = importlib.metadata.distribution("frontier-prep")
    ep = next(e for e in dist.entry_points if e.group == "frontier.ops")
    compatibility = check_entry_point(ep)
    assert compatibility.compatible, compatibility.errors
    from frontier_plugins import manifest_for_distribution

    manifest = manifest_for_distribution(dist)
    assert manifest is not None and manifest["api"] == {"ops": ">=1,<2"}
