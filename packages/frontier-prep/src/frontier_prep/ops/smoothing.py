# SPDX-License-Identifier: Apache-2.0
"""Uniform-adjacency Taubin smoothing with explicit CPU/CUDA execution.

Coordinates and CSR weights use float64. Fixed coordinate components are restored
after the lambda/mu pair, not between its two passes. No extraction, remeshing,
field interpolation or curvature estimator is part of this operation.
"""

from __future__ import annotations

import hashlib
import importlib
from collections import OrderedDict
from threading import Lock
from time import perf_counter
from typing import Any, Literal

import numpy as np
import scipy
import scipy.sparse as sp
from pydantic import Field

from frontier_prep.base import BaseOp, DatasetRef, SampleOutcome
from frontier_prep.io import SampleEditor
from frontier_prep.op import Change, OpWarning, Params, register

Backend = Literal["cpu", "cuda", "auto"]
_CACHE_LIMIT = 64 * 1024 * 1024
_CACHE_ENTRIES = 4
_CACHE: OrderedDict[str, Any] = OrderedDict()
_CACHE_LOCK = Lock()


class TaubinParams(Params):
    backend: Backend = "cpu"
    iterations: int = Field(default=200, ge=0, le=10000, strict=True)
    lam: float = Field(default=0.75, gt=0, le=1, allow_inf_nan=False)
    mu: float = Field(default=-0.76, ge=-1, lt=0, allow_inf_nan=False)
    fixed_mask_field: str | None = Field(
        default=None,
        min_length=1,
        description="Static boolean point field (N,3); True holds that coordinate component.",
    )


def _inputs(vertices: Any, faces: Any, fixed: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v, f = np.asarray(vertices), np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or v.dtype.kind not in "fiu":
        raise ValueError("vertices must be a real numeric (N,3) array")
    if not np.isfinite(v).all():
        raise ValueError("vertices must be finite")
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu":
        raise ValueError("faces must be an integer (M,3) array")
    if f.size and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError("face vertex index is out of range")
    if f.size and np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])):
        raise ValueError("triangles must have three distinct vertex indices")
    mask = np.zeros(v.shape, dtype=bool) if fixed is None else np.asarray(fixed)
    if mask.dtype != np.bool_ or mask.shape != v.shape:
        raise ValueError("fixed must be a boolean (N,3) coordinate-component mask")
    return (
        np.array(v, dtype=np.float64, order="C", copy=True),
        np.array(f, dtype=np.int64, order="C", copy=True),
        np.array(mask, dtype=bool, order="C", copy=True),
    )


def _operator(n: int, faces: np.ndarray) -> tuple[Any, str, bool]:
    digest = hashlib.sha256(f"uniform-csr-v1:{n}:{len(faces)}:".encode())
    digest.update(faces.tobytes())
    key = digest.hexdigest()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached, key, True
    rows = faces[:, [0, 1, 1, 2, 2, 0]].ravel()
    cols = faces[:, [1, 0, 2, 1, 0, 2]].ravel()
    adjacency: Any = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    operator = (sp.diags(1.0 / degree) @ adjacency).tocsr()
    for array in (operator.data, operator.indices, operator.indptr):
        array.flags.writeable = False
    size = _operator_bytes(operator)
    if size <= _CACHE_LIMIT:
        with _CACHE_LOCK:
            _CACHE[key] = operator
            while (
                len(_CACHE) > _CACHE_ENTRIES
                or sum(map(_operator_bytes, _CACHE.values())) > _CACHE_LIMIT
            ):
                _CACHE.popitem(last=False)
    return operator, key, False


def _operator_bytes(operator: Any) -> int:
    return sum(a.nbytes for a in (operator.data, operator.indices, operator.indptr))


class CudaUnavailableError(RuntimeError):
    """CUDA cannot be used on this host; eligible for an explicit auto fallback."""


def _cuda_modules() -> tuple[Any, Any]:
    """Availability only: execution errors after this check must propagate."""
    try:
        cp = importlib.import_module("cupy")
        sparse = importlib.import_module("cupyx.scipy.sparse")
    except (ImportError, OSError) as exc:
        raise CudaUnavailableError(
            "CUDA smoothing unavailable: install frontier-prep[gpu] on Linux with a "
            f"CUDA 12 runtime and compatible NVIDIA driver. {type(exc).__name__}: {exc}"
        ) from exc
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise CudaUnavailableError("CUDA smoothing unavailable: no CUDA devices found")
        cp.cuda.get_current_stream().synchronize()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        # Driver/context initialization failures mean unavailable hardware. Memory,
        # illegal-access and all other execution errors must not turn into a CPU run.
        if exc.status not in {3, 35, 100, 802, 803, 804}:
            raise
        raise CudaUnavailableError(f"CUDA smoothing unavailable: {exc}") from exc
    return cp, sparse


def _iterate(value: Any, operator: Any, fixed: Any, params: TaubinParams) -> Any:
    held = value[fixed].copy()
    for _ in range(params.iterations):
        value += params.lam * (operator @ value - value)
        value += params.mu * (operator @ value - value)
        value[fixed] = held
    return value


def taubin_smooth(
    vertices: Any,
    faces: Any,
    *,
    fixed: Any = None,
    iterations: int = 200,
    lam: float = 0.75,
    mu: float = -0.76,
    backend: Backend = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return new float64 vertices and execution metadata; inputs are never mutated.

    ``auto`` falls back only when CUDA is unavailable before execution. CUDA
    allocation, compilation and computation failures propagate. No faces means
    identity. For mixed connected/isolated meshes, zero-degree rows retain the
    uniform-operator definition (zero neighbor average); explicitly fix unused
    vertices if they should remain unchanged. GPU timings include transfers and
    synchronize the current stream. The bounded host cache contains immutable CSR
    arrays keyed by topology contents, never pointers; no device arrays are cached.
    """
    started = perf_counter()
    params = TaubinParams(backend=backend, iterations=iterations, lam=lam, mu=mu)
    v, f, mask = _inputs(vertices, faces, fixed)
    cp = sparse = None
    reason = None
    actual = "cpu"
    if backend != "cpu":
        try:
            cp, sparse = _cuda_modules()
            actual = "cuda"
        except CudaUnavailableError as exc:
            if backend == "cuda":
                raise
            reason = str(exc)
    initialized = perf_counter()
    operator, topology_hash, hit = _operator(len(v), f)
    prepared = perf_counter()
    versions = {"numpy": np.__version__, "scipy": scipy.__version__}
    device: dict[str, Any] = {}
    if cp is not None and sparse is not None:
        versions["cupy"] = cp.__version__
        device_id = cp.cuda.runtime.getDevice()
        name = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
        device = {
            "id": device_id,
            "name": name.decode() if isinstance(name, bytes) else str(name),
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
            "driver": cp.cuda.runtime.driverGetVersion(),
        }
        gpu_operator = sparse.csr_matrix(
            (cp.asarray(operator.data), cp.asarray(operator.indices), cp.asarray(operator.indptr)),
            shape=operator.shape,
        )
        result = cp.asarray(v)
        gpu_mask = cp.asarray(mask)
        if len(f):
            result = _iterate(result, gpu_operator, gpu_mask, params)
        cp.cuda.get_current_stream().synchronize()
        output = cp.asnumpy(result)
        cp.cuda.get_current_stream().synchronize()
    else:
        output = _iterate(v, operator, mask, params) if len(f) else v
    finished = perf_counter()
    if not np.isfinite(output).all():
        raise FloatingPointError("Taubin smoothing produced nonfinite coordinates")
    return output, {
        "algorithm": "uniform-csr-taubin-v1",
        "requested_backend": backend,
        "backend": actual,
        "fallback_reason": reason,
        "versions": versions,
        "device": device,
        "dtype": "float64",
        "iterations": iterations,
        "lam": lam,
        "mu": mu,
        "fixed_components": int(mask.sum()),
        "boundary_policy": "restore fixed components after each lambda/mu pair",
        "topology_sha256": topology_hash,
        "host_cache_hit": hit,
        "timing_s": {
            "validation_and_backend": initialized - started,
            "host_operator": prepared - initialized,
            "execution_with_transfers": finished - prepared,
            "total": finished - started,
        },
    }


@register
class TaubinSmooth(BaseOp):
    id = "prep.mesh.taubin"
    category = "mesh"
    summary = "Smooth a static triangle surface with uniform Taubin passes in float64."
    description = (
        "Preserves connectivity and field values. Optional boolean (N,3) point-field mask "
        "holds coordinate components after each lambda/mu pair. CPU is default; CUDA is "
        "opt-in. Smoothing changes geometry and does not recompute the simulation fields."
    )
    when_to_use = (
        "Reduce surface discretization noise before a separately validated geometry analysis."
    )
    determinism = "tolerance"
    tolerance = {"rtol": 1e-10, "atol": 1e-12}
    changes_geometry = True
    Params = TaubinParams

    def plan_samples(self, ref: DatasetRef, params: TaubinParams) -> list[str]:
        return [s.id for s in ref.manifest.samples if s.sample_kind == "surface"]

    def plan_sample(self, ed: SampleEditor, params: TaubinParams) -> list[Change]:
        if "points_t" in ed.sample.group:
            raise ValueError(f"{ed.id}: Taubin smoothing requires static geometry")
        if params.fixed_mask_field:
            field = params.fixed_mask_field
            if field not in ed.sample.field_names("point"):
                raise ValueError(f"{ed.id}: missing point field {field!r}")
            array = ed.sample.array(field, "point")
            if array.shape != ed.sample.group["points"].shape or array.dtype != np.bool_:
                raise ValueError(f"{ed.id}: fixed mask must be a static boolean (N,3) point field")
        return (
            []
            if params.iterations == 0
            else [
                Change(
                    target=f"samples/{ed.id}/points", action="modify", detail=params.model_dump()
                )
            ]
        )

    def apply_sample(self, ed: SampleEditor, params: TaubinParams, *, seed: int) -> SampleOutcome:
        self.plan_sample(ed, params)
        s = ed.sample
        fixed = (
            ed.field(params.fixed_mask_field, location="point") if params.fixed_mask_field else None
        )
        points, faces = np.asarray(s.group["points"][...]), s.cells
        output, metadata = taubin_smooth(
            points,
            faces,
            fixed=fixed,
            iterations=params.iterations,
            lam=params.lam,
            mu=params.mu,
            backend=params.backend,
        )
        triangles = output[faces]
        areas = (
            np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            )
            / 2
        )
        area = float(areas.sum())
        bounds = (
            np.stack((output.min(axis=0), output.max(axis=0)), axis=1).ravel().tolist()
            if len(output)
            else [0.0] * 6
        )
        ed.write_root_array("points", output, ["point", "space"])
        ed.write_root_array("cell_area", areas, ["cell"])
        ed.set_attrs(area=area, bounds=bounds)
        stable = {k: v for k, v in metadata.items() if k not in {"timing_s", "host_cache_hit"}}

        def update(meta: Any) -> None:
            meta.mesh.area = area
            meta.mesh.bounds = bounds
            meta.extra.setdefault("geometry", {})["taubin"] = stable

        ed.update_meta(update)
        warnings = [
            OpWarning(
                severity="warn",
                code="geometry-fields-unchanged",
                message="Geometry changed; field values were retained without interpolation.",
                target=ed.id,
            )
        ]
        if metadata["fallback_reason"]:
            warnings.append(
                OpWarning(
                    severity="warn",
                    code="cuda-unavailable",
                    message=metadata["fallback_reason"],
                    target=ed.id,
                )
            )
        return SampleOutcome(
            sample_id=ed.id,
            changes=[Change(target=f"samples/{ed.id}/points", action="modify", detail=stable)],
            stats=metadata,
            warnings=warnings,
        )
