# SPDX-License-Identifier: Apache-2.0
"""``frontier-dataset`` v1.1 array writers and the lazy :class:`Sample` reader.

One zarr v3 store per sample, named by kind (``docs/spec/dataset-format.md`` section 1):

==============  ===============  ==========================================================
``sample_kind`` store            arrays
==============  ===============  ==========================================================
``surface``     ``surface.zarr`` ``points (N,3)``, ``faces (M,3)``, ``cell_area (M,)``,
                                 ``fields/<name>``, ``cell_fields/<name>``, ``lod/<level>/..``
``mesh``        ``mesh.zarr``    ``points (N,D)``, ``cells (K,P)`` padded with -1,
                                 ``cell_types (K,)`` VTK ids, ``fields/``, ``cell_fields/``
``grid``        ``grid.zarr``    ``fields/<name> (T?,X,Y[,Z],C)``, ``coords/x|y|z``, attrs
                                 ``spacing``, ``origin``, ``boundary_conditions``
``point_cloud`` ``points.zarr``  ``points (N,D)``, ``fields/<name>``
==============  ===============  ==========================================================

Every writer takes steady or transient arrays. Passing ``times`` (``(T,)``) makes the sample
transient: every array in ``fields`` / ``cell_fields`` then has time as its first axis and is
chunked ``steps_per_chunk`` (default one) steps per chunk, ``times`` is stored as float64 at the
group root and each array carries a ``dims`` attribute such as ``["time","point","component"]``.
Inputs without a time axis (geometry masks, material maps, node types) go in ``static``.
A moving mesh passes ``points_t (T,N,D)`` next to the reference ``points``.
"""

from __future__ import annotations

import json
import math
import operator
import shutil
import time as _time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from arrayroute.layout import GridLayoutPolicy
from arrayroute.reader import ArrayReader, ReadStats
from arrayroute.session import LocalReadSession
from arrayroute.transfer import BlockArray, copy_blocks
from frontier_data.dataset import (
    CHUNK_ROWS,
    FORMAT_MINOR,
    FORMAT_NAME,
    FORMAT_VERSION,
    STORE_NAMES,
    FieldInfo,
    MeshStats,
    Provenance,
    SampleEntry,
    SampleMeta,
    TimeInfo,
    _compressors,
    _write_array,
    _write_mesh_group,
    fields_of,
    write_json,
    write_sample,
)
from frontier_data.mesh import SurfaceMesh

#: target uncompressed bytes of one chunk of a time-dependent array
STEP_CHUNK_BYTES = 64 << 20
GRID_AXES = ("x", "y", "z")
#: where grid values live: "node" (coords are node positions) or "cell" (coords are the cell
#: centres and ``shape`` counts cells; finite-volume and lattice solvers)
GRID_LOCATIONS = ("node", "cell")
VTK_LINE, VTK_TRIANGLE, VTK_QUAD, VTK_TETRA, VTK_HEXAHEDRON, VTK_WEDGE, VTK_PYRAMID = (
    3,
    5,
    9,
    10,
    12,
    13,
    14,
)

MetaLike = SampleMeta | Mapping[str, Any] | None
ArrayMap = Mapping[str, np.ndarray] | None
TimeIndex = int | slice | Sequence[int] | np.ndarray | None


# -- helpers --------------------------------------------------------------------------------


def time_info(times: np.ndarray, units: str = "s", rtol: float = 1e-4) -> TimeInfo:
    """Summarise a time axis: ``dt`` is the median step, ``uniform`` when all steps agree."""
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size < 2:
        return TimeInfo(n_steps=int(times.size), dt=None, units=units, uniform=True, t0=_t0(times))
    d = np.diff(times)
    dt = float(np.median(d))
    uniform = bool(np.all(np.abs(d - dt) <= rtol * max(abs(dt), 1e-300)))
    return TimeInfo(
        n_steps=int(times.size),
        dt=dt if uniform else None,
        units=units,
        uniform=uniform,
        t0=float(times[0]),
    )


def _t0(times: np.ndarray) -> float:
    return float(times[0]) if times.size else 0.0


def _as_times(times: Any) -> np.ndarray | None:
    if times is None:
        return None
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if t.size == 0:
        raise ValueError("times must not be empty; omit it for a steady sample")
    if t.size > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing")
    return t


def _entity_field(
    name: str, arr: np.ndarray, n_time: int | None, n_entities: int, entity: str
) -> tuple[np.ndarray, list[str]]:
    """Normalise a point/cell field to ``(T?, N, C)`` float32 and return it with its dims."""
    a = np.asarray(arr)
    if a.dtype.kind not in "fiub":
        raise ValueError(f"{entity} field {name!r} has non-numeric dtype {a.dtype}")
    a = a.astype(np.float32, copy=False)
    lead = 0 if n_time is None else 1
    if a.ndim == lead + 1:
        a = a[..., None]
    if a.ndim != lead + 2:
        want = "(T, N[, C])" if lead else "(N[, C])"
        raise ValueError(f"{entity} field {name!r} must be {want}, got shape {a.shape}")
    if lead and a.shape[0] != n_time:
        raise ValueError(f"{entity} field {name!r} has {a.shape[0]} steps but times has {n_time}")
    if a.shape[lead] != n_entities:
        raise ValueError(
            f"{entity} field {name!r} has {a.shape[lead]} {entity}s, expected {n_entities}"
        )
    dims = (["time"] if lead else []) + [entity, "component"]
    return a, dims


def _step_chunks(shape: tuple[int, ...], itemsize: int, steps_per_chunk: int) -> tuple[int, ...]:
    """Chunks for a time-first array: a few steps along time, the row axis split when one step
    would exceed :data:`STEP_CHUNK_BYTES`."""
    rest = shape[1:]
    step_bytes = int(np.prod(rest, dtype=np.int64)) * itemsize
    rows = rest[0] if rest else 1
    if step_bytes > STEP_CHUNK_BYTES and rows > 1:
        rows = max(1, int(rows * STEP_CHUNK_BYTES / step_bytes))
    return (max(1, min(steps_per_chunk, shape[0])), min(rows, max(rest[0], 1)) if rest else 1) + (
        tuple(rest[1:]) if rest else ()
    )


def _leaf(group: Any, name: str) -> tuple[Any, str]:
    """Resolve a possibly nested field name (``"pred/velocity"``) to (sub-group, leaf)."""
    *parents, leaf = name.split("/")
    for part in parents:
        group = group[part] if part in group else group.create_group(part)
    return group, leaf


def array_names(group: Any, prefix: str = "") -> list[str]:
    """Names of every array below ``group``, nested ones as ``"sub/name"``."""
    out = [prefix + str(k) for k in sorted(group.array_keys())]
    for k in sorted(group.group_keys()):
        out += array_names(group[k], f"{prefix}{k}/")
    return out


def auto_steps_per_chunk(
    step_shape: Sequence[int], itemsize: int = 4, target_bytes: int = 256 << 10, max_steps: int = 16
) -> int:
    """Steps per chunk for small per-step arrays (a 2000-node mesh step is ~16 kB): enough
    steps to reach ``target_bytes``, at most ``max_steps`` so a viewer still fetches only "a
    few" steps. Large steps keep one step per chunk. Stores with hundreds of thousands of tiny
    chunk files are slow on network file systems."""
    step_bytes = int(np.prod(step_shape, dtype=np.int64)) * itemsize
    return int(max(1, min(max_steps, target_bytes // max(step_bytes, 1))))


def _put(group: Any, name: str, data: np.ndarray, dims: list[str], steps_per_chunk: int) -> None:
    group, name = _leaf(group, name)
    if dims and dims[0] == "time":
        chunks = _step_chunks(data.shape, data.dtype.itemsize, steps_per_chunk)[: data.ndim]
        _write_array(group, name, data, dims=dims, chunks=chunks)
    elif data.ndim == 0:
        raise ValueError(f"array {name!r} must have at least one axis")
    else:
        rows = min(CHUNK_ROWS, max(data.shape[0], 1))
        if data.ndim > 2:  # static grid arrays: keep whole unless huge
            rows = data.shape[0]
        _write_array(group, name, data, dims=dims, chunks=(rows, *data.shape[1:]))


def _static_dims(
    name: str,
    a: np.ndarray | BlockArray,
    sizes: Mapping[str, int],
    grid_shape: tuple[int, ...] | None,
) -> list[str]:
    if grid_shape is not None:
        nd = len(grid_shape)
        if a.shape[:nd] == tuple(grid_shape) and a.ndim in (nd, nd + 1):
            return list(GRID_AXES[:nd]) + (["component"] if a.ndim == nd + 1 else [])
    for entity, n in sizes.items():
        if a.ndim >= 1 and a.shape[0] == n and a.ndim <= 2:
            return [entity] + (["component"] if a.ndim == 2 else [])
    return [f"dim_{i}" for i in range(a.ndim)]


def _write_static(
    root: Any,
    static: ArrayMap,
    sizes: Mapping[str, int],
    dims: Mapping[str, Sequence[str]] | None,
    grid_shape: tuple[int, ...] | None = None,
) -> list[str]:
    if not static:
        return []
    group = root.create_group("static")
    for name, arr in static.items():
        a = np.asarray(arr)
        if a.dtype.kind == "f":
            a = a.astype(np.float32, copy=False)
        elif a.dtype.kind == "b":
            a = a.astype(np.uint8)
        d = list(dims[name]) if dims and name in dims else _static_dims(name, a, sizes, grid_shape)
        if "time" in d:
            raise ValueError(f"static array {name!r} must not have a time axis")
        _put(group, name, a, d, 1)
    return list(static)


def _write_signals(
    root: Any,
    signals: ArrayMap,
    n_time: int | None,
    spc: int,
    units: Mapping[str, str] | None = None,
) -> list[FieldInfo]:
    """Global signals (force coefficients, probe values, ODE outputs): ``signals/<name>`` as
    ``(T, C)`` with dims ``["time", "component"]``, or ``(C,)`` for a steady sample. ``units``
    (by name) go into ``meta.fields[].units``."""
    infos: list[FieldInfo] = []
    if not signals:
        return infos
    group = root.create_group("signals")
    for name, arr in signals.items():
        a = np.asarray(arr, dtype=np.float32)
        if n_time is not None:
            if a.ndim == 1:
                a = a[:, None]
            if a.ndim != 2 or a.shape[0] != n_time:
                raise ValueError(f"signal {name!r} must be (T[, C]) with T={n_time}, got {a.shape}")
            dims = ["time", "component"]
            sub, leaf = _leaf(group, name)
            # signals are tiny: many steps per chunk, a whole series is one or a few reads
            _write_array(sub, leaf, a, dims=dims, chunks=(max(spc, min(n_time, 4096)), a.shape[1]))
        else:
            a = a.reshape(-1)
            dims = ["component"]
            sub, leaf = _leaf(group, name)
            _write_array(sub, leaf, a, dims=dims, chunks=a.shape)
        infos.append(
            FieldInfo(
                name=name,
                channels=int(a.shape[-1]),
                location="global",
                dims=dims,
                temporal=n_time is not None,
                units=(units or {}).get(name, "-"),
            )
        )
    return infos


def _root_attrs(root: Any, kind: str, times: np.ndarray | None, extra: Mapping[str, Any]) -> None:
    root.attrs.update(
        {
            "format": f"{FORMAT_NAME}/{kind}",
            "sample_kind": kind,
            "version": FORMAT_VERSION,
            "minor": FORMAT_MINOR,
            "temporal": "transient" if times is not None else "steady",
            "n_steps": int(times.size) if times is not None else 1,
            **{k: _jsonable(v) for k, v in extra.items()},
        }
    )
    if times is not None:
        arr = root.create_array(
            "times",
            shape=times.shape,
            dtype=np.float64,
            chunks=times.shape,
            chunk_key_encoding={"name": "default", "configuration": {"separator": "."}},
        )
        arr[...] = times
        arr.attrs["dims"] = ["time"]


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    return v


def _open_store(sdir: Path, kind: str) -> Any:
    import zarr

    sdir.mkdir(parents=True, exist_ok=True)
    for other in STORE_NAMES.values():  # exactly one store per sample
        if (sdir / other).exists():
            shutil.rmtree(sdir / other)
    return zarr.open_group(str(sdir / STORE_NAMES[kind]), mode="w", zarr_format=3)


def _bounds(points: np.ndarray) -> list[float]:
    out = [0.0] * 6
    if points.size:
        lo, hi = points.min(axis=0), points.max(axis=0)
        for i in range(min(points.shape[1], 3)):
            out[2 * i], out[2 * i + 1] = float(lo[i]), float(hi[i])
    return out


def _finish(
    root_dir: Path,
    sample_id: str,
    *,
    kind: str,
    times: np.ndarray | None,
    stats: MeshStats,
    field_infos: list[FieldInfo],
    meta: MetaLike,
    dataset: str,
    parameters: Mapping[str, float] | None,
    group: str | None,
    time_units: str,
    shape: list[int] | None,
    n_cells: int,
    t_start: float,
    forces: Mapping[str, Any] | None = None,
    lod_levels: list[int] | None = None,
) -> SampleEntry:
    """Write ``meta.json`` (and ``forces.json``) and build the manifest entry."""
    sdir = root_dir / "samples" / sample_id
    if isinstance(meta, SampleMeta):
        m = meta.model_copy(deep=True)
    else:
        base: dict[str, Any] = {"id": sample_id, "dataset": dataset, "mesh": stats}
        base.update(dict(meta or {}))
        if "provenance" not in base:
            base["provenance"] = Provenance(source=dataset or "unknown")
        m = SampleMeta.model_validate(base)
    m.id = sample_id
    m.dataset = m.dataset or dataset
    m.mesh = stats
    m.sample_kind = kind
    m.temporal = "transient" if times is not None else "steady"
    if times is not None:
        m.time = time_info(times, time_units)
    if parameters:
        m.parameters = {**m.parameters, **{k: float(v) for k, v in parameters.items()}}
    if group is not None:
        m.group = group
    m.fields = field_infos
    m.lod_levels = lod_levels or [0]
    m.extra.setdefault("timings", {})["write_s"] = _time.perf_counter() - t_start
    write_json(sdir / "meta.json", m.model_dump())
    if forces is not None:
        write_json(sdir / "forces.json", dict(forces))
    return SampleEntry(
        id=sample_id,
        path=f"samples/{sample_id}",
        n_points=stats.n_points,
        n_faces=stats.n_faces,
        lod_levels=m.lod_levels,
        geometry_params=dict(m.geometry_params),
        forces={k: float(v) for k, v in (forces or {}).items() if isinstance(v, int | float)},
        sample_kind=kind,
        temporal=m.temporal,
        n_steps=int(times.size) if times is not None else 1,
        n_cells=n_cells,
        shape=shape,
        parameters=dict(m.parameters),
        group=m.group,
    )


def _write_entity_fields(
    root: Any,
    group_name: str,
    fields: ArrayMap,
    n_time: int | None,
    n_entities: int,
    entity: str,
    steps_per_chunk: int,
    units: Mapping[str, str] | None,
) -> list[FieldInfo]:
    infos: list[FieldInfo] = []
    if not fields:
        return infos
    group = root[group_name] if group_name in root else root.create_group(group_name)
    for name, arr in fields.items():
        a, dims = _entity_field(name, arr, n_time, n_entities, entity)
        _put(group, name, a, dims, steps_per_chunk)
        infos.append(
            FieldInfo(
                name=name,
                channels=int(a.shape[-1]),
                location=entity,
                dims=dims,
                temporal=n_time is not None,
                units=(units or {}).get(name, "-"),
            )
        )
    return infos


def default_cell_types(cells: np.ndarray, point_dim: int) -> np.ndarray:
    """VTK cell type per row of a ``-1``-padded connectivity array."""
    n = (np.asarray(cells) >= 0).sum(axis=1)
    lut = {2: VTK_LINE, 3: VTK_TRIANGLE, 5: VTK_PYRAMID, 6: VTK_WEDGE, 8: VTK_HEXAHEDRON}
    lut[4] = VTK_QUAD if point_dim == 2 else VTK_TETRA
    out = np.zeros(len(n), dtype=np.uint8)
    for k, v in lut.items():
        out[n == k] = v
    return out


# -- writers --------------------------------------------------------------------------------


def write_mesh_sample(
    root: str | Path,
    sample_id: str,
    *,
    points: np.ndarray,
    cells: np.ndarray,
    cell_types: np.ndarray | None = None,
    fields: ArrayMap = None,
    cell_fields: ArrayMap = None,
    times: np.ndarray | Sequence[float] | None = None,
    static: ArrayMap = None,
    signals: ArrayMap = None,
    points_t: np.ndarray | None = None,
    meta: MetaLike = None,
    parameters: Mapping[str, float] | None = None,
    group: str | None = None,
    dataset: str = "",
    time_units: str = "s",
    field_units: Mapping[str, str] | None = None,
    dims: Mapping[str, Sequence[str]] | None = None,
    steps_per_chunk: int = 1,
    attrs: Mapping[str, Any] | None = None,
    lod_levels: Sequence[int] | None = (1, 2, 3),
) -> SampleEntry:
    """Write an unstructured 2D/3D volume mesh sample to ``samples/<id>/mesh.zarr``.

    ``points`` is ``(N, D)`` with D in {2, 3}; ``cells`` is ``(K, P)`` int connectivity padded
    with ``-1``; ``cell_types`` are VTK ids (inferred from the row width when omitted).
    ``fields`` are point data ``(N[, C])`` or, with ``times``, ``(T, N[, C])``; ``cell_fields``
    likewise over cells. ``static`` holds time-independent inputs, ``points_t (T, N, D)`` a
    moving mesh. ``lod_levels`` (default 1, 2, 3) are written as vertex-clustering maps
    under ``lod/<level>`` (:func:`frontier_data.lod.write_mesh_lods`) so viewers can draw the
    mesh coarser; levels whose target exceeds the drawable face count are skipped. Returns
    the manifest :class:`SampleEntry`.
    """
    t0 = _time.perf_counter()
    root = Path(root)
    tt = _as_times(times)
    n_time = None if tt is None else int(tt.size)
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"points must be (N, 2) or (N, 3), got {pts.shape}")
    conn = np.ascontiguousarray(np.asarray(cells, dtype=np.int32))
    if conn.ndim != 2:
        raise ValueError(f"cells must be (K, P) padded with -1, got {conn.shape}")
    if conn.size and (conn.max() >= len(pts) or conn.min() < -1):
        raise ValueError("cells reference points outside [0, N)")
    ctypes = (
        default_cell_types(conn, pts.shape[1])
        if cell_types is None
        else np.asarray(cell_types, dtype=np.uint8).reshape(-1)
    )
    if len(ctypes) != len(conn):
        raise ValueError("cell_types length does not match cells")

    store = _open_store(root / "samples" / sample_id, "mesh")
    _write_array(store, "points", pts, dims=["point", "space"])
    _write_array(store, "cells", conn, dims=["cell", "cell_node"])
    _write_array(store, "cell_types", ctypes, dims=["cell"])
    infos = _write_entity_fields(
        store, "fields", fields, n_time, len(pts), "point", steps_per_chunk, field_units
    )
    infos += _write_entity_fields(
        store, "cell_fields", cell_fields, n_time, len(conn), "cell", steps_per_chunk, field_units
    )
    _write_points_t(store, points_t, n_time, pts, steps_per_chunk)
    infos += _write_signals(store, signals, n_time, steps_per_chunk)
    static_names = _write_static(store, static, {"point": len(pts), "cell": len(conn)}, dims)
    present = [0]
    if lod_levels:
        from frontier_data.lod import build_mesh_lod_maps, drawable_triangles, write_mesh_lods

        tris = drawable_triangles(conn, ctypes, int(pts.shape[1]))
        present = write_mesh_lods(store, pts, build_mesh_lod_maps(pts, tris, lod_levels))
    _root_attrs(
        store,
        "mesh",
        tt,
        {
            "n_points": len(pts),
            "n_cells": len(conn),
            "point_dim": int(pts.shape[1]),
            "bounds": _bounds(pts),
            "static": static_names,
            "time_units": time_units,
            **(attrs or {}),
        },
    )
    stats = MeshStats(n_points=len(pts), n_faces=len(conn), bounds=_bounds(pts), area=0.0)
    return _finish(
        root,
        sample_id,
        kind="mesh",
        times=tt,
        stats=stats,
        field_infos=infos,
        meta=meta,
        dataset=dataset,
        parameters=parameters,
        group=group,
        time_units=time_units,
        shape=None,
        n_cells=len(conn),
        t_start=t0,
        lod_levels=present,
    )


def _write_points_t(
    store: Any, points_t: np.ndarray | None, n_time: int | None, pts: np.ndarray, spc: int
) -> None:
    if points_t is None:
        return
    if n_time is None:
        raise ValueError("points_t needs times")
    p = np.ascontiguousarray(np.asarray(points_t, dtype=np.float32))
    if p.shape != (n_time, *pts.shape):
        raise ValueError(f"points_t must be {(n_time, *pts.shape)}, got {p.shape}")
    _put(store, "points_t", p, ["time", "point", "space"], spc)


def write_points_sample(
    root: str | Path,
    sample_id: str,
    *,
    points: np.ndarray,
    fields: ArrayMap = None,
    times: np.ndarray | Sequence[float] | None = None,
    static: ArrayMap = None,
    signals: ArrayMap = None,
    points_t: np.ndarray | None = None,
    meta: MetaLike = None,
    parameters: Mapping[str, float] | None = None,
    group: str | None = None,
    dataset: str = "",
    time_units: str = "s",
    field_units: Mapping[str, str] | None = None,
    dims: Mapping[str, Sequence[str]] | None = None,
    steps_per_chunk: int = 1,
    attrs: Mapping[str, Any] | None = None,
) -> SampleEntry:
    """Write a point cloud sample to ``samples/<id>/points.zarr`` (``points (N, D)``,
    ``fields/<name>`` as ``(N[, C])`` or ``(T, N[, C])``; particles that move pass
    ``points_t``)."""
    t0 = _time.perf_counter()
    root = Path(root)
    tt = _as_times(times)
    n_time = None if tt is None else int(tt.size)
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    if pts.ndim != 2 or pts.shape[1] not in (1, 2, 3):
        raise ValueError(f"points must be (N, D), got {pts.shape}")
    store = _open_store(root / "samples" / sample_id, "point_cloud")
    _write_array(store, "points", pts, dims=["point", "space"])
    infos = _write_entity_fields(
        store, "fields", fields, n_time, len(pts), "point", steps_per_chunk, field_units
    )
    _write_points_t(store, points_t, n_time, pts, steps_per_chunk)
    infos += _write_signals(store, signals, n_time, steps_per_chunk)
    static_names = _write_static(store, static, {"point": len(pts)}, dims)
    _root_attrs(
        store,
        "point_cloud",
        tt,
        {
            "n_points": len(pts),
            "point_dim": int(pts.shape[1]),
            "bounds": _bounds(pts),
            "static": static_names,
            "time_units": time_units,
            **(attrs or {}),
        },
    )
    stats = MeshStats(n_points=len(pts), n_faces=0, bounds=_bounds(pts), area=0.0)
    return _finish(
        root,
        sample_id,
        kind="point_cloud",
        times=tt,
        stats=stats,
        field_infos=infos,
        meta=meta,
        dataset=dataset,
        parameters=parameters,
        group=group,
        time_units=time_units,
        shape=None,
        n_cells=0,
        t_start=t0,
    )



def _write_grid_array(
    group: Any,
    name: str,
    source: BlockArray,
    *,
    dims: list[str],
    spatial_axes: tuple[int, ...],
    time_axis: int | None,
    steps_per_chunk: int,
    policy: GridLayoutPolicy | None,
    buffer_bytes: int,
    dtype: Any = np.float32,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    target = np.dtype(dtype)
    budget = operator.index(buffer_bytes)
    allowance = budget * target.itemsize // (source.dtype.itemsize + target.itemsize)
    if allowance < target.itemsize:
        raise MemoryError("Grid write buffer cannot hold one source and destination element")
    policy = policy or GridLayoutPolicy()
    policy = replace(
        policy,
        chunk_bytes=min(policy.chunk_bytes, allowance),
        shard_bytes=min(policy.shard_bytes, allowance),
    )
    layout = policy.plan(
        source.shape,
        itemsize=target.itemsize,
        spatial_axes=spatial_axes,
        time_axis=time_axis,
        steps_per_chunk=steps_per_chunk,
    )
    sub, leaf = _leaf(group, name)
    arr = sub.create_array(
        leaf,
        shape=source.shape,
        dtype=target,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=_compressors(),
        chunk_key_encoding={"name": "default", "configuration": {"separator": "."}},
    )
    copy_blocks(
        source,
        lambda region, value: arr.__setitem__(region, value),
        block_shape=layout.shards or layout.chunks,
        dtype=target,
        buffer_bytes=budget,
        cancelled=cancelled,
    )
    arr.attrs["dims"] = dims


def write_grid_sample(
    root: str | Path,
    sample_id: str,
    *,
    fields: Mapping[str, np.ndarray | BlockArray],
    times: np.ndarray | Sequence[float] | None = None,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    coords: Mapping[str, np.ndarray] | None = None,
    boundary_conditions: Mapping[str, Any] | None = None,
    static: Mapping[str, np.ndarray | BlockArray] | None = None,
    signals: ArrayMap = None,
    static_dtypes: Mapping[str, Any] | None = None,
    location: str = "node",
    ndim: int | None = None,
    meta: MetaLike = None,
    parameters: Mapping[str, float] | None = None,
    group: str | None = None,
    dataset: str = "",
    time_units: str = "s",
    field_units: Mapping[str, str] | None = None,
    dims: Mapping[str, Sequence[str]] | None = None,
    steps_per_chunk: int = 1,
    storage_layout: GridLayoutPolicy | None = None,
    attrs: Mapping[str, Any] | None = None,
    write_buffer_bytes: int = 64 << 20,
    cancelled: Callable[[], bool] | None = None,
) -> SampleEntry:
    """Write a regular-grid sample to ``samples/<id>/grid.zarr``.

    ``fields[name]`` is ``(X, Y[, Z][, C])`` or, with ``times``, ``(T, X, Y[, Z][, C])``. The
    number of space axes comes from ``ndim``, ``spacing``, ``origin`` or ``coords``; when none
    is given the last axis is taken as the component axis. ``coords`` (1-D ``x``/``y``/``z``)
    default to ``origin + i * spacing``; non-uniform grids pass ``coords`` and get
    ``spacing = null``. ``boundary_conditions`` is free-form JSON, by convention
    ``{"x": "periodic", "y": ["wall", "open"]}``. ``location="cell"`` declares cell-centred
    values: ``coords`` are then the cell centres and the grid shape counts cells (root
    attribute ``location``). ``signals`` are global time series ``(T[, C])``.
    Grid fields use bounded spatial chunks grouped into larger Zarr shards. Pass a
    ``GridLayoutPolicy`` to tune the spatial edges and decoded-byte limits. The temporal
    ``steps_per_chunk`` is an upper bound; byte limits take precedence. Existing stores
    are not rewritten. Field values still follow the dataset's float32 writer contract.
    Fields and static grid arrays may be borrowed ``BlockArray`` sources. Writes visit one
    shard at a time; ``write_buffer_bytes`` bounds returned blocks plus dtype conversion.
    Source scratch, coordinates/signals, Zarr codec buffers and OS caches are additional.
    ``static_dtypes`` explicitly overrides stored static-array dtypes.
    Use a dataset publication transaction to keep a failed sample private.
    """
    t0 = _time.perf_counter()
    root = Path(root)
    if not fields:
        raise ValueError("a grid sample needs at least one field")
    tt = _as_times(times)
    lead = 0 if tt is None else 1
    for hint in (spacing, origin, coords):
        if ndim is None and hint is not None:
            ndim = len(hint)
    first = BlockArray.from_array(next(iter(fields.values())))
    if ndim is None:
        ndim = len(first.shape) - lead - 1
    if ndim not in (1, 2, 3):
        raise ValueError(f"grids have 1 to 3 space axes, got {ndim}")
    shape = tuple(int(s) for s in first.shape[lead : lead + ndim])
    axes = list(GRID_AXES[:ndim])

    store = _open_store(root / "samples" / sample_id, "grid")
    fgroup = store.create_group("fields")
    infos: list[FieldInfo] = []
    for name, arr in fields.items():
        a = BlockArray.from_array(arr)
        if a.dtype.kind not in "biuf":
            raise ValueError(f"grid field {name!r} must contain real numeric values")
        if len(a.shape) == lead + ndim:
            a = a.with_component_axis()
        if len(a.shape) != lead + ndim + 1 or tuple(a.shape[lead : lead + ndim]) != shape:
            raise ValueError(f"grid field {name!r} has shape {a.shape}; expected grid {shape}+(C,)")
        if lead and tt is not None and a.shape[0] != tt.size:
            raise ValueError(f"grid field {name!r} has {a.shape[0]} steps, times has {tt.size}")
        d = (["time"] if lead else []) + axes + ["component"]
        _write_grid_array(
            fgroup,
            name,
            a,
            dims=d,
            spatial_axes=tuple(range(lead, lead + ndim)),
            time_axis=0 if lead else None,
            steps_per_chunk=steps_per_chunk,
            policy=storage_layout,
            buffer_bytes=write_buffer_bytes,
            cancelled=cancelled,
        )
        infos.append(
            FieldInfo(
                name=name,
                channels=int(a.shape[-1]),
                location="grid",
                dims=d,
                temporal=bool(lead),
                units=(field_units or {}).get(name, "-"),
            )
        )

    org = [float(x) for x in (origin if origin is not None else [0.0] * ndim)]
    cgroup = store.create_group("coords")
    spc: list[float] | None
    if coords is not None:
        cs = [np.asarray(coords[ax], dtype=np.float64).reshape(-1) for ax in axes]
        org = [float(c[0]) for c in cs]
        steps = [np.diff(c) for c in cs]
        is_uniform = all(d.size == 0 or np.allclose(d, d[0], rtol=1e-4, atol=0) for d in steps)
        spc = [float(d.mean()) if d.size else 1.0 for d in steps] if is_uniform else None
        if spacing is not None:
            spc = [float(x) for x in spacing]
    else:
        spc = [float(x) for x in (spacing if spacing is not None else [1.0] * ndim)]
        cs = [org[i] + spc[i] * np.arange(shape[i], dtype=np.float64) for i in range(ndim)]
    for ax, c, n in zip(axes, cs, shape, strict=True):
        if c.shape != (n,):
            raise ValueError(f"coords[{ax!r}] has length {c.size}, grid axis has {n}")
        _write_array(cgroup, ax, c, dims=[ax], chunks=c.shape)
    infos += _write_signals(
        store, signals, None if tt is None else int(tt.size), steps_per_chunk, field_units
    )
    if location not in GRID_LOCATIONS:
        raise ValueError(f"grid location must be one of {GRID_LOCATIONS}, got {location!r}")
    static_names = []
    if static:
        sg = store.create_group("static")
        for name, value in static.items():
            source = BlockArray.from_array(value)
            sd = (
                list(dims[name]) if dims and name in dims else _static_dims(name, source, {}, shape)
            )
            if "time" in sd or len(sd) != len(source.shape):
                raise ValueError(f"Static grid array {name!r} has invalid dimensions")
            dtype = (
                np.float32
                if source.dtype.kind == "f"
                else (np.uint8 if source.dtype.kind == "b" else source.dtype)
            )
            _write_grid_array(
                sg,
                name,
                source,
                dims=sd,
                spatial_axes=tuple(range(min(ndim, source.ndim))),
                time_axis=None,
                steps_per_chunk=1,
                policy=storage_layout,
                buffer_bytes=write_buffer_bytes,
                dtype=(static_dtypes or {}).get(name, dtype),
                cancelled=cancelled,
            )
            static_names.append(name)
    _root_attrs(
        store,
        "grid",
        tt,
        {
            "shape": list(shape),
            "axes": axes,
            "location": location,
            "spacing": spc,
            "origin": org,
            "boundary_conditions": dict(boundary_conditions or {}),
            "static": static_names,
            "time_units": time_units,
            **(attrs or {}),
        },
    )
    bounds = [0.0] * 6
    for i, c in enumerate(cs):
        bounds[2 * i], bounds[2 * i + 1] = float(c.min()), float(c.max())
    n_pts = int(np.prod(shape))
    stats = MeshStats(n_points=n_pts, n_faces=0, bounds=bounds, area=0.0)
    if boundary_conditions and not isinstance(meta, SampleMeta):
        meta = {"boundary_conditions": _jsonable(boundary_conditions), **dict(meta or {})}
    return _finish(
        root,
        sample_id,
        kind="grid",
        times=tt,
        stats=stats,
        field_infos=infos,
        meta=meta,
        dataset=dataset,
        parameters=parameters,
        group=group,
        time_units=time_units,
        shape=list(shape),
        n_cells=0,
        t_start=t0,
    )


def write_surface_sample(
    root: str | Path,
    sample_id: str,
    mesh: SurfaceMesh | Mapping[int, SurfaceMesh],
    *,
    fields: ArrayMap = None,
    cell_fields: ArrayMap = None,
    times: np.ndarray | Sequence[float] | None = None,
    static: ArrayMap = None,
    signals: ArrayMap = None,
    points_t: np.ndarray | None = None,
    meta: MetaLike = None,
    forces: Mapping[str, Any] | None = None,
    parameters: Mapping[str, float] | None = None,
    group: str | None = None,
    dataset: str = "",
    time_units: str = "s",
    field_units: Mapping[str, str] | None = None,
    dims: Mapping[str, Sequence[str]] | None = None,
    steps_per_chunk: int = 1,
    stl_path: Path | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> SampleEntry:
    """Write a triangulated surface sample to ``samples/<id>/surface.zarr``.

    ``mesh`` is a :class:`SurfaceMesh` or ``{lod_level: SurfaceMesh}`` (level 0 required); its
    own ``point_fields`` / ``cell_fields`` are stored as time-independent fields. Steady samples
    are byte-compatible with v1. With ``times``, ``fields`` (``(T, N[, C])`` point data) and
    ``cell_fields`` (``(T, M[, C])``) are the transient arrays; extra LOD levels receive the
    point fields by nearest-vertex lookup so a viewer can animate a coarse level.
    """
    t0 = _time.perf_counter()
    root = Path(root)
    lods: dict[int, SurfaceMesh] = (
        {0: mesh} if isinstance(mesh, SurfaceMesh) else {int(k): v for k, v in mesh.items()}
    )
    if 0 not in lods:
        raise ValueError("LOD level 0 (full mesh) is required")
    full = lods[0]
    tt = _as_times(times)
    stats = MeshStats(
        n_points=full.n_points, n_faces=full.n_faces, bounds=list(full.bounds), area=full.area
    )
    if tt is None:
        if fields or cell_fields:
            full = SurfaceMesh(
                points=full.points,
                faces=full.faces,
                point_fields={**full.point_fields, **dict(fields or {})},
                cell_fields={**full.cell_fields, **dict(cell_fields or {})},
                cell_area=full.cell_area,
            )
            lods = {**lods, 0: full}
        if isinstance(meta, SampleMeta):
            m = meta.model_copy(deep=True)
        else:
            m = SampleMeta.model_validate(
                {
                    "id": sample_id,
                    "dataset": dataset,
                    "mesh": stats,
                    "provenance": Provenance(source=dataset or "unknown"),
                    **dict(meta or {}),
                }
            )
        if parameters:
            m.parameters = {**m.parameters, **{k: float(v) for k, v in parameters.items()}}
        m.group = group if group is not None else m.group
        if field_units:
            if not m.fields:
                m.fields = fields_of(full)
            for f in m.fields:
                if f.name in field_units:
                    f.units = field_units[f.name]
        entry = write_sample(root, sample_id, lods, m, dict(forces or {}), stl_path=stl_path)
        if static or attrs or signals:
            import zarr

            store = zarr.open_group(str(root / entry.path / "surface.zarr"), mode="r+")
            sizes = {"point": full.n_points, "cell": full.n_faces}
            _write_signals(store, signals, None, 1)
            store.attrs["static"] = _write_static(store, static, sizes, dims)
            store.attrs.update({k: _jsonable(v) for k, v in (attrs or {}).items()})
        entry.parameters = dict(m.parameters)
        entry.group = m.group
        return entry

    n_time = int(tt.size)
    # a transient array replaces a steady field of the same name carried by the mesh object
    lods = {k: _without_fields(v, fields, cell_fields) for k, v in lods.items()}
    full = lods[0]
    store = _open_store(root / "samples" / sample_id, "surface")
    _write_mesh_group(store, full)
    infos = [
        FieldInfo(name=n, channels=int(a.shape[1]), location="cell", dims=["cell", "component"])
        for n, a in full.cell_fields.items()
    ] + [
        FieldInfo(name=n, channels=int(a.shape[1]), location="point", dims=["point", "component"])
        for n, a in full.point_fields.items()
    ]
    infos += _write_entity_fields(
        store, "fields", fields, n_time, full.n_points, "point", steps_per_chunk, field_units
    )
    infos += _write_entity_fields(
        store,
        "cell_fields",
        cell_fields,
        n_time,
        full.n_faces,
        "cell",
        steps_per_chunk,
        field_units,
    )
    _write_points_t(store, points_t, n_time, full.points, steps_per_chunk)
    infos += _write_signals(store, signals, n_time, steps_per_chunk)
    lod_group = store.create_group("lod")
    for level in sorted(lods):
        if level == 0:
            continue
        g = lod_group.create_group(str(level))
        _write_mesh_group(g, lods[level])
        if fields:
            from scipy.spatial import KDTree

            _, idx = KDTree(full.points).query(lods[level].points)
            sub = {k: np.asarray(v)[:, idx] for k, v in fields.items()}
            _write_entity_fields(
                g, "fields", sub, n_time, lods[level].n_points, "point", steps_per_chunk, None
            )
    sizes = {"point": full.n_points, "cell": full.n_faces}
    static_names = _write_static(store, static, sizes, dims)
    _root_attrs(
        store,
        "surface",
        tt,
        {
            "lod_levels": sorted(lods),
            "static": static_names,
            "time_units": time_units,
            **(attrs or {}),
        },
    )
    sdir = root / "samples" / sample_id
    if stl_path is not None and Path(stl_path).exists():
        shutil.copyfile(stl_path, sdir / "geometry.stl")
    return _finish(
        root,
        sample_id,
        kind="surface",
        times=tt,
        stats=stats,
        field_infos=infos,
        meta=meta,
        dataset=dataset,
        parameters=parameters,
        group=group,
        time_units=time_units,
        shape=None,
        n_cells=full.n_faces,
        t_start=t0,
        forces=dict(forces or {}),
        lod_levels=sorted(lods),
    )


def _without_fields(mesh: SurfaceMesh, fields: ArrayMap, cell_fields: ArrayMap) -> SurfaceMesh:
    drop_p, drop_c = set(fields or {}), set(cell_fields or {})
    if not (drop_p & set(mesh.point_fields)) and not (drop_c & set(mesh.cell_fields)):
        return mesh
    return SurfaceMesh(
        points=mesh.points,
        faces=mesh.faces,
        point_fields={k: v for k, v in mesh.point_fields.items() if k not in drop_p},
        cell_fields={k: v for k, v in mesh.cell_fields.items() if k not in drop_c},
        cell_area=mesh.cell_area,
    )


class SampleAppender:
    """Write a transient sample one step at a time (solver co-processing, connectors that
    stream a results file): the arrays grow along the time axis, nothing is held in memory.

    ```python
    with DatasetWriter("ds", "run").append_sample("case", kind="mesh", points=p, cells=c) as a:
        for t, u in solver:                       # u: (N, 2)
            a.append(t, fields={"velocity": u}, signals={"cd": [cd]})
    ```

    ``kind`` and the geometry keywords are those of the matching ``write_*_sample`` function
    (without ``fields`` / ``times``). The first :meth:`append` fixes field names and shapes;
    every later call must pass the same names. :meth:`close` (or leaving the ``with`` block)
    writes ``times``, ``meta.json`` and returns the manifest :class:`SampleEntry`.
    """

    def __init__(self, root: str | Path, sample_id: str, kind: str = "mesh", **kwargs: Any) -> None:
        if kind not in WRITERS:
            raise ValueError(f"kind must be one of {sorted(WRITERS)}, got {kind!r}")
        for key in ("fields", "cell_fields", "times", "signals", "points_t"):
            if key in kwargs:
                raise ValueError(f"pass {key!r} to append(), not to the appender")
        self.root, self.sample_id, self.kind = Path(root), sample_id, kind
        self.kwargs = kwargs
        self.times: list[float] = []
        self.entry: SampleEntry | None = None
        self._store: Any = None
        self._arrays: dict[str, Any] = {}

    def __enter__(self) -> SampleAppender:
        return self

    def __exit__(self, exc_type: object, *exc: object) -> None:
        if exc_type is None:
            self.close()

    @property
    def n_steps(self) -> int:
        return len(self.times)

    def append(
        self,
        time: float,
        fields: ArrayMap = None,
        cell_fields: ArrayMap = None,
        signals: ArrayMap = None,
        points: np.ndarray | None = None,
    ) -> int:
        """Append one step (arrays without a time axis); returns its index."""
        if self.times and time <= self.times[-1]:
            raise ValueError(f"time {time} is not after the previous step {self.times[-1]}")
        groups = {
            "fields": dict(fields or {}),
            "cell_fields": dict(cell_fields or {}),
            "signals": {k: np.atleast_1d(np.asarray(v)) for k, v in (signals or {}).items()},
        }
        if self._store is None:
            import zarr

            first = {g: {k: np.asarray(v)[None] for k, v in d.items()} for g, d in groups.items()}
            kw = dict(self.kwargs)
            if first["cell_fields"]:
                kw["cell_fields"] = first["cell_fields"]
            if points is not None:
                kw["points_t"] = np.asarray(points)[None]
            self.entry = WRITERS[self.kind](
                self.root,
                self.sample_id,
                fields=first["fields"] or None,
                signals=first["signals"] or None,
                times=[time],
                **kw,
            )
            path = self.root / "samples" / self.sample_id / STORE_NAMES[self.kind]
            self._store = zarr.open_group(str(path), mode="r+")
            for g, d in groups.items():
                for name in d:
                    self._arrays[f"{g}/{name}"] = self._store[g][name]
            if points is not None:
                self._arrays["points_t"] = self._store["points_t"]
            self.times.append(float(time))
            return 0
        given = {f"{g}/{k}": np.asarray(v) for g, d in groups.items() for k, v in d.items()}
        if points is not None:
            given["points_t"] = np.asarray(points)
        if set(given) != set(self._arrays):
            raise ValueError(
                f"every step must carry the same arrays: expected {sorted(self._arrays)}, "
                f"got {sorted(given)}"
            )
        i = len(self.times)
        for key, value in given.items():
            arr = self._arrays[key]
            step = value.astype(arr.dtype, copy=False)
            if step.shape != tuple(arr.shape[1:]):
                if step.shape + (1,) == tuple(arr.shape[1:]):
                    step = step[..., None]
                else:
                    raise ValueError(f"{key}: step shape {value.shape} != {tuple(arr.shape[1:])}")
            arr.resize((i + 1, *arr.shape[1:]))
            arr[i] = step
        self.times.append(float(time))
        return i

    def close(self) -> SampleEntry:
        """Finalise ``times``, attributes and ``meta.json``; returns the manifest entry."""
        if self._store is None or self.entry is None:
            raise ValueError("no step was appended")
        times = np.asarray(self.times, dtype=np.float64)
        tarr = self._store["times"]
        tarr.resize(times.shape)
        tarr[...] = times
        self._store.attrs["n_steps"] = int(times.size)
        mpath = self.root / "samples" / self.sample_id / "meta.json"
        meta = SampleMeta.model_validate(json.loads(mpath.read_text()))
        meta.time = time_info(times, meta.time.units if meta.time else "s")
        write_json(mpath, meta.model_dump())
        self.entry.n_steps = int(times.size)
        self._store = None
        return self.entry


WRITERS = {
    "surface": write_surface_sample,
    "mesh": write_mesh_sample,
    "grid": write_grid_sample,
    "point_cloud": write_points_sample,
}


# -- reader ---------------------------------------------------------------------------------


def default_dims(location: str, ndim: int, transient: bool, kind: str) -> list[str]:
    """Dims of an array without a ``dims`` attribute (v1 stores, foreign writers)."""
    if kind == "grid" and location == "fields":
        lead = 1 if transient else 0
        nsp = ndim - lead - 1
        return (["time"] if lead else []) + list(GRID_AXES[:nsp]) + ["component"]
    entity = "cell" if location == "cell_fields" else "point"
    if ndim == 3:
        return ["time", entity, "component"]
    return [entity, "component"][:ndim]


class Sample:
    """Lazy view of one sample store. Arrays are only read when asked for, and
    ``field(name, t=...)`` fetches just the chunks of the requested steps.

    Numerical reads use a worker-owned ArrayRoute session (128 MiB by default).
    Read buffers and gathered outputs are admitted before payload reads; backend
    caches, codec scratch, dtype conversions and consumer-retained arrays are separate.
    Metadata must stay fixed while reading. Call refresh() after external writes.
    Use spawn with native engines; a Sample and its session are not thread-safe.
    """

    def __init__(
        self, sample_dir: Path, kind: str, group: Any, sample_id: str | None,
        *, session: LocalReadSession | None = None,
    ) -> None:
        self.dir = Path(sample_dir)
        self.kind = kind
        self.group: Any = group
        self.id = sample_id or self.dir.name
        self.read_session = session or LocalReadSession()
        self._owns_session = session is None
        self.read_stats = ReadStats()
        self._times: np.ndarray | None = None
        self._meta: SampleMeta | None = None
        self._store: Path | None = None
        # zarr re-reads metadata on every lookup: keep opened arrays and dims per sample
        self._arrays: dict[tuple[str, str | None, int], Any] = {}
        self._dims: dict[tuple[str, str | None, int], list[str]] = {}

    @classmethod
    def open(
        cls, sample_dir: str | Path, sample_id: str | None = None,
        *, session: LocalReadSession | None = None,
    ) -> Sample:
        import zarr

        sample_dir = Path(sample_dir)
        found = [(k, n) for k, n in STORE_NAMES.items() if (sample_dir / n).exists()]
        if len(found) != 1:
            names = [n for _, n in found] or "none"
            raise FileNotFoundError(
                f"{sample_dir}: expected exactly one of {sorted(STORE_NAMES.values())}, "
                f"found {names}"
            )
        kind, name = found[0]
        return cls(sample_dir, kind, zarr.open_group(str(sample_dir / name), mode="r"),
                   sample_id, session=session)

    @classmethod
    def open_store(
        cls, store: str | Path, sample_id: str | None = None,
        *, session: LocalReadSession | None = None,
    ) -> Sample:
        """Open a single zarr store whatever its file name (e.g. ``predictions/<sid>.zarr``);
        the kind comes from the root attribute ``format`` (``frontier-dataset/<kind>``)."""
        import zarr

        store = Path(store)
        group = zarr.open_group(str(store), mode="r")
        fmt = str(group.attrs.get("format", ""))
        kind = fmt.rsplit("/", 1)[-1] if "/" in fmt else ""
        if kind not in STORE_NAMES:
            by_name = {v: k for k, v in STORE_NAMES.items()}
            kind = by_name.get(store.name, "grid" if "coords" in group else "mesh")
        out = cls(store.parent, kind, group, sample_id or store.stem, session=session)
        out._store = store
        return out

    def __getstate__(self) -> dict:
        return {**vars(self), "_times": None, "_arrays": {}, "_dims": {},
                "read_stats": ReadStats()}

    def refresh(self, *, session: LocalReadSession | None = None) -> None:
        """Reopen after writes finish. Borrowed sessions require a fresh replacement.

        The caller owns shared contexts: after changing immutable inputs, create a
        new TensorStoreReadGroup too, rather than reusing that group's cached data.
        """
        import zarr

        if session is self.read_session or (session is None and not self._owns_session):
            raise ValueError("Refreshing a borrowed session requires a fresh read session/context")
        if session is not None:
            self.read_session = session
            self._owns_session = False
        else:
            self.read_session.clear()
        self.group = zarr.open_group(store=self.group.store, path=self.group.path, mode="r")
        self._arrays.clear()
        self._dims.clear()
        self._times = None
        self._meta = None
        self.read_stats = ReadStats()

    def _reader(self, array: Any):
        from zarr.storage import LocalStore

        if not isinstance(array.store, LocalStore):
            raise ValueError("Sample reads require a local or mounted store; use a remote adapter")
        path = str(array.path)
        if (not path or path.startswith("/") or "\\" in path or "\0" in path
                or any(part in {"", ".", ".."} for part in path.split("/"))):
            raise ValueError("Sample array path must stay inside its store")
        return self.read_session.open(array.store.root, path)

    def _read(self, array: Any, leading: TimeIndex = None) -> np.ndarray:
        reader = self._reader(array)
        selection = (slice(None),) * len(reader.backend.spec.shape)
        try:
            if leading is None:
                values = reader.read(selection, stats=self.read_stats)
            elif isinstance(leading, bool | np.bool_):
                raise TypeError("Time indices must be integers, not booleans")
            elif isinstance(leading, int | np.integer | slice):
                axis = int(leading) if isinstance(leading, int | np.integer) else leading
                values = reader.read((axis, *selection[1:]), stats=self.read_stats)
            else:
                indices = np.asarray(leading)
                if indices.ndim != 1:
                    raise ValueError("Time indices must be one-dimensional")
                if indices.size and indices.dtype.kind not in {"i", "u"}:
                    raise TypeError("Time indices must be integers")
                values = self.read_session.read_rows(
                    array.store.root, str(array.path), indices.tolist(), stats=self.read_stats,
                )
        except MemoryError as error:
            raise MemoryError(
                "Sample selection exceeds the ArrayRoute read allowance; select fewer steps, "
                "stream a region with FieldReader, or configure a larger LocalReadSession. "
                + str(error)
            ) from error
        return values if values.flags.writeable else values.copy()

    # -- time ------------------------------------------------------------------------------

    @property
    def store_path(self) -> Path:
        return self._store or self.dir / STORE_NAMES[self.kind]

    @property
    def attrs(self) -> dict[str, Any]:
        return dict(self.group.attrs)

    @property
    def times(self) -> np.ndarray | None:
        """``(T,)`` float64 time stamps, ``None`` for a steady sample."""
        if self._times is None and "times" in self.group:
            self._times = self._read(self.group["times"]).astype(np.float64, copy=False)
        return self._times

    @property
    def temporal(self) -> str:
        return "transient" if "times" in self.group else "steady"

    @property
    def is_transient(self) -> bool:
        return "times" in self.group

    @property
    def n_steps(self) -> int:
        return int(self.group["times"].shape[0]) if "times" in self.group else 1

    @property
    def dt(self) -> float | None:
        t = self.times
        return None if t is None else time_info(t).dt

    @property
    def meta(self) -> SampleMeta:
        if self._meta is None:
            self._meta = SampleMeta.model_validate(json.loads((self.dir / "meta.json").read_text()))
        return self._meta

    # -- arrays ----------------------------------------------------------------------------

    def _sub(self, lod: int) -> Any:
        if lod == 0:
            return self.group
        lod_group: Any = self.group["lod"]
        return lod_group[str(lod)]

    def field_names(self, location: str | None = None, lod: int = 0) -> list[str]:
        """Names under ``fields/`` (``location="point"`` or ``"grid"``), ``cell_fields/``
        (``"cell"``) or both (``None``, cell names last, no duplicates)."""
        g = self._sub(lod)
        out: list[str] = []
        if location in (None, "point", "grid") and "fields" in g:
            out += array_names(g["fields"])
        if location in (None, "cell") and "cell_fields" in g:
            out += [k for k in array_names(g["cell_fields"]) if k not in out]
        return out

    @property
    def static_names(self) -> list[str]:
        if "static" not in self.group:
            return []
        static: Any = self.group["static"]
        return sorted(str(k) for k in static.array_keys())

    def array(self, name: str, location: str | None = None, lod: int = 0) -> Any:
        """The zarr array of a field (no data read). ``name`` may be nested
        (``"pred/velocity"``) or a full path such as ``"cell_fields/p"`` / ``"static/mask"``."""
        key = (name, location, lod)
        if key not in self._arrays:
            self._arrays[key] = self._find_array(name, location, lod)
        return self._arrays[key]

    def _find_array(self, name: str, location: str | None, lod: int) -> Any:
        g = self._sub(lod)
        if name.split("/", 1)[0] in ("fields", "cell_fields", "static", "coords", "signals"):
            return g[name]
        order = {"point": ["fields"], "grid": ["fields"], "cell": ["cell_fields"]}.get(
            location or "", ["fields", "cell_fields"]
        )
        for loc in order:
            if loc in g and name in g[loc]:
                return g[loc][name]
        raise KeyError(f"field {name!r} not found in {self.store_path} ({order})")

    def native_location(self, name: str) -> str:
        """``point``, ``cell`` or ``grid``: where the field's data lives according to
        ``meta.fields[].location`` (a cell field may also have a derived point copy under
        ``fields/``). ``array(name)`` / ``dims(name)`` / ``field(name)`` without a location
        keep the format's rule (the point array first, then the cell array), which the v1
        readers and the server rely on; code that pairs an array with ``meta.fields[]`` must
        pass ``location=sample.native_location(name)``."""
        try:
            native = next((f.location for f in self.meta.fields if f.name == name), None)
        except (OSError, ValueError):
            native = None
        if native in ("point", "cell", "grid"):
            return str(native)
        g = self.group
        if "fields" in g and name in g["fields"]:
            return "grid" if self.kind == "grid" else "point"
        return "cell"

    def _location_of(self, name: str, location: str | None, lod: int) -> str:
        if name.split("/", 1)[0] in ("fields", "cell_fields", "static", "coords", "signals"):
            return name.split("/", 1)[0]
        g = self._sub(lod)
        if location == "cell" or ("fields" not in g or name not in g["fields"]):
            return "cell_fields"
        return "fields"

    def dims(self, name: str, location: str | None = None, lod: int = 0) -> list[str]:
        key = (name, location, lod)
        if key not in self._dims:
            arr = self.array(name, location, lod)
            d = arr.attrs.get("dims")
            if isinstance(d, list):
                self._dims[key] = [str(x) for x in d]
            else:
                loc = self._location_of(name, location, lod)
                self._dims[key] = default_dims(loc, arr.ndim, self.is_transient, self.kind)
        return list(self._dims[key])

    def is_time_dependent(self, name: str, location: str | None = None) -> bool:
        return self.dims(name, location)[:1] == ["time"]

    def field(
        self, name: str, t: TimeIndex = None, *, location: str | None = None, lod: int = 0
    ) -> np.ndarray:
        """Read a field. ``t`` selects steps of a time-dependent array: an ``int`` returns one
        step without the time axis, a ``slice`` / index list keeps it, ``None`` reads every
        step. ``t`` is ignored for time-independent arrays."""
        arr = self.array(name, location, lod)
        return self._read(arr, t if self.dims(name, location, lod)[:1] == ["time"] else None)

    @property
    def signal_names(self) -> list[str]:
        """Global signals stored under ``signals/`` (force coefficients, probes, ODE outputs)."""
        return array_names(self.group["signals"]) if "signals" in self.group else []

    def signal(self, name: str, t: TimeIndex = None) -> np.ndarray:
        """A global signal: ``(T, C)`` (``t`` selects steps as in :meth:`field`) or ``(C,)``."""
        signals: Any = self.group["signals"]
        arr = signals[name]
        return self._read(arr, t if arr.ndim >= 2 else None)

    @property
    def location(self) -> str:
        """Grid samples: ``"node"`` or ``"cell"`` (cell-centred values, coords = centres)."""
        return str(self.attrs.get("location", "node"))

    def static(self, name: str) -> np.ndarray:
        static: Any = self.group["static"]
        return self._read(static[name])

    def points(self, t: int | None = None, lod: int = 0) -> np.ndarray:
        """Point coordinates ``(N, D)``; with ``t`` and a moving mesh, the positions at step
        ``t``."""
        if self.kind == "grid":
            axes = self.axes
            readers = [self._reader(self.group["coords"][axis]) for axis in axes]
            shape = tuple(reader.backend.spec.shape[0] for reader in readers)
            output_bytes = math.prod(shape) * len(axes) * np.dtype(np.float32).itemsize
            read_bytes = sum(reader.plan((slice(None),)).buffer_bytes_upper_bound
                             for reader in readers)
            if output_bytes + read_bytes > self.read_session.buffer_bytes:
                raise MemoryError("Grid points exceed the ArrayRoute allowance; select a region")
            pool = self.read_session.read_budget
            reservation = pool.reserve(output_bytes + read_bytes) if pool else nullcontext()
            with reservation:
                result = np.empty((*shape, len(axes)), dtype=np.float32)
                for index, reader in enumerate(readers):
                    # The aggregate reservation includes coordinate decoding; do not nest it.
                    worker = ArrayReader(reader.backend, buffer_bytes=reader.buffer_bytes)
                    values = worker.read((slice(None),), stats=self.read_stats)
                    broadcast = [1] * len(axes)
                    broadcast[index] = shape[index]
                    result[..., index] = values.reshape(broadcast)
                    del values
                return result.reshape(-1, len(axes))
        g = self._sub(lod)
        if t is not None and "points_t" in g:
            return self._read(g["points_t"], t)
        return self._read(g["points"])

    def surface(self, lod: int = 0, *, fields: bool = True) -> SurfaceMesh:
        """Materialize a prepared surface with the existing steady-field contract.

        Each array read is bounded; the returned mesh retains all selected arrays.
        Transient fields require an explicit time selection via field(), and are
        omitted here as in the legacy surface helper.
        """
        if self.kind != "surface":
            raise ValueError("SurfaceMesh materialization requires a surface sample")
        group = self._sub(lod)
        point_fields: dict[str, np.ndarray] = {}
        cell_fields: dict[str, np.ndarray] = {}
        if fields:
            for key, target in (("fields", point_fields), ("cell_fields", cell_fields)):
                if key in group:
                    for name in sorted(group[key].array_keys()):
                        array = group[key][name]
                        if array.ndim <= 2:
                            target[str(name)] = self._read(array)
        return SurfaceMesh(
            points=np.asarray(self._read(group["points"]), dtype=np.float32),
            faces=np.asarray(self._read(group["faces"]), dtype=np.int32),
            point_fields=point_fields,
            cell_fields=cell_fields,
            cell_area=(np.asarray(self._read(group["cell_area"]), dtype=np.float32)
                       if "cell_area" in group else None),
        )

    @property
    def has_moving_points(self) -> bool:
        return "points_t" in self.group

    @property
    def cells(self) -> np.ndarray:
        """``(K, P)`` connectivity padded with -1 (``faces`` for a surface)."""
        return self._read(self.group["faces" if self.kind == "surface" else "cells"])

    @property
    def cell_types(self) -> np.ndarray:
        if "cell_types" in self.group:
            return self._read(self.group["cell_types"])
        key = "faces" if self.kind == "surface" else "cells"
        count = int(self.group[key].shape[0])
        if count > self.read_session.buffer_bytes:
            raise MemoryError("Cell types exceed the ArrayRoute allowance")
        pool = self.read_session.read_budget
        with pool.reserve(count) if pool else nullcontext():
            return np.full(count, VTK_TRIANGLE, dtype=np.uint8)

    def triangles(self) -> np.ndarray:
        """Triangulated connectivity ``(M, 3)`` for plotting (quads split in two; cells with
        more than four nodes are skipped)."""
        c = self.cells
        n = (c >= 0).sum(axis=1)
        tri = [c[n == 3][:, :3]]
        q = c[n == 4]
        if len(q) and (self.attrs.get("point_dim", 3) == 2 or self.kind == "surface"):
            tri += [q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
        return np.concatenate(tri) if tri else np.zeros((0, 3), dtype=np.int32)

    # -- grid ------------------------------------------------------------------------------

    @property
    def axes(self) -> list[str]:
        axes = self.attrs.get("axes")
        if isinstance(axes, list):
            return [str(a) for a in axes]
        coords: Any = self.group["coords"] if "coords" in self.group else None
        return [a for a in GRID_AXES if coords is not None and a in coords]

    @property
    def coords(self) -> dict[str, np.ndarray]:
        coords: Any = self.group["coords"]
        return {a: self._read(coords[a]) for a in self.axes}

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial shape of a grid sample."""
        shape = self.attrs.get("shape")
        if isinstance(shape, list):
            return tuple(int(s) for s in shape)
        return tuple(len(c) for c in self.coords.values())

    @property
    def spacing(self) -> list[float] | None:
        s = self.attrs.get("spacing")
        return [float(x) for x in s] if isinstance(s, list) else None

    @property
    def origin(self) -> list[float] | None:
        s = self.attrs.get("origin")
        return [float(x) for x in s] if isinstance(s, list) else None

    @property
    def boundary_conditions(self) -> dict[str, Any]:
        bc = self.attrs.get("boundary_conditions")
        return dict(bc) if isinstance(bc, dict) else {}

    def __repr__(self) -> str:
        return (
            f"Sample({self.id!r}, kind={self.kind}, temporal={self.temporal}, "
            f"n_steps={self.n_steps}, fields={self.field_names()})"
        )
