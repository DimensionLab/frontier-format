# SPDX-License-Identifier: Apache-2.0
"""Validate a dataset directory against the ``frontier-dataset`` v1 / v1.1 contract
(``docs/spec/dataset-format.md`` section 1). ``python -m frontier_data validate <dataset>``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_data.arrays import array_names
from frontier_data.dataset import (
    DOMAINS,
    FORMAT_NAME,
    FORMAT_VERSION,
    SAMPLE_KINDS,
    STORE_NAMES,
    TEMPORAL_KINDS,
    Manifest,
    SampleMeta,
)

MAX_STEPS_PER_CHUNK = 16  # "one step (or a few) per chunk"
SPACE_DIMS = {"x", "y", "z"}


@dataclass
class Issue:
    level: str  # "error" | "warning"
    where: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "where": self.where, "message": self.message}


@dataclass
class ValidationReport:
    root: str
    issues: list[Issue] = field(default_factory=list)
    n_samples: int = 0
    n_arrays: int = 0
    version: str = "?"

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def err(self, where: str, message: str) -> None:
        self.issues.append(Issue("error", where, message))

    def warn(self, where: str, message: str) -> None:
        self.issues.append(Issue("warning", where, message))

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "ok": self.ok,
            "version": self.version,
            "n_samples": self.n_samples,
            "n_arrays": self.n_arrays,
            "errors": [i.to_dict() for i in self.errors],
            "warnings": [i.to_dict() for i in self.warnings],
        }


def _check_field(
    rep: ValidationReport,
    where: str,
    arr: Any,
    *,
    location: str,
    kind: str,
    n_time: int | None,
    sizes: dict[str, int],
    minor: int,
    grid_shape: tuple[int, ...] | None,
) -> None:
    rep.n_arrays += 1
    dims = arr.attrs.get("dims")
    shape = tuple(int(s) for s in arr.shape)
    if dims is None:
        if minor >= 1:
            rep.err(where, "missing `dims` attribute (required since v1.1)")
        if n_time is not None and len(shape) >= 1 and shape[0] == n_time and len(shape) == 3:
            rep.warn(where, "looks time-dependent but has no `dims`; readers assume v1 defaults")
        return
    if not isinstance(dims, list) or not all(isinstance(d, str) for d in dims):
        rep.err(where, f"`dims` must be a list of strings, got {dims!r}")
        return
    if len(dims) != len(shape):
        rep.err(where, f"`dims` {dims} does not match array rank {len(shape)} (shape {shape})")
        return
    if "time" in dims:
        if dims[0] != "time":
            rep.err(where, f"time must be the first axis, dims are {dims}")
            return
        if location == "static":
            rep.err(where, "static arrays must not have a time axis")
        if n_time is None:
            rep.err(where, "has a time axis but the sample has no `times` array")
        elif shape[0] != n_time:
            rep.err(where, f"time axis has {shape[0]} steps, `times` has {n_time}")
        chunk0 = int(arr.chunks[0])
        if chunk0 > MAX_STEPS_PER_CHUNK and shape[0] > MAX_STEPS_PER_CHUNK:
            rep.warn(
                where,
                f"{chunk0} steps per chunk; a viewer cannot fetch a single step cheaply "
                f"(use <= {MAX_STEPS_PER_CHUNK})",
            )
    for d, n in zip(dims, shape, strict=True):
        if d in sizes and n != sizes[d]:
            rep.err(where, f"axis {d!r} has length {n}, expected {sizes[d]}")
    if location in ("fields", "cell_fields"):
        if dims[-1] != "component":
            rep.err(where, f"last axis of a field must be `component`, dims are {dims}")
        if kind == "grid":
            space = [d for d in dims if d in SPACE_DIMS]
            if grid_shape is not None and len(space) != len(grid_shape):
                rep.err(where, f"grid field dims {dims} do not cover the {len(grid_shape)}-D grid")
        else:
            want = "cell" if location == "cell_fields" else "point"
            if want not in dims:
                rep.err(where, f"{location} array must have a {want!r} axis, dims are {dims}")
    if arr.dtype.kind == "f" and arr.dtype.itemsize != 4 and location != "coords":
        rep.warn(where, f"dtype {arr.dtype}; the reference writers store float32")


def _validate_store(
    rep: ValidationReport, sid: str, kind: str, path: Path, minor: int, deep: bool
) -> dict[str, Any]:
    import zarr

    where = f"samples/{sid}/{path.name}"
    info: dict[str, Any] = {"n_steps": 1, "temporal": "steady"}
    try:
        g: Any = zarr.open_group(str(path), mode="r")
    except Exception as e:  # noqa: BLE001 - report any store error
        rep.err(where, f"cannot open zarr group: {e}")
        return info
    n_time: int | None = None
    if "times" in g:
        t = np.asarray(g["times"][...])
        n_time = int(t.shape[0])
        info.update(n_steps=n_time, temporal="transient")
        if t.ndim != 1 or t.dtype != np.float64:
            rep.err(f"{where}/times", f"must be (T,) float64, got {t.shape} {t.dtype}")
        elif n_time > 1 and not np.all(np.diff(t) > 0):
            rep.err(f"{where}/times", "must be strictly increasing")
        if not np.all(np.isfinite(t)):
            rep.err(f"{where}/times", "contains non-finite values")
    sizes: dict[str, int] = {}
    grid_shape: tuple[int, ...] | None = None
    if kind in ("surface", "mesh", "point_cloud"):
        if "points" not in g:
            rep.err(where, "missing `points`")
            return info
        pts = g["points"]
        sizes["point"] = int(pts.shape[0])
        d = int(pts.shape[1]) if pts.ndim == 2 else -1
        if pts.ndim != 2 or (kind == "surface" and d != 3) or d not in (1, 2, 3):
            rep.err(f"{where}/points", f"must be (N, D), got {tuple(pts.shape)}")
        conn_name = {"surface": "faces", "mesh": "cells"}.get(kind)
        if conn_name:
            if conn_name not in g:
                rep.err(where, f"missing `{conn_name}`")
            else:
                conn = g[conn_name]
                sizes["cell"] = int(conn.shape[0])
                if conn.ndim != 2 or conn.dtype.kind != "i":
                    rep.err(f"{where}/{conn_name}", "must be a 2-D integer array")
                elif kind == "surface" and conn.shape[1] != 3:
                    rep.err(f"{where}/faces", f"must be (M, 3), got {tuple(conn.shape)}")
                elif deep and conn.size:
                    c = np.asarray(conn[...])
                    if c.max() >= sizes["point"] or c.min() < (0 if kind == "surface" else -1):
                        rep.err(f"{where}/{conn_name}", "indices outside [0, N) (pad with -1)")
        if kind == "mesh":
            if "cell_types" not in g:
                rep.err(where, "missing `cell_types`")
            elif tuple(g["cell_types"].shape) != (sizes.get("cell", -1),):
                rep.err(f"{where}/cell_types", "must be (K,)")
        if kind == "surface" and "cell_area" not in g:
            rep.warn(where, "missing `cell_area`")
        if "points_t" in g:
            p = g["points_t"]
            if n_time is None or tuple(p.shape) != (n_time, *pts.shape):
                rep.err(f"{where}/points_t", f"must be (T, N, D), got {tuple(p.shape)}")
    else:  # grid
        if "coords" not in g:
            rep.err(where, "missing `coords` group")
        else:
            axes = [a for a in ("x", "y", "z") if a in g["coords"]]
            grid_shape = tuple(int(g["coords"][a].shape[0]) for a in axes)
            sizes.update(dict(zip(axes, grid_shape, strict=True)))
            if not axes:
                rep.err(f"{where}/coords", "needs at least `x`")
        for attr in ("spacing", "origin", "boundary_conditions"):
            if attr not in g.attrs:
                (rep.err if minor >= 1 and attr != "boundary_conditions" else rep.warn)(
                    where, f"missing attribute `{attr}`"
                )
        if "fields" not in g or not array_names(g["fields"]):
            rep.err(where, "a grid sample needs at least one array under `fields/`")
    for loc in ("fields", "cell_fields", "static"):
        if loc not in g:
            continue
        for name in array_names(g[loc]):
            arr = g[loc][name]
            _check_field(
                rep,
                f"{where}/{loc}/{name}",
                arr,
                location=loc,
                kind=kind,
                n_time=n_time,
                sizes=sizes,
                minor=minor,
                grid_shape=grid_shape,
            )
    if "signals" in g:
        for name in array_names(g["signals"]):
            arr = g["signals"][name]
            rep.n_arrays += 1
            sdims = arr.attrs.get("dims")
            ok = sdims in (["time", "component"], ["component"])
            if not ok:
                rep.err(f"{where}/signals/{name}", f"dims must be [time, component], got {sdims}")
            elif sdims[0] == "time" and (n_time is None or int(arr.shape[0]) != n_time):
                rep.err(f"{where}/signals/{name}", "time axis does not match `times`")
    loc = g.attrs.get("location")
    if kind == "grid" and loc is not None and loc not in ("node", "cell"):
        rep.err(where, f"grid attribute `location` must be node or cell, got {loc!r}")
    sk = g.attrs.get("sample_kind")
    if sk is not None and sk != kind:
        rep.err(where, f"attrs.sample_kind={sk!r} but the store is a {kind} store")
    declared = g.attrs.get("temporal")
    if declared is not None and declared != info["temporal"]:
        rep.err(where, f"attrs.temporal={declared!r} but the store is {info['temporal']}")
    if n_time is not None:
        has_time_array = any(
            isinstance(g[loc][n].attrs.get("dims"), list)
            and g[loc][n].attrs["dims"][:1] == ["time"]
            for loc in ("fields", "cell_fields")
            if loc in g
            for n in array_names(g[loc])
        )
        if not has_time_array and "points_t" not in g:
            rep.warn(where, "has `times` but no time-dependent array")
    if kind == "surface":
        for lvl, why in lod_mismatches(g).items():
            rep.err(f"{where}/lod/{lvl}", why)
    return info


def _listed_levels(g: Any) -> list[int]:
    return [int(x) for x in (g.attrs.get("lod_levels") or []) if int(x) > 0]


def lod_mismatches(g: Any) -> dict[int, str]:
    """Stored surface LOD levels that visibly do not describe level 0, with the reason: a level
    listed in ``lod_levels`` that is missing, a level without geometry, field names that
    differ from level 0's at a location, or an array whose entity axis is not the level's own
    point / face count. Metadata only: no array is read. A level with the right names and
    stale *values* cannot be detected this way."""
    out: dict[int, str] = {}
    if "lod" not in g:
        return {int(x): "listed in lod_levels but not stored" for x in _listed_levels(g)}
    lod: Any = g["lod"]
    on_disk = sorted(int(k) for k in lod.group_keys() if str(k).isdigit())
    listed = _listed_levels(g)
    for lvl in sorted(set(listed) - set(on_disk)):
        out[lvl] = "listed in lod_levels but not stored"
    for lvl in on_disk:
        sub = lod[str(lvl)]
        if "points" not in sub or "faces" not in sub:
            out[lvl] = "no geometry (points / faces)"
            continue
        sizes = {"fields": int(sub["points"].shape[0]), "cell_fields": int(sub["faces"].shape[0])}
        for loc, n in sizes.items():
            base = sorted(array_names(g[loc])) if loc in g else []
            names = sorted(array_names(sub[loc])) if loc in sub else []
            if names != base:
                missing = sorted(set(base) - set(names))
                extra = sorted(set(names) - set(base))
                out[lvl] = (
                    f"{loc} differ from level 0 (missing {missing}, extra {extra}): the level "
                    "is out of date; rebuild it (prep.mesh.decimate)"
                )
                break
            bad = []
            for name in names:
                arr = sub[loc][name]
                dims = arr.attrs.get("dims")
                axis = 1 if isinstance(dims, list) and dims[:1] == ["time"] else 0
                if arr.ndim <= axis or int(arr.shape[axis]) != n:
                    bad.append(name)
            if bad:
                out[lvl] = f"{loc} {bad} do not have {n} rows"
                break
    return out


def validate_dataset(root: str | Path, *, deep: bool = True) -> ValidationReport:
    """Check ``root`` against the contract. ``deep`` also reads connectivity arrays to check
    index ranges. Returns a :class:`ValidationReport`; ``report.ok`` is False on any error."""
    root = Path(root)
    rep = ValidationReport(root=str(root))
    mpath = root / "manifest.json"
    if not mpath.exists():
        rep.err("manifest.json", "not found")
        return rep
    try:
        raw = json.loads(mpath.read_text())
        m = Manifest.model_validate(raw)
    except Exception as e:  # noqa: BLE001
        rep.err("manifest.json", f"cannot parse: {e}")
        return rep
    rep.version = f"{m.version}.{m.minor}"
    if m.format != FORMAT_NAME:
        rep.err("manifest.json", f"format must be {FORMAT_NAME!r}, got {m.format!r}")
    if m.version != FORMAT_VERSION:
        rep.err("manifest.json", f"unsupported major version {m.version}")
    if m.minor >= 1:
        for key in ("sample_kind", "temporal", "domain"):
            if key not in raw:
                rep.err("manifest.json", f"v1.1 manifest is missing `{key}`")
        for key in ("license", "citation", "hub_id"):
            if raw.get(key) is None:
                rep.warn("manifest.json", f"`{key}` is not set")
    if m.sample_kind not in SAMPLE_KINDS:
        rep.err("manifest.json", f"sample_kind {m.sample_kind!r} not in {SAMPLE_KINDS}")
    if m.temporal not in TEMPORAL_KINDS:
        rep.err("manifest.json", f"temporal {m.temporal!r} not in {TEMPORAL_KINDS}")
    if m.domain not in DOMAINS:
        rep.err("manifest.json", f"domain {m.domain!r} not in {DOMAINS}")
    if m.temporal == "transient" and m.time is None:
        rep.err("manifest.json", "transient dataset without `time`")
    ids = m.sample_ids()
    if len(set(ids)) != len(ids):
        rep.err("manifest.json", "duplicate sample ids")
    known = set(ids)
    seen: dict[str, str] = {}
    for split, sids in m.splits.items():
        for sid in sids:
            if sid not in known:
                rep.err("manifest.json", f"split {split!r} references unknown sample {sid!r}")
            if sid in seen and seen[sid] != split:
                rep.err("manifest.json", f"sample {sid!r} is in splits {seen[sid]!r} and {split!r}")
            seen[sid] = split
    any_transient = False
    max_steps = 0
    for entry in m.samples:
        rep.n_samples += 1
        sdir = root / entry.path
        where = f"samples/{entry.id}"
        if not sdir.is_dir():
            rep.err(where, f"directory {entry.path} not found")
            continue
        stores = [(k, n) for k, n in STORE_NAMES.items() if (sdir / n).exists()]
        if len(stores) != 1:
            rep.err(
                where,
                f"exactly one of {sorted(STORE_NAMES.values())} is required, found "
                f"{[n for _, n in stores]}",
            )
            continue
        kind, store = stores[0]
        if m.minor >= 1 and entry.sample_kind != kind:
            rep.err(where, f"manifest says sample_kind={entry.sample_kind!r}, store is {store}")
        info = _validate_store(rep, entry.id, kind, sdir / store, m.minor, deep)
        any_transient |= info["temporal"] == "transient"
        max_steps = max(max_steps, info["n_steps"])
        if m.minor >= 1 and entry.temporal != info["temporal"]:
            rep.err(where, f"manifest says {entry.temporal}, store is {info['temporal']}")
        if m.minor >= 1 and entry.n_steps != info["n_steps"]:
            rep.err(where, f"manifest n_steps={entry.n_steps}, store has {info['n_steps']}")
        mp = sdir / "meta.json"
        if not mp.exists():
            rep.err(where, "meta.json not found")
            continue
        try:
            meta = SampleMeta.model_validate(json.loads(mp.read_text()))
        except Exception as e:  # noqa: BLE001
            rep.err(f"{where}/meta.json", f"cannot parse: {e}")
            continue
        if info["temporal"] == "transient" and meta.time is None:
            (rep.err if m.minor >= 1 else rep.warn)(f"{where}/meta.json", "missing `time`")
        elif meta.time is not None and meta.time.n_steps != info["n_steps"]:
            rep.err(f"{where}/meta.json", "time.n_steps does not match the store")
    if any_transient and m.temporal != "transient":
        rep.err("manifest.json", "temporal is `steady` but some samples are transient")
    if m.time is not None and any_transient and m.time.n_steps != max_steps:
        rep.warn("manifest.json", f"time.n_steps={m.time.n_steps}, longest sample has {max_steps}")
    return rep
