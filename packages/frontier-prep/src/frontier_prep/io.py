# SPDX-License-Identifier: Apache-2.0
"""Sample-level editing primitives used by ops.

Every write creates new files (arrays are removed and re-created, JSON is written atomically);
nothing is modified in place, which is what keeps the content-addressed version store sound
(``frontier_data.versions``). A field rename is a directory rename: the chunk files keep their
inodes, so the next version shares them.

Derived LOD levels (``lod/<n>/...``) follow level 0: a rename is applied to every stored level
too, and every other change of a level-0 array sets :attr:`SampleEditor.lods_stale`, which the
engine (``BaseOp``) answers by rebuilding the stored levels from the new level 0.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from frontier_data.arrays import Sample, _step_chunks, auto_steps_per_chunk
from frontier_data.dataset import (
    CHUNK_ROWS,
    STORE_NAMES,
    FieldInfo,
    Manifest,
    SampleMeta,
    _compressors,
    write_json,
)

LOCATION_GROUP = {"point": "fields", "grid": "fields", "cell": "cell_fields", "static": "static"}
GROUP_LOCATION = {"fields": "point", "cell_fields": "cell", "static": "static"}
#: root arrays that define the geometry a derived LOD level is built from
GEOMETRY_ARRAYS = ("points", "faces", "cells", "cell_types", "cell_area", "points_t")


def read_manifest(root: Path) -> Manifest:
    return Manifest.model_validate(json.loads((Path(root) / "manifest.json").read_text()))


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temporary file and rename: versioned files are read-only hardlinks."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def save_manifest(root: Path, manifest: Manifest) -> None:
    write_json(Path(root) / "manifest.json", manifest.model_dump())


class SampleEditor:
    """Open one sample directory for reading and whole-array replacement."""

    def __init__(self, sample_dir: Path, sample_id: str | None = None) -> None:
        self.dir = Path(sample_dir)
        self.id = sample_id or self.dir.name
        self._sample: Sample | None = None
        self._meta: SampleMeta | None = None
        self.bytes_read = 0
        self.bytes_written = 0
        #: a level-0 array (a field at a point/cell location, or the geometry) was rewritten,
        #: added or deleted, so stored LOD levels no longer describe level 0. Renames and
        #: units-only metadata changes do not set it (a rename is applied to the levels).
        self.lods_stale = False
        #: the stale part includes the geometry (``points``, ``faces``, ``cells``, ...); a
        #: ``mesh.zarr`` level is a vertex map, so only this invalidates it
        self.geometry_stale = False

    # -- reading -----------------------------------------------------------------------------

    @property
    def sample(self) -> Sample:
        if self._sample is None:
            self._sample = Sample.open(self.dir, self.id)
        return self._sample

    def reopen(self) -> Sample:
        self._sample = None
        return self.sample

    @property
    def kind(self) -> str:
        return self.sample.kind

    @property
    def store_path(self) -> Path:
        return self.sample.store_path

    @property
    def meta(self) -> SampleMeta:
        if self._meta is None:
            self._meta = SampleMeta.model_validate(json.loads((self.dir / "meta.json").read_text()))
        return self._meta

    def field(self, name: str, t: Any = None, *, location: str | None = None) -> np.ndarray:
        a = self.sample.field(name, t, location=location)
        self.bytes_read += a.nbytes
        return a

    def array(self, name: str, location: str | None = None) -> Any:
        return self.sample.array(name, location)

    def dims(self, name: str, location: str | None = None) -> list[str]:
        return self.sample.dims(name, location)

    def location_of(self, name: str) -> str:
        """``point`` | ``cell`` | ``grid`` for a field name. When a name exists at both
        locations the cell array is the native one (point copies are derived)."""
        s = self.sample
        if name in s.field_names("cell"):
            return "cell"
        return "grid" if self.kind == "grid" else "point"

    def field_names(self, location: str | None = None) -> list[str]:
        return self.sample.field_names(location)

    def target(self, name: str, location: str | None = None) -> str:
        """Change target path ``samples/<id>/<fields|cell_fields>/<name>``."""
        loc = location or (self.location_of(name) if name in self.field_names() else "point")
        return f"samples/{self.id}/{LOCATION_GROUP.get(loc, 'fields')}/{name}"

    def iter_steps(self, name: str, *, location: str | None = None) -> Iterator[np.ndarray]:
        """One step at a time (time-dependent arrays), never a whole trajectory. Steps are
        read one chunk at a time so a chunk holding 16 steps is decompressed once."""
        loc = location if location == "cell" else None
        if not self.sample.is_time_dependent(name, loc):
            yield self.field(name, location=loc)
            return
        arr = self.array(name, loc)
        n = int(arr.shape[0])
        block = max(int(arr.chunks[0]), 1)
        for t0 in range(0, n, block):
            chunk = np.asarray(arr[t0 : min(t0 + block, n)])
            self.bytes_read += chunk.nbytes
            for k in range(chunk.shape[0]):
                yield chunk[k]

    def field_info(self, name: str) -> FieldInfo | None:
        for f in self.meta.fields:
            if f.name == name:
                return f
        return None

    # -- writing -----------------------------------------------------------------------------

    def mark_lods_stale(self, *, geometry: bool = False) -> None:
        """Tell the engine that level 0 changed (helpers that write arrays without going
        through ``write_field`` / ``write_root_array`` call this)."""
        self.lods_stale = True
        self.geometry_stale = self.geometry_stale or geometry

    def lod_levels_on_disk(self) -> list[int]:
        """Derived levels (> 0) that have a group under ``lod/``; the directory is the truth,
        ``meta.lod_levels`` and the root attribute follow it."""
        d = self.store_path / "lod"
        if not d.is_dir():
            return []
        return sorted(
            int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit() and p.name != "0"
        )

    def _group(self, location: str) -> Any:
        import zarr

        root: Any = zarr.open_group(str(self.store_path), mode="r+")
        gname = LOCATION_GROUP[location]
        return root[gname] if gname in root else root.create_group(gname)

    def array_path(self, name: str, location: str) -> Path:
        return self.store_path / LOCATION_GROUP[location] / name

    def write_field(
        self,
        name: str,
        data: np.ndarray,
        *,
        location: str,
        dims: Sequence[str] | None = None,
        units: str | None = None,
        chunks: tuple[int, ...] | None = None,
        description: str | None = None,
        meta_extra: dict[str, Any] | None = None,
    ) -> None:
        """Replace (or add) ``fields/<name>`` / ``cell_fields/<name>`` wholesale. ``data`` is
        ``(T?, N, C)`` for point/cell fields or ``(T?, X, Y[, Z], C)`` for grids; a missing
        component axis is added."""
        a = np.asarray(data)
        if a.dtype.kind == "f":
            a = a.astype(np.float32, copy=False)
        if dims is None:
            dims = self._infer_dims(name, a, location)
        dims = list(dims)
        if a.ndim == len(dims) - 1 and dims[-1] == "component":
            a = a[..., None]
        if a.ndim != len(dims):
            raise ValueError(f"{name}: data ndim {a.ndim} does not match dims {dims}")
        a = np.ascontiguousarray(a)
        if chunks is None:
            chunks = self._chunks(a, dims)
        group = self._group(location)
        *parents, leaf = name.split("/")
        for part in parents:
            group = group[part] if part in group else group.create_group(part)
        target = self.array_path(name, location)
        # rewriting an array with the values it already holds (an op that is run again)
        # leaves the stored LOD levels valid
        unchanged = location == "static" or self._holds(name, location, a, dims)
        if target.exists():
            shutil.rmtree(target)
        arr = group.create_array(
            leaf, shape=a.shape, dtype=a.dtype, chunks=chunks, compressors=_compressors()
        )
        if a.size:
            arr[...] = a
        arr.attrs["dims"] = dims
        self.bytes_written += a.nbytes
        if not unchanged:
            self.mark_lods_stale()
        self._record_field(name, a, location, dims, units, description, meta_extra)
        self.reopen()

    def _holds(self, name: str, location: str, a: np.ndarray, dims: list[str]) -> bool:
        """The stored array already has exactly these values, dtype and dims. Only asked when
        the sample stores LOD levels (it reads the stored array)."""
        if not self.array_path(name, location).exists() or not self.lod_levels_on_disk():
            return False
        try:
            old = self.sample.array(name, location)
            if tuple(old.shape) != a.shape or old.dtype != a.dtype:
                return False
            if self.sample.dims(name, location) != dims:
                return False
            return bool(np.array_equal(np.asarray(old[...]), a, equal_nan=a.dtype.kind == "f"))
        except (KeyError, ValueError, TypeError):
            return False

    def _infer_dims(self, name: str, a: np.ndarray, location: str) -> list[str]:
        try:
            existing = self.dims(name, None if location == "grid" else location)
            if len(existing) in (a.ndim, a.ndim + 1):
                return existing
        except KeyError:
            pass
        if self.kind == "grid":
            axes = self.sample.axes
            lead = ["time"] if a.ndim == len(axes) + 2 else []
            return lead + axes + ["component"]
        entity = "cell" if location == "cell" else "point"
        lead = ["time"] if a.ndim == 3 else []
        return lead + [entity, "component"]

    @staticmethod
    def _chunks(a: np.ndarray, dims: list[str]) -> tuple[int, ...]:
        if dims and dims[0] == "time":
            spc = auto_steps_per_chunk(a.shape[1:], a.dtype.itemsize)
            return _step_chunks(a.shape, a.dtype.itemsize, spc)[: a.ndim]
        if a.ndim > 2:
            return tuple(a.shape)
        return (min(CHUNK_ROWS, max(a.shape[0], 1)),) + tuple(a.shape[1:])

    def _record_field(
        self,
        name: str,
        a: np.ndarray,
        location: str,
        dims: list[str],
        units: str | None,
        description: str | None,
        extra: dict[str, Any] | None,
    ) -> None:
        info = self.field_info(name)
        temporal = bool(dims and dims[0] == "time")
        if info is None:
            info = FieldInfo(name=name, location=location, channels=int(a.shape[-1]))
            self.meta.fields.append(info)
        # one entry per name: the cell array is the native one, a point copy never demotes it
        # and never overwrites its dims (location and dims of an entry describe one array)
        point_copy = (
            info.location == "cell"
            and location == "point"
            and self.array_path(name, "cell").exists()
        )
        if not point_copy:
            info.location = location
            info.channels = int(a.shape[-1])
            info.dims = dims
            info.temporal = temporal
        if units is not None:
            info.units = units
        if description is not None:
            info.description = description
        if extra:
            self.meta.extra.setdefault("fields", {}).setdefault(name, {}).update(extra)
        self.save_meta()

    def rename_field(self, old: str, new: str, *, location: str | None = None) -> None:
        """Directory rename: no chunk is rewritten, the next version shares them. Without
        ``location`` every location holding ``old`` (a cell field and its point copy) is
        renamed."""
        locs = (
            [location]
            if location
            else [loc for loc in ("point", "cell") if old in self.field_names(loc)]
        )
        if not locs:
            raise KeyError(f"{self.id}: no field {old!r}")
        for loc in locs:
            src, dst = self.array_path(old, loc), self.array_path(new, loc)
            if dst.exists():
                raise FileExistsError(f"{self.id}: field {new!r} already exists at {loc}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
        self._rename_in_lods(old, new, locs)
        for info in self.meta.fields:
            if info.name == old:
                info.name = new
        if old in self.meta.field_roles:
            self.meta.field_roles[new] = self.meta.field_roles.pop(old)
        extra = self.meta.extra.get("fields", {})
        if old in extra:
            extra[new] = extra.pop(old)
        self.save_meta()
        self.reopen()

    def _rename_in_lods(self, old: str, new: str, locs: Sequence[str]) -> None:
        """Apply a level-0 rename to every stored LOD group (directory rename, bitwise). A
        level that does not hold ``old`` where level 0 did, or already holds ``new``, was not
        describing level 0 any more: it is marked stale and rebuilt by the engine."""
        if self.kind != "surface":
            return  # mesh levels are vertex maps (no field copies), grids have no levels
        for level in self.lod_levels_on_disk():
            base = self.store_path / "lod" / str(level)
            for loc in locs:
                src = base / LOCATION_GROUP[loc] / old
                dst = base / LOCATION_GROUP[loc] / new
                if not src.exists() or dst.exists():
                    self.mark_lods_stale()
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.rename(src, dst)

    def delete_field(self, name: str, *, location: str | None = None) -> None:
        loc = location or self.location_of(name)
        p = self.array_path(name, loc)
        if p.exists():
            shutil.rmtree(p)
            if loc != "static":
                self.mark_lods_stale()
        self.reopen()
        if name not in self.field_names():
            self.meta.fields = [f for f in self.meta.fields if f.name != name]
        self.save_meta()  # an entry whose array survives at the other location is reconciled
        self.reopen()

    def set_units(self, name: str, units: str, *, declared: str | None = None) -> None:
        info = self.field_info(name)
        if info is None:
            raise KeyError(name)
        info.units = units
        if declared is not None:
            self.meta.extra.setdefault("fields", {}).setdefault(name, {})["declared_units"] = (
                declared
            )
        self.save_meta()

    def write_root_array(self, name: str, data: np.ndarray, dims: Sequence[str]) -> None:
        """Replace a root array such as ``points``, ``faces``, ``cells``, ``times``."""
        import zarr

        root: Any = zarr.open_group(str(self.store_path), mode="r+")
        a = np.ascontiguousarray(np.asarray(data))
        p = self.store_path / name
        if p.exists():
            shutil.rmtree(p)
        chunks = (
            tuple(a.shape)
            if name == "times" or a.ndim == 1
            else (min(CHUNK_ROWS, max(a.shape[0], 1)),) + tuple(a.shape[1:])
        )
        arr = root.create_array(
            name, shape=a.shape, dtype=a.dtype, chunks=chunks, compressors=_compressors()
        )
        if a.size:
            arr[...] = a
        arr.attrs["dims"] = list(dims)
        self.bytes_written += a.nbytes
        if name in GEOMETRY_ARRAYS:
            self.mark_lods_stale(geometry=True)
        self.reopen()

    def set_attrs(self, **attrs: Any) -> None:
        import zarr

        root: Any = zarr.open_group(str(self.store_path), mode="r+")
        root.attrs.update(attrs)
        self.reopen()

    def update_meta(self, fn: Callable[[SampleMeta], None]) -> None:
        fn(self.meta)
        self.save_meta()

    def save_meta(self) -> None:
        self._prune_meta_fields()
        write_json(self.dir / "meta.json", self.meta.model_dump())

    def _prune_meta_fields(self) -> None:
        """Drop field entries whose array no longer exists at any location, and make the
        ``location`` / ``dims`` of the others describe one array: the cell array when the name
        exists at both locations (a point copy is derived), else the one that exists."""
        try:
            s = self.reopen()
            present = set(s.field_names()) | set(s.signal_names)
        except (FileNotFoundError, KeyError):
            return
        self.meta.fields = [f for f in self.meta.fields if f.name in present]
        if self.kind == "grid":
            return
        cell, point = set(s.field_names("cell")), set(s.field_names("point"))
        for f in self.meta.fields:
            if f.location not in ("point", "cell"):
                continue
            loc = "cell" if f.name in cell else "point" if f.name in point else None
            if loc is None:
                continue
            stale_dims = bool(f.dims) and (("cell" in f.dims) != (loc == "cell"))
            if f.location == loc and not stale_dims:
                continue
            dims = s.dims(f.name, loc)
            f.location = loc
            f.dims = dims
            f.temporal = dims[:1] == ["time"]
            f.channels = int(s.array(f.name, loc).shape[-1])

    def forces(self) -> dict[str, Any]:
        p = self.dir / "forces.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def save_forces(self, forces: dict[str, Any]) -> None:
        write_json(self.dir / "forces.json", forces)

    def store_name(self) -> str:
        return STORE_NAMES[self.kind]


def sync_manifest_fields(root: Path, manifest: Manifest | None = None) -> Manifest:
    """Rebuild ``manifest.fields`` from the samples' ``meta.json`` (names, units, locations,
    channels), keeping the first description seen. Deterministic: samples in manifest order."""
    m = manifest or read_manifest(root)
    seen: dict[str, FieldInfo] = {}
    for entry in m.samples:
        mp = Path(root) / entry.path / "meta.json"
        if not mp.exists():
            continue
        meta = SampleMeta.model_validate(json.loads(mp.read_text()))
        for f in meta.fields:
            if f.name not in seen:
                seen[f.name] = f.model_copy()
            elif seen[f.name].units in ("-", "") and f.units not in ("-", ""):
                seen[f.name].units = f.units
    for f in seen.values():
        # one location and matching dims per name (metadata only; the sample's meta.json is
        # reconciled against its arrays whenever an editor saves it)
        if f.location in ("point", "cell") and f.dims:
            other = "point" if f.location == "cell" else "cell"
            f.dims = [f.location if d == other else d for d in f.dims]
    m.fields = list(seen.values())
    return m
