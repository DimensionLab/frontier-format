# SPDX-License-Identifier: Apache-2.0
"""Reader/writer for the ``frontier-dataset`` v1 / v1.1 directory layout.

v1.1 (``docs/spec/dataset-format.md``) adds ``mesh.zarr``, ``grid.zarr`` and ``points.zarr``
samples, transient arrays (time first, ``times``, ``dims`` attrs, one step per chunk) and
``static/`` inputs. The array writers and the lazy :class:`~frontier_data.arrays.Sample` reader
live in :mod:`frontier_data.arrays` and are re-exported here.

```
<dataset>/
  manifest.json
  samples/<id>/meta.json        # v_CFD vector + mesh stats + provenance
  samples/<id>/surface.zarr     # zarr v3 group (see ``write_surface``)
  samples/<id>/forces.json
  samples/<id>/geometry.stl     # optional copy of the raw STL
  quality/report.json, quality/report.html
  pipeline.json                 # the pipeline that produced the dataset
```
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

from frontier_data.mesh import SurfaceMesh

if TYPE_CHECKING:
    from frontier_data.arrays import Sample

FORMAT_NAME = "frontier-dataset"
FORMAT_VERSION = 1
FORMAT_MINOR = 1
SAMPLE_KINDS = ("surface", "mesh", "grid", "point_cloud")
TEMPORAL_KINDS = ("steady", "transient")
DOMAINS = ("cfd", "structural", "thermal", "multiphysics", "materials", "geophysical", "other")
#: zarr store name per sample kind
STORE_NAMES = {
    "surface": "surface.zarr",
    "mesh": "mesh.zarr",
    "grid": "grid.zarr",
    "point_cloud": "points.zarr",
}
CHUNK_ROWS = 1 << 18  # 262144 rows -> ~3 MB chunks for (N,3) float32


class FieldInfo(BaseModel):
    name: str
    units: str = "-"
    description: str = ""
    channels: int = 1
    # where the native data lives: "point" | "cell" | "grid" | "global" (signals/)
    location: str = "point"
    dims: list[str] | None = None  # v1.1: e.g. ["time", "point", "component"]
    temporal: bool = False  # v1.1: True when the array has a leading time axis


class TimeInfo(BaseModel):
    """Time axis summary (``manifest.time`` and ``meta.time``)."""

    n_steps: int
    dt: float | None = None  # None when the steps are not uniform
    units: str = "s"
    uniform: bool = True
    t0: float = 0.0
    averaging_window: float | None = None  # for time-averaged (steady) samples


class MeshStats(BaseModel):
    n_points: int
    n_faces: int
    bounds: list[float]  # xmin, xmax, ymin, ymax, zmin, zmax
    area: float
    native_n_cells: int | None = None
    native_area_sum: float | None = None  # from boundary_cell_area_<i>.npy when available
    stl_n_faces: int | None = None
    stl_area: float | None = None


class RawFileRecord(BaseModel):
    role: str
    repo_path: str
    url: str
    sha256: str
    size: int
    revision: str | None = None


class Provenance(BaseModel):
    source: str  # e.g. "huggingface:neashton/ahmedml"
    license: str | None = None
    citation: str | None = None
    files: list[RawFileRecord] = Field(default_factory=list)
    ingested_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    ingest_tool: str = "frontier-data"


class SampleMeta(BaseModel):
    """The ``v_CFD`` sample vector: everything one needs to interpret a sample."""

    id: str
    dataset: str
    solver: str = "unknown"
    solver_version: str | None = None
    application: str | None = None
    turbulence_model: str | None = None
    turbulence_family: str | None = None
    physics: str | None = None
    fidelity: str | None = None
    fidelity_tier: str | None = None
    mesh: MeshStats
    geometry_params: dict[str, float] = Field(default_factory=dict)
    geometry_param_units: str | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    boundary_conditions: dict[str, Any] = Field(default_factory=dict)
    forces: dict[str, Any] = Field(default_factory=dict)
    fields: list[FieldInfo] = Field(default_factory=list)
    field_roles: dict[str, str] = Field(default_factory=dict)
    lod_levels: list[int] = Field(default_factory=list)
    lod_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=lambda: Provenance(source="unknown"))
    # v1.1
    sample_kind: str = "surface"
    temporal: str = "steady"
    time: TimeInfo | None = None
    parameters: dict[str, float] = Field(default_factory=dict)  # Re, Ma, AoA, material constants
    group: str | None = None  # trajectory id when one trajectory was cut into several samples
    extra: dict[str, Any] = Field(default_factory=dict)


class SampleEntry(BaseModel):
    id: str
    path: str
    n_points: int = 0
    n_faces: int = 0
    lod_levels: list[int] = Field(default_factory=lambda: [0])
    geometry_params: dict[str, float] = Field(default_factory=dict)
    forces: dict[str, float] = Field(default_factory=dict)
    # v1.1
    sample_kind: str = "surface"
    temporal: str = "steady"
    n_steps: int = 1
    n_cells: int = 0
    shape: list[int] | None = None  # grid samples: spatial shape
    parameters: dict[str, float] = Field(default_factory=dict)
    group: str | None = None


class Manifest(BaseModel):
    format: str = FORMAT_NAME
    version: int = FORMAT_VERSION
    minor: int = 0  # 0 = v1 (steady surfaces only); writers in this package emit 1
    sample_kind: str = "surface"
    temporal: str = "steady"
    time: TimeInfo | None = None
    domain: str = "cfd"
    license: str | None = None
    citation: str | None = None
    hub_id: str | None = None
    parameters: list[str] = Field(default_factory=list)
    name: str
    source: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    samples: list[SampleEntry] = Field(default_factory=list)
    splits: dict[str, list[str]] = Field(default_factory=dict)
    split_strategy: dict[str, Any] | None = None
    split_families: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fields: list[FieldInfo] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    geometry_params: list[str] = Field(default_factory=list)
    pipeline: str | None = None
    quality: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    # what one case cost to simulate, as the source states it (hub entry ``compute``):
    # {wall_clock_hours_per_case?, cores?, nodes?, node_hours_per_case?, hardware?, source}
    simulation: dict[str, Any] | None = None

    def sample_ids(self) -> list[str]:
        return [s.id for s in self.samples]


# -- zarr ---------------------------------------------------------------------------------


def _compressors() -> list[Any]:
    from zarr.codecs import BloscCodec

    return [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")]


def _write_array(
    group: Any,
    name: str,
    data: np.ndarray,
    *,
    dims: list[str] | None = None,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
) -> Any:
    data = np.ascontiguousarray(data)
    if chunks is None:
        chunks = (min(CHUNK_ROWS, max(data.shape[0], 1)),) + tuple(data.shape[1:])
    chunks = tuple(max(int(c), 1) for c in chunks)
    arr = group.create_array(
        name,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        shards=shards,
        compressors=_compressors(),
        chunk_key_encoding={"name": "default", "configuration": {"separator": "."}},
    )
    if data.size:
        arr[...] = data
    if dims is not None:
        arr.attrs["dims"] = list(dims)
    return arr


def _write_mesh_group(group: Any, mesh: SurfaceMesh) -> None:
    _write_array(group, "points", mesh.points.astype(np.float32))
    _write_array(group, "faces", mesh.faces.astype(np.int32))
    _write_array(group, "cell_area", mesh.areas().astype(np.float32))
    fields = group.create_group("fields")
    for name, arr in mesh.point_fields.items():
        _write_array(fields, name, arr.astype(np.float32), dims=["point", "component"])
    if mesh.cell_fields:
        cfields = group.create_group("cell_fields")
        for name, arr in mesh.cell_fields.items():
            _write_array(cfields, name, arr.astype(np.float32), dims=["cell", "component"])
    group.attrs.update(
        {
            "n_points": mesh.n_points,
            "n_faces": mesh.n_faces,
            "area": mesh.area,
            "bounds": list(mesh.bounds),
        }
    )


def write_surface(path: Path, lods: dict[int, SurfaceMesh]) -> None:
    """Write ``surface.zarr`` (zarr v3, blosc/zstd). Level 0 lives at the root, others under
    ``lod/<level>``."""
    import zarr

    if 0 not in lods:
        raise ValueError("LOD level 0 (full mesh) is required")
    if path.exists():
        shutil.rmtree(path)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    _write_mesh_group(root, lods[0])
    lod_group = root.create_group("lod")
    for level in sorted(lods):
        if level == 0:
            continue
        _write_mesh_group(lod_group.create_group(str(level)), lods[level])
    root.attrs["lod_levels"] = sorted(lods)
    root.attrs["format"] = f"{FORMAT_NAME}/surface"
    root.attrs["sample_kind"] = "surface"
    root.attrs["version"] = FORMAT_VERSION
    root.attrs["minor"] = FORMAT_MINOR
    root.attrs["temporal"] = "steady"


def _read_mesh_group(group: Any, *, fields: bool = True) -> SurfaceMesh:
    points = np.asarray(group["points"][...], dtype=np.float32)
    faces = np.asarray(group["faces"][...], dtype=np.int32)
    cell_area = (
        np.asarray(group["cell_area"][...], dtype=np.float32) if "cell_area" in group else None
    )
    pf: dict[str, np.ndarray] = {}
    cf: dict[str, np.ndarray] = {}
    if fields:
        if "fields" in group:
            fg = group["fields"]
            pf = {str(k): np.asarray(fg[k][...]) for k in fg.array_keys() if fg[k].ndim <= 2}
        if "cell_fields" in group:
            cg = group["cell_fields"]
            cf = {str(k): np.asarray(cg[k][...]) for k in cg.array_keys() if cg[k].ndim <= 2}
    return SurfaceMesh(
        points=points, faces=faces, point_fields=pf, cell_fields=cf, cell_area=cell_area
    )


def read_surface(path: Path, lod: int = 0, *, fields: bool = True) -> SurfaceMesh:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    lod_group: Any = root["lod"] if lod != 0 else None
    group = root if lod == 0 else lod_group[str(lod)]
    return _read_mesh_group(group, fields=fields)


def surface_lod_levels(path: Path) -> list[int]:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    levels: Any = root.attrs.get("lod_levels")
    return [int(x) for x in levels] if isinstance(levels, list) else [0]


# -- writer ---------------------------------------------------------------------------------


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=_json_default))
    tmp.replace(path)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, BaseModel):
        return o.model_dump()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def fields_of(mesh: SurfaceMesh) -> list[FieldInfo]:
    """Describe the fields stored on ``mesh`` (cell-native ones first)."""
    out = [
        FieldInfo(name=n, channels=int(a.shape[1]), location="cell")
        for n, a in mesh.cell_fields.items()
    ]
    out += [
        FieldInfo(name=n, channels=int(a.shape[1]), location="point")
        for n, a in mesh.point_fields.items()
        if n not in mesh.cell_fields
    ]
    return out


def write_sample(
    root: Path,
    sample_id: str,
    lods: dict[int, SurfaceMesh],
    meta: SampleMeta,
    forces: dict[str, Any],
    *,
    stl_path: Path | None = None,
) -> SampleEntry:
    """Write one sample directory. Safe to call from worker processes."""
    root = Path(root)
    sdir = root / "samples" / sample_id
    sdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    write_surface(sdir / "surface.zarr", lods)
    meta = meta.model_copy(update={"lod_levels": sorted(lods)})
    if not meta.fields:
        meta.fields = fields_of(lods[0])
    meta.extra.setdefault("timings", {})["write_s"] = time.perf_counter() - t0
    write_json(sdir / "meta.json", meta.model_dump())
    write_json(sdir / "forces.json", forces)
    if stl_path is not None and Path(stl_path).exists():
        shutil.copyfile(stl_path, sdir / "geometry.stl")
    full = lods[0]
    return SampleEntry(
        id=sample_id,
        path=f"samples/{sample_id}",
        n_points=full.n_points,
        n_faces=full.n_faces,
        lod_levels=sorted(lods),
        geometry_params=dict(meta.geometry_params),
        forces={k: float(v) for k, v in forces.items() if isinstance(v, int | float)},
    )


class DatasetWriter:
    """Assemble a dataset directory: call :func:`write_sample` per sample (possibly in workers),
    then :meth:`finalize` once to write ``manifest.json``."""

    def __init__(
        self,
        root: str | Path,
        name: str,
        source: dict[str, Any] | None = None,
        *,
        domain: str = "cfd",
        license: str | None = None,  # noqa: A002 - contract field name
        citation: str | None = None,
        hub_id: str | None = None,
        time_units: str = "s",
    ) -> None:
        if domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "samples").mkdir(exist_ok=True)
        self.time_units = time_units
        self.manifest = Manifest(
            name=name,
            source=source or {},
            minor=FORMAT_MINOR,
            domain=domain,
            license=license,
            citation=citation,
            hub_id=hub_id,
        )

    def _write_v11(self, fn: Any, sample_id: str, kwargs: dict[str, Any]) -> SampleEntry:
        kwargs.setdefault("dataset", self.manifest.name)
        kwargs.setdefault("time_units", self.time_units)
        entry = fn(self.root, sample_id, **kwargs)
        self.add_entry(entry, read_entry_meta(self.root, entry))
        return entry

    def write_surface_sample(self, sample_id: str, mesh: Any, **kwargs: Any) -> SampleEntry:
        """See :func:`frontier_data.arrays.write_surface_sample`."""
        from frontier_data.arrays import write_surface_sample

        kwargs["mesh"] = mesh
        return self._write_v11(write_surface_sample, sample_id, kwargs)

    def write_mesh_sample(self, sample_id: str, **kwargs: Any) -> SampleEntry:
        """See :func:`frontier_data.arrays.write_mesh_sample`."""
        from frontier_data.arrays import write_mesh_sample

        return self._write_v11(write_mesh_sample, sample_id, kwargs)

    def write_grid_sample(self, sample_id: str, **kwargs: Any) -> SampleEntry:
        """See :func:`frontier_data.arrays.write_grid_sample`."""
        from frontier_data.arrays import write_grid_sample

        return self._write_v11(write_grid_sample, sample_id, kwargs)

    def write_points_sample(self, sample_id: str, **kwargs: Any) -> SampleEntry:
        """See :func:`frontier_data.arrays.write_points_sample`."""
        from frontier_data.arrays import write_points_sample

        return self._write_v11(write_points_sample, sample_id, kwargs)

    def write_sample(
        self,
        sample_id: str,
        lods: dict[int, SurfaceMesh],
        meta: SampleMeta,
        forces: dict[str, Any],
        *,
        stl_path: Path | None = None,
    ) -> SampleEntry:
        if not meta.fields:
            meta.fields = fields_of(lods[0])
        entry = write_sample(self.root, sample_id, lods, meta, forces, stl_path=stl_path)
        self.add_entry(entry, meta)
        return entry

    def append_sample(self, sample_id: str, kind: str = "mesh", **kwargs: Any) -> Any:
        """Step-by-step writer (:class:`frontier_data.arrays.SampleAppender`); the sample joins
        the manifest when the appender is closed."""
        from frontier_data.arrays import SampleAppender

        writer = self

        class _Tracked(SampleAppender):
            def close(self) -> SampleEntry:
                entry = super().close()
                writer.add_entry(entry, read_entry_meta(writer.root, entry))
                return entry

        kwargs.setdefault("dataset", self.manifest.name)
        kwargs.setdefault("time_units", self.time_units)
        return _Tracked(self.root, sample_id, kind, **kwargs)

    def add_entry(self, entry: SampleEntry, meta: SampleMeta | None = None) -> None:
        self.manifest.samples = [s for s in self.manifest.samples if s.id != entry.id] + [entry]
        if meta is not None:
            known = {f.name for f in self.manifest.fields}
            for f in meta.fields:
                if f.name not in known:
                    self.manifest.fields.append(f)
                    known.add(f.name)
            for c in meta.conditions:
                if c not in self.manifest.conditions:
                    self.manifest.conditions.append(c)
            for g in meta.geometry_params:
                if g not in self.manifest.geometry_params:
                    self.manifest.geometry_params.append(g)
            for q in meta.parameters:
                if q not in self.manifest.parameters:
                    self.manifest.parameters.append(q)

    def finalize(self) -> Manifest:
        self.manifest.samples.sort(key=lambda s: _natural_key(s.id))
        samples = self.manifest.samples
        self.manifest.stats = {
            "n_samples": len(samples),
            "n_points_total": int(sum(s.n_points for s in samples)),
            "n_faces_total": int(sum(s.n_faces for s in samples)),
            "n_steps_total": int(sum(s.n_steps for s in samples)),
        }
        if samples:
            kinds = [s.sample_kind for s in samples]
            self.manifest.sample_kind = max(set(kinds), key=kinds.count)
            transient = [s for s in samples if s.temporal == "transient"]
            self.manifest.temporal = "transient" if transient else "steady"
            if transient:
                metas = [read_entry_meta(self.root, s) for s in transient]
                times = [m.time for m in metas if m is not None and m.time is not None]
                dts = {round(t.dt, 12) if t.dt is not None else None for t in times}
                dt = next(iter(dts)) if len(dts) == 1 else None
                self.manifest.time = TimeInfo(
                    n_steps=max(s.n_steps for s in transient),
                    dt=dt,
                    units=times[0].units if times else self.time_units,
                    uniform=dt is not None and all(t.uniform for t in times),
                    t0=min((t.t0 for t in times), default=0.0),
                )
        write_json(self.root / "manifest.json", self.manifest.model_dump())
        return self.manifest


def read_entry_meta(root: Path, entry: SampleEntry) -> SampleMeta | None:
    """Read back the ``meta.json`` of ``entry`` (``None`` when missing)."""
    path = Path(root) / entry.path / "meta.json"
    if not path.exists():
        return None
    return SampleMeta.model_validate(json.loads(path.read_text()))


def _natural_key(s: str) -> tuple[Any, ...]:
    import re

    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))


# -- reader ---------------------------------------------------------------------------------


class Dataset:
    """Read access to a ``frontier-dataset`` v1 / v1.1 directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        mpath = self.root / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"{mpath} not found: not a frontier-dataset directory")
        self.manifest = Manifest.model_validate(json.loads(mpath.read_text()))
        if self.manifest.format != FORMAT_NAME:
            raise ValueError(f"unexpected dataset format {self.manifest.format!r}")

    @classmethod
    def open(cls, root: str | Path) -> Dataset:
        return cls(root)

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def sample_ids(self) -> list[str]:
        return self.manifest.sample_ids()

    def __len__(self) -> int:
        return len(self.manifest.samples)

    def entry(self, sample_id: str) -> SampleEntry:
        for s in self.manifest.samples:
            if s.id == sample_id:
                return s
        raise KeyError(sample_id)

    def sample_dir(self, sample_id: str) -> Path:
        """The directory of a sample. The manifest's ``path`` is data: an absolute path, a
        ``..`` component or a symlink that leaves the dataset directory is refused, so a
        manifest cannot make a reader (or an op) work on files elsewhere on disk."""
        rel = self.entry(sample_id).path
        pure = PurePosixPath(str(rel).replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts or (pure.parts and ":" in pure.parts[0]):
            raise ValueError(
                f"sample {sample_id!r}: path {rel!r} leaves the dataset directory "
                "(absolute paths and '..' are not allowed in a manifest)"
            )
        path = self.root / rel
        root = self.root.resolve()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(
                f"sample {sample_id!r}: {rel!r} resolves to {resolved}, outside the dataset "
                f"directory {root} (a symlink that leaves the dataset is not followed)"
            )
        return path

    def meta(self, sample_id: str) -> SampleMeta:
        return SampleMeta.model_validate(
            json.loads((self.sample_dir(sample_id) / "meta.json").read_text())
        )

    def forces(self, sample_id: str) -> dict[str, Any]:
        path = self.sample_dir(sample_id) / "forces.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def sample(self, sample_id: str) -> Sample:
        """Lazy v1.1 view of any sample kind: ``ds.sample(sid).field("u", t=slice(0, 4))``."""
        from frontier_data.arrays import Sample

        return Sample.open(self.sample_dir(sample_id), sample_id=sample_id)

    def samples(self) -> Iterator[Sample]:
        for sid in self.sample_ids:
            yield self.sample(sid)

    @property
    def is_transient(self) -> bool:
        return self.manifest.temporal == "transient"

    def surface(self, sample_id: str, lod: int = 0, *, fields: bool = True) -> SurfaceMesh:
        return read_surface(self.sample_dir(sample_id) / "surface.zarr", lod, fields=fields)

    def lod_levels(self, sample_id: str) -> list[int]:
        return self.entry(sample_id).lod_levels

    def geometry_params(self) -> dict[str, dict[str, float]]:
        return {s.id: dict(s.geometry_params) for s in self.manifest.samples}

    def save_manifest(self) -> None:
        write_json(self.root / "manifest.json", self.manifest.model_dump())

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
