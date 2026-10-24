# SPDX-License-Identifier: Apache-2.0
"""The Well (Ohana et al., NeurIPS 2024, arXiv:2412.00568) -> transient grid samples.

Each dataset is a HF dataset repo ``polymathic-ai/<name>`` with ``data/{train,valid,test}/*.hdf5``.
One HDF5 file holds several trajectories that share the simulation parameters:

* root attrs: ``dataset_name``, ``n_spatial_dims``, ``n_trajectories``, ``grid_type``,
  ``simulation_parameters``;
* ``dimensions/``: ``time`` and the space axes named in its ``spatial_dims`` attribute;
* ``scalars/``: the simulation parameters (constant per file or one value per trajectory);
* ``boundary_conditions/<name>``: attrs ``bc_type`` (``PERIODIC`` / ``WALL`` / ``OPEN``) and
  ``associated_dims``;
* ``t0_fields`` (scalars), ``t1_fields`` (vectors), ``t2_fields`` (rank-2 tensors): datasets
  shaped ``([n_traj], [n_time], *space, *tensor)`` where the leading axes exist only when the
  dataset attrs ``sample_varying`` / ``time_varying`` are true and a space axis only when its
  entry in ``dim_varying`` is true.

A trajectory becomes one ``grid.zarr`` sample: time-varying fields under ``fields/`` as
``(T, X, Y[, Z], C)``, constant fields under ``static/``, the scalars as ``parameters``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from arrayroute.hdf5 import HDF5ReadSession
from arrayroute.plan import plan, tiles
from arrayroute.reader import ArrayReader, ReadStats
from frontier_data.ingest.common import (
    ProgressFn,
    finish_dataset,
    safe_id,
    split_counts,
)
from frontier_data.sources._offline import require_online

log = logging.getLogger(__name__)

HF_ORG = "polymathic-ai"
LICENSE = "CC-BY-4.0"
CITATION = (
    "Ohana et al., The Well: a Large-Scale Collection of Diverse Physics Simulations for "
    "Machine Learning, NeurIPS 2024 Datasets and Benchmarks (arXiv:2412.00568)"
)
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
#: manifest.domain per dataset (default "cfd")
DOMAINS = {
    "active_matter": "other",
    "gray_scott_reaction_diffusion": "other",
    "acoustic_scattering_discontinuous": "other",
    "acoustic_scattering_inclusions": "other",
    "acoustic_scattering_maze": "other",
    "helmholtz_staircase": "other",
    "planetswe": "geophysical",
    "viscoelastic_instability": "cfd",
}


def _decode(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return [_decode(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _flag(ds: Any, name: str, default: bool) -> bool:
    return bool(ds.attrs[name]) if name in ds.attrs else default


def _materialize(
    reader: ArrayReader,
    selection: tuple,
    *,
    output_limit: int,
    cancelled: Callable[[], bool] | None,
    stats: ReadStats | None,
) -> np.ndarray:
    if cancelled is not None and cancelled():
        raise InterruptedError("Well ingestion cancelled")
    request = plan(reader.backend.spec, selection)
    if request.output_bytes > output_limit:
        raise MemoryError("Selected Well values exceed the materialization allowance")
    next(tiles(reader.backend.spec, selection, buffer_bytes=reader.buffer_bytes), None)
    result = np.empty(request.output_shape, dtype=reader.backend.dtype)
    for tile, values in reader.stream(selection, cancelled=cancelled, stats=stats):
        result[tile.destination] = values
        del values
    return result


def _field_plan(ds: Any, traj: int, step_slice: slice, space: tuple[int, ...]):
    sample_varying = _flag(ds, "sample_varying", True)
    time_varying = _flag(ds, "time_varying", True)
    varying = (
        [bool(x) for x in np.asarray(ds.attrs["dim_varying"]).reshape(-1)]
        if "dim_varying" in ds.attrs
        else [True] * len(space)
    )
    if len(varying) != len(space):
        raise ValueError(f"{ds.name}: dim_varying must match the spatial axes")
    selection = []
    if sample_varying:
        selection.append(traj)
    if time_varying:
        selection.append(step_slice)
    selection += [slice(None)] * (ds.ndim - len(selection))
    source_shape = []
    for size, sel in zip(ds.shape, selection, strict=True):
        if isinstance(sel, slice):
            source_shape.append(len(range(*sel.indices(size))))
        elif not 0 <= sel < size:
            raise ValueError(f"{ds.name}: trajectory index exceeds source shape")
    lead = int(time_varying)
    if len(source_shape) < lead + sum(varying):
        raise ValueError(f"{ds.name}: field rank does not match its spatial metadata")
    spatial = iter(source_shape[lead : lead + sum(varying)])
    full = source_shape[:lead]
    for size, active in zip(space, varying, strict=True):
        actual = next(spatial) if active else 1
        if active and actual != size:
            raise ValueError(f"{ds.name}: varying spatial extent disagrees with coordinates")
        full.append(actual)
    tensor = source_shape[lead + sum(varying) :]
    expanded = (*source_shape[:lead], *space, math.prod(tensor))
    return tuple(selection), (*full, *tensor), expanded, time_varying


def read_well_file(
    path: str | Path,
    *,
    max_trajectories: int | None = None,
    fields: Sequence[str] | None = None,
    time_stride: int = 1,
    max_steps: int | None = None,
    read_buffer_bytes: int = 32 << 20,
    hdf5_strategy: Literal["auto", "native"] = "auto",
    materialization_bytes: int = 512 << 20,
    cancelled: Callable[[], bool] | None = None,
    read_stats: ReadStats | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Yield selected trajectories with ArrayRoute payload reads.

    Time selection is applied before reading. The materialization allowance covers retained
    fields plus conservative native-to-float32 conversion/broadcast buffers per trajectory.
    Read buffers, coordinates/timestamps, native codec scratch and caller-retained earlier
    trajectories are additional. Sources must stay immutable while the iterator is open.
    This is a bounded materializing API, not a streaming grid writer.
    """
    for label, value in (
        ("time_stride", time_stride),
        ("read_buffer_bytes", read_buffer_bytes),
        ("materialization_bytes", materialization_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    for label, value in (("max_steps", max_steps), ("max_trajectories", max_trajectories)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{label} must be a positive integer or None")
    if fields is not None and (isinstance(fields, str) or len(set(fields)) != len(fields)):
        raise ValueError("Fields must be a sequence of unique names")

    def check_cancel():
        if cancelled is not None and cancelled():
            raise InterruptedError("Well ingestion cancelled")

    with HDF5ReadSession(
        path, buffer_bytes=read_buffer_bytes, strided_reads=hdf5_strategy
    ) as session:
        check_cancel()
        f = session.metadata()
        dims = session.metadata("dimensions")
        axes = [str(a) for a in _decode(dims.attrs.get("spatial_dims", ["x", "y", "z"]))]
        axes = [a for a in axes if a in dims]
        if not 1 <= len(axes) <= 3 or len(set(axes)) != len(axes):
            raise ValueError("Well dimensions must declare one to three unique spatial axes")
        coords = {}
        for axis in axes:
            r = session.open(f"dimensions/{axis}")
            coords[axis] = (
                _materialize(
                    r,
                    (slice(None),) * len(r.backend.spec.shape),
                    output_limit=read_buffer_bytes // 2,
                    cancelled=cancelled,
                    stats=read_stats,
                )
                .astype(np.float64)
                .reshape(-1)
            )
        space = tuple(len(coords[a]) for a in axes)
        if not all(space):
            raise ValueError("Well spatial coordinates must not be empty")
        time_ds = session.metadata("dimensions/time")
        if time_ds.ndim not in (1, 2):
            raise ValueError("Well time coordinates must have one or two dimensions")
        n_traj = int(f.attrs.get("n_trajectories", 1))
        bcs: dict[str, Any] = {}
        if "boundary_conditions" in f:
            bgroup = session.metadata("boundary_conditions")
            for name in sorted(bgroup):
                g = session.metadata(f"boundary_conditions/{name}")
                kind = str(_decode(g.attrs.get("bc_type", name))).lower()
                for ax in _decode(g.attrs.get("associated_dims", [])) or []:
                    prev = bcs.get(str(ax))
                    bcs[str(ax)] = kind if prev in (None, kind) else [*np.atleast_1d(prev), kind]
        attrs = {k: _decode(v) for k, v in f.attrs.items()}
        scalars = session.metadata("scalars") if "scalars" in f else {}
        n = n_traj if max_trajectories is None else min(n_traj, max_trajectories)
        for traj in range(n):
            check_cancel()
            count = len(range(0, time_ds.shape[-1], time_stride))
            count = min(count, max_steps) if max_steps is not None else count
            if count < 1:
                raise ValueError("Well trajectory has no time steps")
            step_slice = slice(0, min(time_ds.shape[-1], count * time_stride), time_stride)
            time_selection = ((traj,) if time_ds.ndim == 2 else ()) + (step_slice,)
            times = _materialize(
                session.open("dimensions/time"),
                time_selection,
                output_limit=read_buffer_bytes // 2,
                cancelled=cancelled,
                stats=read_stats,
            ).astype(np.float64)
            plans = []
            seen = set()
            retained = 0
            for gname in ("t0_fields", "t1_fields", "t2_fields"):
                if gname not in f:
                    continue
                group = session.metadata(gname)
                for name in sorted(group):
                    if fields is not None and name not in fields:
                        continue
                    if name in seen:
                        raise ValueError(f"Duplicate Well field name: {name}")
                    seen.add(name)
                    check_cancel()
                    ds = session.metadata(f"{gname}/{name}")
                    selection, reshaped, expanded, tv = _field_plan(ds, traj, step_slice, space)
                    if tv and expanded[0] != count:
                        raise ValueError(
                            f"{name}: field time extent disagrees with time coordinates"
                        )
                    reader = session.open(f"{gname}/{name}") if ds.ndim else None
                    if np.dtype(ds.dtype).kind not in "biuf":
                        raise TypeError(f"{name}: Well fields must be real numeric values")
                    raw_bytes = (
                        plan(reader.backend.spec, selection).output_bytes
                        if reader is not None
                        else ds.dtype.itemsize
                    )
                    final_bytes = math.prod(expanded) * 4
                    if retained + raw_bytes + final_bytes > materialization_bytes:
                        raise MemoryError(
                            f"{name}: selected trajectory exceeds materialization_bytes; "
                            "select fewer fields/steps or explicitly increase the allowance"
                        )
                    if reader is not None:
                        next(
                            tiles(reader.backend.spec, selection, buffer_bytes=read_buffer_bytes),
                            None,
                        )
                    plans.append(
                        (name, f"{gname}/{name}", reader, selection, reshaped, expanded, tv)
                    )
                    retained += final_bytes
            if fields is not None and set(fields) - seen:
                raise ValueError(f"Unknown Well fields: {sorted(set(fields) - seen)}")
            out_fields: dict[str, np.ndarray] = {}
            static: dict[str, np.ndarray] = {}
            for name, source_path, reader, selection, reshaped, expanded, tv in plans:
                check_cancel()
                arr = (
                    session.read_scalar(source_path)
                    if reader is None
                    else _materialize(
                        reader,
                        selection,
                        output_limit=materialization_bytes,
                        cancelled=cancelled,
                        stats=read_stats,
                    )
                )
                # Flatten tensor components only after broadcasting absent spatial axes.
                raw = arr.reshape(reshaped)
                lead = int(tv)
                tensor = reshaped[lead + len(space) :]
                view = np.broadcast_to(raw, (*expanded[: lead + len(space)], *tensor))
                value = np.ascontiguousarray(view, dtype=np.float32).reshape(expanded)
                (out_fields if tv else static)[name] = value
                del arr, raw, view, value
            params: dict[str, float] = {}
            for name in sorted(scalars):
                check_cancel()
                ds = session.metadata(f"scalars/{name}")
                if ds.shape == ():
                    v = session.read_scalar(f"scalars/{name}")
                elif ds.size == 1:
                    v = session.open(f"scalars/{name}").read((0,) * ds.ndim)
                elif ds.ndim == 1 and ds.size == n_traj:
                    v = session.open(f"scalars/{name}").read((traj,))
                else:
                    continue
                params[str(name)] = float(v.item())
            yield {
                "index": traj,
                "fields": out_fields,
                "static": static,
                "times": times,
                "coords": coords,
                "axes": axes,
                "boundary_conditions": bcs,
                "parameters": params,
                "attrs": attrs,
            }


def convert_the_well(
    files: Mapping[str, Sequence[str | Path]],
    out: str | Path,
    *,
    name: str,
    trajectories: dict[str, int | None] | None = None,
    per_file: int | None = None,
    fields: Sequence[str] | None = None,
    time_stride: int = 1,
    max_steps: int | None = None,
    read_buffer_bytes: int = 32 << 20,
    hdf5_strategy: Literal["auto", "native"] = "auto",
    materialization_bytes: int = 512 << 20,
    cancelled: Callable[[], bool] | None = None,
    quality: bool = True,
    provenance_files: dict[str, dict[str, Any]] | None = None,
    progress: ProgressFn | None = None,
    hub_id: str | None = None,
    id_with_split: bool = True,
) -> Any:
    """Convert local The Well HDF5 files into a ``frontier-dataset`` v1.1 at ``out``.

    ``files`` maps a split name (``train`` / ``val`` / ``test``; The Well's ``valid`` is
    accepted) to HDF5 paths; ``trajectories[split]`` caps the number of trajectories taken
    from that split and ``per_file`` the number taken from each file (to spread a small budget
    over the parameter values). Sample ids are ``<split>_<file suffix>_<index>``; pass
    ``id_with_split=False`` when the caller re-splits afterwards. Returns the opened dataset.
    """
    from frontier_data.dataset import DatasetWriter, Provenance, RawFileRecord

    writer = DatasetWriter(
        out,
        name=name,
        source={
            "kind": "the_well",
            "repo": f"{HF_ORG}/{name}",
            "time_stride": time_stride,
            "max_steps": max_steps,
        },
        domain=DOMAINS.get(name, "cfd"),
        license=LICENSE,
        citation=CITATION,
        hub_id=hub_id or f"the-well-{name.replace('_', '-')}",
        time_units="simulation",
    )
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    counters: dict[str, int] = {}
    paths = [(SPLIT_MAP.get(s, s), Path(p)) for s, ps in files.items() for p in ps]
    stems = [p.stem for _, p in paths]
    common = _common_prefix(stems)
    total = len(paths)
    for k, (split, path) in enumerate(sorted(paths, key=lambda sp: (sp[0], sp[1].name))):
        cap = (trajectories or {}).get(split)
        left = None if cap is None else cap - len(splits[split])
        if left is not None and left <= 0:
            continue
        n_take = min(x for x in (left, per_file, 10**9) if x is not None)
        short = safe_id(path.stem[len(common) :]) or f"f{k}"
        for traj in read_well_file(
            path,
            max_trajectories=n_take,
            fields=fields,
            time_stride=time_stride,
            max_steps=max_steps,
            read_buffer_bytes=read_buffer_bytes,
            hdf5_strategy=hdf5_strategy,
            materialization_bytes=materialization_bytes,
            cancelled=cancelled,
        ):
            if id_with_split:
                sid = f"{split}_{short}_{traj['index']:03d}"
            else:
                counters[short] = counters.get(short, -1) + 1
                sid = f"{short}_{counters[short]:03d}"
            rec = (provenance_files or {}).get(str(path))
            prov = Provenance(
                source=f"huggingface:{HF_ORG}/{name}",
                license=LICENSE,
                citation=CITATION,
                files=[RawFileRecord(role="hdf5", **rec)] if rec else [],
            )
            attrs = traj["attrs"]
            writer.write_grid_sample(
                sid,
                fields=traj["fields"],
                static=traj["static"] or None,
                times=traj["times"],
                coords=traj["coords"],
                boundary_conditions=traj["boundary_conditions"],
                parameters=traj["parameters"],
                meta={
                    "solver": "the_well",
                    "physics": str(attrs.get("dataset_name", name)),
                    "fidelity": "dns",
                    "provenance": prov,
                    "extra": {
                        "source_file": path.name,
                        "source_split": split,
                        "trajectory_index": traj["index"],
                        "grid_type": attrs.get("grid_type"),
                        "time_stride": time_stride,
                    },
                },
            )
            splits[split].append(sid)
        if progress:
            progress(f"converted {path.name}", k + 1, total)
    strategy = {"name": "source", "note": "The Well's own train/valid/test files"}
    return finish_dataset(writer, splits, strategy=strategy, quality=quality, progress=progress)


def _common_prefix(stems: Sequence[str]) -> str:
    """Longest common prefix of the file stems, cut at a ``_`` boundary and keeping the name of
    a trailing parameter (``layer_tcool_0.03`` / ``layer_tcool_0.10`` -> ``layer_``)."""
    if len(stems) < 2:
        return ""
    prefix = stems[0]
    for s in stems[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
    prefix = prefix[: prefix.rfind("_") + 1]
    if prefix and all(s[len(prefix) : len(prefix) + 1].isdigit() for s in stems):
        prefix = prefix[: prefix.rstrip("_").rfind("_") + 1]
    return prefix


def list_well_files(name: str, revision: str | None = None) -> dict[str, list[tuple[str, int]]]:
    """``{split: [(repo_path, size_bytes), ...]}`` for ``polymathic-ai/<name>``."""
    require_online(f"listing {HF_ORG}/{name} on the Hugging Face Hub")
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(f"{HF_ORG}/{name}", revision=revision, files_metadata=True)
    out: dict[str, list[tuple[str, int]]] = {"train": [], "valid": [], "test": []}
    for s in info.siblings or []:
        parts = s.rfilename.split("/")
        if (
            len(parts) == 3
            and parts[0] == "data"
            and parts[1] in out
            and parts[2].endswith((".hdf5", ".h5"))
        ):
            out[parts[1]].append((s.rfilename, int(s.size or 0)))
    return {k: sorted(v) for k, v in out.items()}


def ingest_the_well(
    name: str,
    out: str | Path,
    *,
    trajectories: int | None = 8,
    source_splits: Sequence[str] = ("train", "valid", "test"),
    per_file: int | None = None,
    raw_dir: str | Path | None = None,
    fields: Sequence[str] | None = None,
    time_stride: int = 1,
    max_steps: int | None = None,
    read_buffer_bytes: int = 32 << 20,
    hdf5_strategy: Literal["auto", "native"] = "auto",
    materialization_bytes: int = 512 << 20,
    cancelled: Callable[[], bool] | None = None,
    max_bytes: float | None = None,
    resplit: bool = False,
    seed: int = 0,
    download: bool = True,
    revision: str | None = None,
    quality: bool = True,
    progress: ProgressFn | None = None,
) -> Any:
    """Download a bounded subset of a The Well dataset and convert it.

    ``trajectories`` is the total budget, distributed 80/10/10 over The Well's own
    train/valid/test files (``None`` = everything in ``source_splits``). Files are fetched one
    at a time, in name order, until each split has enough trajectories or ``max_bytes`` of
    downloads is reached. ``source_splits=("valid", "test")`` with ``resplit=True`` is the
    cheap option: those files are ~6x smaller than the train files, and the trajectories are
    then re-split at random *by trajectory* into train/val/test.
    """
    import h5py

    from frontier_data.sources.huggingface import ProvenanceStore, resolve_url

    raw = Path(raw_dir) if raw_dir else Path("datasets/raw/the_well") / name
    raw.mkdir(parents=True, exist_ok=True)
    src = [s for s in ("train", "valid", "test") if s in source_splits]
    if trajectories is None:
        budget: dict[str, int | None] = dict.fromkeys(src)
    elif resplit or len(src) < 3:
        each, rem = divmod(trajectories, len(src))
        budget = {s: each + (1 if i < rem else 0) for i, s in enumerate(src)}
    else:
        budget = dict(zip(src, split_counts(trajectories), strict=True))
    if download:
        listing = list_well_files(name, revision)
    else:
        listing = {
            s: [
                (str(p.relative_to(raw)), p.stat().st_size)
                for p in sorted((raw / "data" / s).glob("*.hdf5"))
            ]
            for s in src
        }
    store = ProvenanceStore(raw)
    files: dict[str, list[Path]] = {}
    prov: dict[str, dict[str, Any]] = {}
    spent = 0
    for split in src:
        have = 0
        want = budget[split]
        for repo_path, size in listing.get(split, []):
            if want is not None and have >= want:
                break
            local = raw / repo_path
            if not local.exists():
                if not download:
                    continue
                if max_bytes is not None and spent + size > max_bytes:
                    log.warning("the_well: byte budget reached before %s", repo_path)
                    break
                require_online(f"downloading {HF_ORG}/{name}/{repo_path}")
                from huggingface_hub import hf_hub_download

                log.info("the_well: downloading %s (%.0f MB)", repo_path, size / 1e6)
                hf_hub_download(
                    repo_id=f"{HF_ORG}/{name}",
                    filename=repo_path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=str(raw),
                )
                spent += size
            url = resolve_url(f"{HF_ORG}/{name}", repo_path, revision, "dataset")
            rec = store.record(repo_path, local, url, revision)
            prov[str(local)] = {
                "repo_path": repo_path,
                "url": url,
                "sha256": rec.sha256,
                "size": rec.size,
                "revision": revision,
            }
            with h5py.File(str(local), "r") as f:
                n = int(f.attrs.get("n_trajectories", 1))
            have += n if per_file is None else min(n, per_file)
            files.setdefault(split, []).append(local)
            if progress:
                progress(f"fetched {repo_path}", have, want or have)
    store.save()
    if not files:
        raise FileNotFoundError(f"no The Well files for {name!r} under {raw}")
    ds = convert_the_well(
        files,
        out,
        name=name,
        trajectories={SPLIT_MAP[s]: budget[s] for s in src},
        per_file=per_file,
        fields=fields,
        time_stride=time_stride,
        max_steps=max_steps,
        read_buffer_bytes=read_buffer_bytes,
        hdf5_strategy=hdf5_strategy,
        materialization_bytes=materialization_bytes,
        cancelled=cancelled,
        quality=quality and not (resplit or len(src) < 3),
        provenance_files=prov,
        progress=progress,
        id_with_split=not (resplit or len(src) < 3),
    )
    if resplit or len(src) < 3:
        from frontier_data.splits import random_split

        res = random_split(ds.sample_ids, seed=seed)
        res.strategy["note"] = "re-split by trajectory (one sample = one whole trajectory)"
        ds.manifest.splits = res.splits
        ds.manifest.split_strategy = res.strategy
        ds.manifest.split_families = {"random": res.as_family()}
        ds.save_manifest()
        if quality:
            from frontier_data.quality import run_quality

            run_quality(ds, progress=progress)
    return ds
