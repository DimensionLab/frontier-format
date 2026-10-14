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
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from frontier_data.ingest.common import (
    ProgressFn,
    finish_dataset,
    safe_id,
    split_counts,
    subsample_time,
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


def _field_view(
    ds: Any, traj: int, steps: np.ndarray | None, space: tuple[int, ...], rank: int
) -> tuple[np.ndarray, bool]:
    """Read one trajectory of a field dataset as ``([T], *space, C)`` float32."""
    sample_varying = _flag(ds, "sample_varying", True)
    time_varying = _flag(ds, "time_varying", True)
    nd = len(space)
    dim_varying = (
        [bool(x) for x in np.asarray(ds.attrs["dim_varying"]).reshape(-1)]
        if "dim_varying" in ds.attrs
        else [True] * nd
    )
    sel: list[Any] = []
    if sample_varying:
        sel.append(traj)
    if time_varying:
        sel.append(slice(None))
    arr = np.asarray(ds[tuple(sel)] if sel else ds[...])
    if time_varying and steps is not None:
        arr = arr[steps]
    lead = 1 if time_varying else 0
    # insert the space axes that do not vary, then broadcast
    shape = list(arr.shape)
    full: list[int] = shape[:lead]
    it = iter(shape[lead : lead + sum(dim_varying)])
    for i in range(nd):
        full.append(next(it) if dim_varying[i] else 1)
    tensor = shape[lead + sum(dim_varying) :]
    arr = arr.reshape(*full, *tensor)
    arr = np.broadcast_to(arr, (*full[:lead], *space, *tensor))
    n_comp = int(np.prod(tensor)) if tensor else 1
    arr = arr.reshape(*arr.shape[: lead + nd], n_comp)
    del rank
    return np.ascontiguousarray(arr, dtype=np.float32), time_varying


def read_well_file(
    path: str | Path,
    *,
    max_trajectories: int | None = None,
    fields: Sequence[str] | None = None,
    time_stride: int = 1,
    max_steps: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one dict per trajectory of a The Well HDF5 file: ``fields`` (time-varying,
    ``(T, *space, C)``), ``static``, ``times``, ``coords``, ``boundary_conditions``,
    ``parameters``, ``attrs``, ``index``."""
    import h5py

    with h5py.File(str(path), "r") as f:
        dims: Any = f["dimensions"]
        axes = [str(a) for a in _decode(dims.attrs.get("spatial_dims", ["x", "y", "z"]))]
        axes = [a for a in axes if a in dims]
        coords = {a: np.asarray(dims[a][...], dtype=np.float64).reshape(-1) for a in axes}
        space = tuple(len(coords[a]) for a in axes)
        time_ds = dims["time"]
        n_traj = int(f.attrs.get("n_trajectories", 1))
        bcs: dict[str, Any] = {}
        if "boundary_conditions" in f:
            bgroup: Any = f["boundary_conditions"]
            for name in bgroup:
                g = bgroup[name]
                kind = str(_decode(g.attrs.get("bc_type", name))).lower()
                for ax in _decode(g.attrs.get("associated_dims", [])) or []:
                    prev = bcs.get(str(ax))
                    bcs[str(ax)] = kind if prev in (None, kind) else [*np.atleast_1d(prev), kind]
        attrs = {k: _decode(v) for k, v in f.attrs.items()}
        scalars: Any = f["scalars"] if "scalars" in f else {}
        n = n_traj if max_trajectories is None else min(n_traj, max_trajectories)
        for traj in range(n):
            times = np.asarray(time_ds[...], dtype=np.float64)
            if times.ndim == 2:
                times = times[traj]
            steps = subsample_time(len(times), time_stride, max_steps)
            out_fields: dict[str, np.ndarray] = {}
            static: dict[str, np.ndarray] = {}
            for rank, gname in enumerate(("t0_fields", "t1_fields", "t2_fields")):
                if gname not in f:
                    continue
                group: Any = f[gname]
                for name in group:
                    if fields is not None and name not in fields:
                        continue
                    arr, tv = _field_view(group[name], traj, steps, space, rank)
                    (out_fields if tv else static)[str(name)] = arr
            params: dict[str, float] = {}
            for name in scalars:
                v = np.asarray(scalars[name][...], dtype=np.float64).reshape(-1)
                if v.size == 1:
                    params[str(name)] = float(v[0])
                elif v.size == n_traj:
                    params[str(name)] = float(v[traj])
            yield {
                "index": traj,
                "fields": out_fields,
                "static": static,
                "times": times[steps],
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
