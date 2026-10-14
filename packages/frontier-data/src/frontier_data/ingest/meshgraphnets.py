# SPDX-License-Identifier: Apache-2.0
"""DeepMind MeshGraphNets datasets (Pfaff et al., ICLR 2021, arXiv:2010.03409) -> transient
mesh samples, without TensorFlow.

The datasets live at ``https://storage.googleapis.com/dm-meshgraphnets/<name>/`` as
``meta.json`` + ``{train,valid,test}.tfrecord``. A TFRecord file is a sequence of
``[u64 length][u32 crc][payload][u32 crc]`` frames; every payload is a ``tf.train.Example``
protobuf whose features are single ``bytes`` values holding the raw little-endian array of one
feature (``meta.json`` gives dtype and shape; the leading shape entry is 1 for ``static``
features and ``trajectory_length`` for ``dynamic`` ones). Both layers are simple enough to
parse by hand, which also lets us **stream** just the first N trajectories of a multi-GB file
with an HTTP range request.

Mapping to the v1.1 mesh layout: ``mesh_pos`` -> ``points``, ``cells`` -> ``cells``,
``world_pos`` (cloth / plate) -> ``points_t``, other dynamic features -> ``fields/``, other
static features (``node_type``) -> ``static/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from frontier_data.arrays import auto_steps_per_chunk
from frontier_data.ingest.common import ProgressFn, finish_dataset, split_counts, subsample_time
from frontier_data.sources._offline import require_online

log = logging.getLogger(__name__)

BASE_URL = "https://storage.googleapis.com/dm-meshgraphnets"
LICENSE = "CC-BY-4.0"
CITATION = (
    "Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia, Learning Mesh-Based Simulation with Graph "
    "Networks, ICLR 2021 (arXiv:2010.03409)"
)
SPLIT_FILES = {"train": "train", "val": "valid", "test": "test"}
NODE_INFLOW, NODE_OUTFLOW, NODE_WALL = 4, 5, 6


# -- TFRecord framing -----------------------------------------------------------------------


def iter_tfrecord(stream: BinaryIO) -> Iterator[bytes]:
    """Yield the payload of every complete record of a TFRecord stream (CRCs not verified)."""
    while True:
        head = stream.read(12)
        if len(head) < 12:
            return
        (length,) = struct.unpack("<Q", head[:8])
        data = stream.read(length)
        tail = stream.read(4)
        if len(data) < length or len(tail) < 4:
            return  # truncated last record
        yield data


def count_complete_records(path: Path) -> tuple[int, int]:
    """``(n_records, n_bytes)`` of the complete records at the start of ``path``."""
    n, pos = 0, 0
    size = path.stat().st_size
    with path.open("rb") as f:
        while pos + 12 <= size:
            f.seek(pos)
            (length,) = struct.unpack("<Q", f.read(8))
            end = pos + 12 + length + 4
            if end > size:
                break
            n, pos = n + 1, end
    return n, pos


# -- protobuf: tf.train.Example -------------------------------------------------------------


def _varint(buf: bytes | memoryview, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(buf: memoryview) -> Iterator[tuple[int, int, Any]]:
    """Iterate ``(field_number, wire_type, value)`` over a protobuf message."""
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, pos = _varint(buf, pos)
        elif wire == 1:
            val, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            n, pos = _varint(buf, pos)
            val, pos = buf[pos : pos + n], pos + n
        elif wire == 5:
            val, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield num, wire, val


def parse_example(payload: bytes) -> dict[str, memoryview]:
    """``tf.train.Example`` -> ``{feature name: raw bytes of its first bytes_list value}``."""
    out: dict[str, memoryview] = {}
    for num, _, features in _fields(memoryview(payload)):
        if num != 1:  # Example.features
            continue
        for fnum, _, entry in _fields(features):
            if fnum != 1:  # Features.feature (map entry)
                continue
            key, value = "", None
            for enum_, _, v in _fields(entry):
                if enum_ == 1:
                    key = bytes(v).decode()
                elif enum_ == 2:  # Feature
                    for knum, _, lst in _fields(v):
                        if knum == 1:  # bytes_list
                            for bnum, _, raw in _fields(lst):
                                if bnum == 1 and value is None:
                                    value = raw
            if key and value is not None:
                out[key] = value
    return out


def decode_trajectory(payload: bytes, meta: dict[str, Any]) -> dict[str, np.ndarray]:
    """One record -> ``{feature: array}`` shaped as in ``meta.json`` (static features keep
    their leading 1)."""
    raw = parse_example(payload)
    out: dict[str, np.ndarray] = {}
    for name, spec in meta["features"].items():
        if name not in raw:
            continue
        arr = np.frombuffer(raw[name], dtype=np.dtype(spec["dtype"]).newbyteorder("<"))
        out[name] = arr.reshape(spec["shape"])
    return out


def read_tfrecord_trajectories(
    path: str | Path, meta: dict[str, Any], max_trajectories: int | None = None
) -> Iterator[dict[str, np.ndarray]]:
    """Yield decoded trajectories from a local ``.tfrecord`` (possibly a truncated prefix)."""
    with Path(path).open("rb") as f:
        for i, payload in enumerate(iter_tfrecord(f)):
            if max_trajectories is not None and i >= max_trajectories:
                return
            yield decode_trajectory(payload, meta)


# -- download (prefix streaming) ------------------------------------------------------------


def _checked_url(url: str) -> str:
    """``urllib`` also opens ``file:`` and ``ftp:`` URLs; a download is http(s) only."""
    if not url.lower().startswith(("https://", "http://")):
        raise ValueError(f"refusing to download {url!r}: only http(s) URLs are fetched")
    return url


def fetch_prefix(
    url: str, dest: Path, n_records: int, *, chunk: int = 1 << 20, timeout: float = 60.0
) -> tuple[int, int]:
    """Make ``dest`` hold at least the first ``n_records`` records of the TFRecord at ``url``.
    Resumes from the end of the last complete local record with an HTTP range request and stops
    as soon as enough records are on disk. Returns ``(records, bytes)`` now available."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have, valid = count_complete_records(dest) if dest.exists() else (0, 0)
    if have >= n_records:
        return have, valid
    require_online(f"downloading {url}", url=_checked_url(url))
    req = urllib.request.Request(url, headers={"Range": f"bytes={valid}-"})
    with (
        urllib.request.urlopen(req, timeout=timeout) as resp,
        dest.open("r+b" if valid else "wb") as f,
    ):
        f.seek(valid)
        f.truncate()
        need = 0  # bytes still missing from the record being received
        head = b""
        while have < n_records:
            block = resp.read(chunk)
            if not block:
                break
            f.write(block)
            view = memoryview(block)
            while len(view):
                if need == 0:
                    take = min(12 - len(head), len(view))
                    head += bytes(view[:take])
                    view = view[take:]
                    if len(head) == 12:
                        need = struct.unpack("<Q", head[:8])[0] + 4
                        head = b""
                else:
                    take = min(need, len(view))
                    need -= take
                    view = view[take:]
                    if need == 0:
                        have += 1
                        if have >= n_records:
                            break
    return count_complete_records(dest)


def fetch_meta(name: str, raw_dir: Path, base_url: str = BASE_URL) -> dict[str, Any]:
    path = raw_dir / "meta.json"
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        url = _checked_url(f"{base_url}/{name}/meta.json")
        require_online(f"downloading {url}", url=url)
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - scheme checked
            path.write_bytes(r.read())
    return json.loads(path.read_text())


# -- conversion -----------------------------------------------------------------------------


def cylinder_parameters(points: np.ndarray, node_type: np.ndarray, velocity0: np.ndarray):
    """Conditioning scalars of a ``cylinder_flow`` trajectory recovered from the mesh: inlet
    velocity (max / mean of the parabolic profile) and the cylinder's position and diameter
    (wall nodes away from the channel walls)."""
    nt = node_type.reshape(-1)
    out: dict[str, float] = {}
    inflow = nt == NODE_INFLOW
    if inflow.any():
        u = velocity0[inflow, 0]
        out["u_inlet_max"] = float(u.max())
        out["u_inlet_mean"] = float(u.mean())
    wall = nt == NODE_WALL
    if wall.any():
        y = points[:, 1]
        ymin, ymax = float(y.min()), float(y.max())
        eps = 1e-6 * max(ymax - ymin, 1.0)
        cyl = wall & (y > ymin + eps) & (y < ymax - eps)
        if cyl.sum() >= 3:
            c = points[cyl]
            centre = 0.5 * (c.min(axis=0) + c.max(axis=0))
            out["cylinder_x"] = float(centre[0])
            out["cylinder_y"] = float(centre[1])
            out["cylinder_d"] = float((c.max(axis=0) - c.min(axis=0)).mean())
    return out


def trajectory_to_sample(
    traj: dict[str, np.ndarray],
    meta: dict[str, Any],
    *,
    time_stride: int = 1,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Map a decoded trajectory to keyword arguments of
    :func:`frontier_data.arrays.write_mesh_sample`."""
    feats = meta["features"]
    n_steps = int(meta.get("trajectory_length") or max(v.shape[0] for v in traj.values()))
    steps = subsample_time(n_steps, time_stride, max_steps)
    dt = float(meta.get("dt") or 1.0)
    points = np.asarray(traj["mesh_pos"][0], dtype=np.float32)
    cells = np.asarray(traj["cells"][0], dtype=np.int32)
    fields: dict[str, np.ndarray] = {}
    static: dict[str, np.ndarray] = {}
    points_t = None
    for name, arr in traj.items():
        if name in ("mesh_pos", "cells"):
            continue
        dynamic = feats[name]["type"] == "dynamic"
        if name == "world_pos":
            points_t = np.asarray(arr[steps] if dynamic else arr, dtype=np.float32)
            continue
        if dynamic:
            fields[name] = np.asarray(arr[steps], dtype=np.float32)
        else:
            a = arr[0]
            static[name] = a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a
    kwargs: dict[str, Any] = {
        "points": points,
        "cells": cells,
        "fields": fields,
        "static": static,
        "times": dt * steps.astype(np.float64),
        # a step of these meshes is a few kB: group steps so a trajectory is tens of chunk
        # files instead of hundreds
        "steps_per_chunk": auto_steps_per_chunk((len(points), 3)),
    }
    if points_t is not None:
        if points_t.shape[-1] != points.shape[-1]:  # cloth: 2D material mesh, 3D world
            kwargs["points"] = points_t[0]
            static["mesh_pos"] = points
        kwargs["points_t"] = points_t
    return kwargs


def convert_meshgraphnets(
    files: dict[str, str | Path],
    meta: dict[str, Any],
    out: str | Path,
    *,
    name: str = "cylinder_flow",
    trajectories: dict[str, int | None] | None = None,
    time_stride: int = 1,
    max_steps: int | None = None,
    quality: bool = True,
    provenance_files: dict[str, dict[str, Any]] | None = None,
    progress: ProgressFn | None = None,
    hub_id: str | None = None,
) -> Any:
    """Convert local MeshGraphNets ``.tfrecord`` files (``files[split]``, split in
    ``train`` / ``val`` / ``test``; ``valid`` accepted) into a ``frontier-dataset`` v1.1 at
    ``out``. ``trajectories[split]`` caps how many records are taken per split."""
    from frontier_data.dataset import DatasetWriter, Provenance, RawFileRecord

    writer = DatasetWriter(
        out,
        name=f"meshgraphnets-{name.replace('_', '-')}",
        source={
            "kind": "meshgraphnets",
            "dataset": name,
            "url": f"{BASE_URL}/{name}",
            "simulator": meta.get("simulator"),
            "time_stride": time_stride,
            "max_steps": max_steps,
        },
        domain="cfd" if name in ("cylinder_flow", "airfoil") else "structural",
        license=LICENSE,
        citation=CITATION,
        hub_id=hub_id or f"meshgraphnets-{name.replace('_', '-')}",
        time_units="s",
    )
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for src_split, path in files.items():
        split = "val" if src_split == "valid" else src_split
        cap = (trajectories or {}).get(split)
        rec = (provenance_files or {}).get(str(path))
        prov = Provenance(
            source=f"{BASE_URL}/{name}",
            license=LICENSE,
            citation=CITATION,
            files=[RawFileRecord(role="tfrecord", **rec)] if rec else [],
        )
        for i, traj in enumerate(read_tfrecord_trajectories(path, meta, cap)):
            kwargs = trajectory_to_sample(traj, meta, time_stride=time_stride, max_steps=max_steps)
            params: dict[str, float] = {}
            if name == "cylinder_flow":
                params = cylinder_parameters(
                    kwargs["points"], kwargs["static"]["node_type"], kwargs["fields"]["velocity"][0]
                )
            sid = f"{split}_{i:04d}"
            writer.write_mesh_sample(
                sid,
                parameters=params,
                meta={
                    "solver": str(meta.get("simulator") or "unknown"),
                    "physics": "incompressible-ns-2d" if name == "cylinder_flow" else name,
                    "fidelity": "laminar-dns" if name == "cylinder_flow" else None,
                    "provenance": prov,
                    "extra": {
                        "source_file": Path(path).name,
                        "trajectory_index": i,
                        "time_stride": time_stride,
                        "source_dt": meta.get("dt"),
                    },
                },
                **kwargs,
            )
            splits[split].append(sid)
            if progress:
                progress(f"converted {sid}", i + 1, cap or i + 1)
    strategy = {"name": "source", "note": "MeshGraphNets train/valid/test files"}
    return finish_dataset(writer, splits, strategy=strategy, quality=quality, progress=progress)


def ingest_meshgraphnets(
    name: str,
    out: str | Path,
    *,
    trajectories: int | dict[str, int] = 12,
    raw_dir: str | Path | None = None,
    time_stride: int = 1,
    max_steps: int | None = None,
    download: bool = True,
    base_url: str = BASE_URL,
    quality: bool = True,
    progress: ProgressFn | None = None,
) -> Any:
    """Stream the first trajectories of each split of a MeshGraphNets dataset and convert
    them. ``trajectories`` is a total (split 80/10/10 over train/valid/test) or
    ``{"train": n, "val": n, "test": n}``. Only the bytes of the requested records are
    downloaded (``cylinder_flow``: ~13.6 MB per trajectory); the prefixes are kept in
    ``raw_dir`` and extended on later calls."""
    raw = Path(raw_dir) if raw_dir else Path("datasets/raw/meshgraphnets") / name
    if isinstance(trajectories, int):
        counts = dict(zip(("train", "val", "test"), split_counts(trajectories), strict=True))
    else:
        counts = {k: int(v) for k, v in trajectories.items()}
    meta = (
        fetch_meta(name, raw, base_url) if download else json.loads((raw / "meta.json").read_text())
    )
    files: dict[str, str | Path] = {}
    prov: dict[str, dict[str, Any]] = {}
    for split, n in counts.items():
        if n <= 0:
            continue
        remote = f"{SPLIT_FILES[split]}.tfrecord"
        dest = raw / remote
        url = f"{base_url}/{name}/{remote}"
        if download:
            log.info("meshgraphnets: fetching the first %d records of %s", n, url)
            have, nbytes = fetch_prefix(url, dest, n)
        else:
            have, nbytes = count_complete_records(dest)
        if have < n:
            log.warning("meshgraphnets: %s has only %d of %d records", remote, have, n)
        h = hashlib.sha256()
        with dest.open("rb") as f:
            left = nbytes
            while left > 0:
                block = f.read(min(1 << 20, left))
                if not block:
                    break
                h.update(block)
                left -= len(block)
        prov[str(dest)] = {
            "repo_path": f"{name}/{remote}[0:{nbytes}]",
            "url": url,
            "sha256": h.hexdigest(),
            "size": nbytes,
        }
        files[split] = dest
        if progress:
            progress(f"fetched {remote} ({nbytes / 1e6:.0f} MB)", have, n)
    return convert_meshgraphnets(
        files,
        meta,
        out,
        name=name,
        trajectories={k: v for k, v in counts.items()},
        time_stride=time_stride,
        max_steps=max_steps,
        quality=quality,
        provenance_files=prov,
        progress=progress,
    )


def write_tfrecord(path: str | Path, trajectories: Sequence[dict[str, np.ndarray]]) -> None:
    """Write trajectories as a MeshGraphNets-style TFRecord (tests and fixtures; CRC fields are
    zero, which readers that verify checksums will reject)."""

    def varint(n: int) -> bytes:
        out = bytearray()
        while True:
            b = n & 0x7F
            n >>= 7
            out.append(b | (0x80 if n else 0))
            if not n:
                return bytes(out)

    def ld(num: int, payload: bytes) -> bytes:
        return varint((num << 3) | 2) + varint(len(payload)) + payload

    with Path(path).open("wb") as f:
        for traj in trajectories:
            entries = b""
            for key, arr in traj.items():
                raw = np.ascontiguousarray(arr).tobytes()
                feature = ld(1, ld(1, raw))  # Feature.bytes_list.value
                entries += ld(1, ld(1, key.encode()) + ld(2, feature))
            payload = ld(1, entries)
            f.write(struct.pack("<Q", len(payload)) + b"\0\0\0\0" + payload + b"\0\0\0\0")
