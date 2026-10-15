# SPDX-License-Identifier: Apache-2.0
"""Transient prediction files shared by ``frontier_train`` (writer), ``frontier_eval`` (reader)
and the server/viewer (the ``.zarr`` stores).

A predictions directory holds, per sample (= one rollout of one trajectory):

``<sid>.npz``
    ``times (T,)`` f64, ``pred (T, ..., C)`` and ``true (T, ..., C)`` f32 with the channels of
    all fields concatenated on the last axis, ``field_names (F,)``, ``field_channels (F,)``,
    ``t_index (T,)`` (indices of the predicted steps in the source trajectory); optional
    ``ensemble (E, T, ..., C)`` (members; ``pred`` is then their mean), ``points (N, D)``,
    ``spacing (D,)``.
``<sid>.json``
    ``{"id", "kind", "dataset", "split", "parameters", "dt", "history", "train_horizon",
    "model", ...}``
``<sid>.zarr``
    a ``frontier-dataset`` v1.1 grid or mesh store (same arrays as ``grid.zarr`` /
    ``mesh.zarr``, root attribute ``sample_kind``) with fields ``<name>_pred``,
    ``<name>_true``, ``<name>_err`` (pred - true) and, for an ensemble, ``<name>_std`` under
    ``fields/`` and the rollout ``times``. :func:`pack_prediction_store` tars it as
    ``<sid>.zarr.tar``; uploaded as the run artifact ``predictions/<sid>.zarr.tar`` the server's
    time-indexed endpoints serve it (``?field=velocity_pred&t=12``).

plus one ``summary.json`` (:data:`SUMMARY_FORMAT`) listing the samples and fields.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_data.dataset import FORMAT_MINOR, FORMAT_NAME, FORMAT_VERSION, write_json

SUMMARY_FORMAT = "frontier-predictions-transient"
SUMMARY_VERSION = 1


@dataclass
class TransientPrediction:
    """One rollout: ``pred`` / ``true`` are ``{field: (T, ..., C)}``."""

    id: str
    times: np.ndarray
    pred: dict[str, np.ndarray]
    true: dict[str, np.ndarray]
    info: dict[str, Any] = field(default_factory=dict)
    ensemble: dict[str, np.ndarray] | None = None  # {field: (E, T, ..., C)}
    points: np.ndarray | None = None
    spacing: list[float] | None = None
    t_index: np.ndarray | None = None

    @property
    def n_steps(self) -> int:
        return int(len(self.times))

    @property
    def field_names(self) -> list[str]:
        return list(self.pred)

    @property
    def kind(self) -> str:
        return str(self.info.get("kind", "grid"))

    @property
    def dt(self) -> float | None:
        dt = self.info.get("dt")
        if dt is not None:
            return float(dt)
        return float(np.median(np.diff(self.times))) if len(self.times) > 1 else None


def _stack(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(v, dtype=np.float32) for v in fields.values()], axis=-1)


def _split(arr: np.ndarray, names: Sequence[str], channels: Sequence[int]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    c0 = 0
    for n, c in zip(names, channels, strict=True):
        out[str(n)] = arr[..., c0 : c0 + int(c)]
        c0 += int(c)
    return out


def write_transient_prediction(
    out_dir: str | Path,
    sample_id: str,
    *,
    times: np.ndarray,
    pred: Mapping[str, np.ndarray],
    true: Mapping[str, np.ndarray],
    kind: str = "grid",
    info: Mapping[str, Any] | None = None,
    ensemble: Mapping[str, np.ndarray] | None = None,
    t_index: np.ndarray | None = None,
    points: np.ndarray | None = None,
    cells: np.ndarray | None = None,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    boundary_conditions: Mapping[str, Any] | None = None,
    static: Mapping[str, np.ndarray] | None = None,
    write_zarr: bool = True,
    field_units: Mapping[str, str] | None = None,
) -> Path:
    """Write ``<sid>.npz`` + ``<sid>.json`` (+ ``<sid>.zarr``). ``pred[name]`` and
    ``true[name]`` are ``(T, X, Y[, Z], C)`` for ``kind="grid"`` or ``(T, N, C)`` for
    ``"mesh"`` / ``"point_cloud"`` (then ``points`` is required, ``cells`` for a mesh).
    ``field_units`` maps original field names to units; prediction, truth, signed error
    and ensemble standard deviation retain the same units. No unit conversion occurs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(times, dtype=np.float64)
    names = list(pred)
    if list(true) != names:
        raise ValueError(f"pred fields {names} != true fields {list(true)}")
    p = {k: np.asarray(v, dtype=np.float32) for k, v in pred.items()}
    t = {k: np.asarray(true[k], dtype=np.float32) for k in names}
    for k in names:
        if p[k].shape != t[k].shape or p[k].shape[0] != len(times):
            raise ValueError(
                f"field {k!r}: pred {p[k].shape} / true {t[k].shape} / {len(times)} steps"
            )
    arrays: dict[str, Any] = {
        "times": times,
        "pred": _stack(p),
        "true": _stack(t),
        "field_names": np.array(names),
        "field_channels": np.array([p[k].shape[-1] for k in names], dtype=np.int32),
        "t_index": np.asarray(
            t_index if t_index is not None else np.arange(len(times)), dtype=np.int64
        ),
    }
    if ensemble is not None:
        arrays["ensemble"] = _stack({k: ensemble[k] for k in names})
    if points is not None:
        arrays["points"] = np.asarray(points, dtype=np.float32)
    if spacing is not None:
        arrays["spacing"] = np.asarray(spacing, dtype=np.float64)
    np.savez_compressed(out_dir / f"{sample_id}.npz", **arrays)
    meta = {"id": sample_id, "kind": kind, "n_steps": int(len(times)), **dict(info or {})}
    if field_units is not None:
        meta["field_units"] = dict(field_units)
    if len(times) > 1:
        meta.setdefault("dt", float(np.median(np.diff(times))))
    write_json(out_dir / f"{sample_id}.json", meta)
    if write_zarr:
        std = (
            {k: np.asarray(ensemble[k], dtype=np.float32).std(axis=0) for k in names}
            if ensemble is not None
            else None
        )
        _write_store(
            out_dir / f"{sample_id}.zarr",
            kind,
            times,
            p,
            t,
            points,
            cells,
            spacing,
            origin,
            boundary_conditions,
            static,
            std,
            str(meta.get("time_units", "s")),
            field_units,
        )
    return out_dir / f"{sample_id}.npz"


def _write_store(
    path: Path,
    kind: str,
    times: np.ndarray,
    pred: Mapping[str, np.ndarray],
    true: Mapping[str, np.ndarray],
    points: np.ndarray | None,
    cells: np.ndarray | None,
    spacing: Sequence[float] | None,
    origin: Sequence[float] | None,
    boundary_conditions: Mapping[str, Any] | None,
    static: Mapping[str, np.ndarray] | None,
    std: Mapping[str, np.ndarray] | None = None,
    time_units: str = "s",
    field_units: Mapping[str, str] | None = None,
) -> None:
    """Reuse the dataset writers in a scratch directory, then move the store into place."""
    from frontier_data.arrays import write_grid_sample, write_mesh_sample, write_points_sample
    from frontier_data.dataset import STORE_NAMES

    fields: dict[str, np.ndarray] = {}
    for k in pred:
        fields[f"{k}_pred"] = pred[k]
        fields[f"{k}_true"] = true[k]
        fields[f"{k}_err"] = pred[k] - true[k]
        if std is not None and k in std:
            fields[f"{k}_std"] = std[k]
    units = {
        f"{name}_{suffix}": unit
        for name, unit in (field_units or {}).items()
        for suffix in ("pred", "true", "err", "std")
        if f"{name}_{suffix}" in fields
    }
    tmp = path.parent / f".{path.stem}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    if kind == "grid":
        nd = next(iter(pred.values())).ndim - 2
        write_grid_sample(
            tmp,
            "s",
            fields=fields,
            field_units=units,
            times=times,
            time_units=time_units,
            ndim=nd,
            spacing=spacing,
            origin=origin,
            boundary_conditions=boundary_conditions,
            static=static,
        )
    elif kind == "mesh" and cells is not None and points is not None:
        write_mesh_sample(
            tmp,
            "s",
            points=points,
            cells=cells,
            fields=fields,
            field_units=units,
            times=times,
            static=static,
            time_units=time_units,
        )
    elif points is not None:
        kind = "point_cloud"
        write_points_sample(
            tmp,
            "s",
            points=points,
            fields=fields,
            field_units=units,
            times=times,
            static=static,
            time_units=time_units,
        )
    else:
        raise ValueError("mesh / point_cloud predictions need `points`")
    if units:
        import zarr

        # Dataset writers put units in meta.json; prediction archives retain only the store.
        store = zarr.open_group(str(tmp / "samples" / "s" / STORE_NAMES[kind]), mode="a")
        for name, unit in units.items():
            store[f"fields/{name}"].attrs["units"] = unit
    if path.exists():
        shutil.rmtree(path)
    shutil.move(str(tmp / "samples" / "s" / STORE_NAMES[kind]), str(path))
    shutil.rmtree(tmp)


def write_predictions_summary(
    out_dir: str | Path, samples: Sequence[str], extra: Mapping[str, Any] | None = None
) -> Path:
    out_dir = Path(out_dir)
    doc = {
        "format": SUMMARY_FORMAT,
        "version": SUMMARY_VERSION,
        "dataset_format": f"{FORMAT_NAME} v{FORMAT_VERSION}.{FORMAT_MINOR}",
        "samples": list(samples),
        **dict(extra or {}),
    }
    write_json(out_dir / "summary.json", doc)
    return out_dir / "summary.json"


def read_transient_prediction(out_dir: str | Path, sample_id: str) -> TransientPrediction:
    out_dir = Path(out_dir)
    with np.load(out_dir / f"{sample_id}.npz", allow_pickle=False) as z:
        names = [str(n) for n in z["field_names"]]
        ch = [int(c) for c in z["field_channels"]]
        jpath = out_dir / f"{sample_id}.json"
        return TransientPrediction(
            id=sample_id,
            times=np.asarray(z["times"], dtype=np.float64),
            pred=_split(z["pred"], names, ch),
            true=_split(z["true"], names, ch),
            info=json.loads(jpath.read_text()) if jpath.exists() else {},
            ensemble=_split(z["ensemble"], names, ch) if "ensemble" in z else None,
            points=np.asarray(z["points"]) if "points" in z else None,
            spacing=[float(x) for x in z["spacing"]] if "spacing" in z else None,
            t_index=np.asarray(z["t_index"]) if "t_index" in z else None,
        )


def is_transient_predictions(out_dir: str | Path) -> bool:
    path = Path(out_dir) / "summary.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("format") == SUMMARY_FORMAT
    except (OSError, ValueError):
        return False


def list_transient_predictions(out_dir: str | Path) -> list[str]:
    """Sample ids in ``summary.json`` order (or every ``*.npz`` when there is no summary)."""
    out_dir = Path(out_dir)
    path = out_dir / "summary.json"
    if path.exists():
        ids = json.loads(path.read_text()).get("samples")
        if isinstance(ids, list):
            return [str(i) for i in ids]
    return sorted(p.stem for p in out_dir.glob("*.npz"))


def pack_prediction_store(out_dir: str | Path, sample_id: str) -> Path:
    """Tar ``<sid>.zarr`` into ``<sid>.zarr.tar`` (uncompressed: the chunks already are; the
    archive's top-level directory is ``<sid>.zarr``). Upload it as the run artifact
    ``predictions/<sid>.zarr.tar``."""
    import tarfile

    out_dir = Path(out_dir)
    store = out_dir / f"{sample_id}.zarr"
    if not store.is_dir():
        raise FileNotFoundError(store)
    dest = out_dir / f"{sample_id}.zarr.tar"
    with tarfile.open(dest, "w") as tar:
        tar.add(store, arcname=store.name)
    return dest
