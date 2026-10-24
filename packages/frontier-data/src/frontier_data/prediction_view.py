# SPDX-License-Identifier: Apache-2.0
"""Selected reads from immutable prediction Zarr stores, through ArrayRoute."""

from __future__ import annotations

import json
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from arrayroute.plan import Selection, Tile
from arrayroute.reader import ArrayReader, ReadStats
from arrayroute.session import LocalReadSession

Role = Literal["pred", "true", "err", "std", "valid"]


class PredictionView:
    """Read fields/regions without materializing a rollout or opening its NPZ copy.

    The caller must keep the store and sidecar fixed for this object's lifetime. Read
    budgets cover output and planned decoded tiles, not backend caches, codec scratch,
    metadata or arrays the caller retains across calls. Sessions are worker-local.
    """

    def __init__(
        self, root: str | Path, sample_id: str, *,
        session: LocalReadSession | None = None, metadata_bytes: int = 1 << 20,
    ):
        import zarr

        if (not sample_id or sample_id in {".", ".."}
                or any(c in sample_id for c in "/\\\x00")):
            raise ValueError("Prediction sample id must be a single safe path component")
        if metadata_bytes < 1:
            raise ValueError("metadata_bytes must be positive")
        self.id = sample_id
        self.path = Path(root) / f"{sample_id}.zarr"
        self.session = session if session is not None else LocalReadSession()
        self.stats = ReadStats()
        self.group: Any = zarr.open_group(str(self.path), mode="r")
        self.info: dict[str, Any] = {}
        sidecar = Path(root) / f"{sample_id}.json"
        if sidecar.exists():
            with sidecar.open("rb") as stream:
                raw = stream.read(metadata_bytes + 1)
            if len(raw) > metadata_bytes:
                raise MemoryError("Prediction metadata exceeds allowance")
            info = json.loads(raw)
            if not isinstance(info, dict):
                raise ValueError("Prediction metadata must be an object")
            self.info = info
        self.kind = str(self.group.attrs.get("sample_kind", ""))
        self.time_units = str(self.group.attrs.get("time_units", ""))
        times = self.session.open(self.path, "times")
        if len(times.backend.spec.shape) != 1 or times.backend.dtype.kind != "f":
            raise ValueError("Prediction times must be a floating-point vector")
        self.n_steps = times.backend.spec.shape[0]
        self.field_names = tuple(sorted(
            name[:-5] for name in self.group["fields"].array_keys() if name.endswith("_pred")
        ))
        if not self.field_names:
            raise ValueError("Prediction store has no predicted fields")
        for name in self.field_names:
            p, t = self.reader(name, "pred"), self.reader(name, "true")
            if (p.backend.spec.shape != t.backend.spec.shape
                    or p.backend.spec.shape[0] != self.n_steps
                    or len(p.backend.spec.shape) < 3):
                raise ValueError(f"{name}: prediction, reference and time shapes disagree")
            if self.units(name, "pred") != self.units(name, "true"):
                raise ValueError(f"{name}: prediction and reference units disagree")

    def reader(self, field: str, role: Role = "pred") -> ArrayReader:
        if field not in self.field_names or role not in {"pred", "true", "err", "std", "valid"}:
            raise ValueError("Unknown prediction field or role")
        return self.session.open(self.path, f"fields/{field}_{role}")

    def units(self, field: str, role: Role = "pred") -> str | None:
        self.reader(field, role)
        return self.group[f"fields/{field}_{role}"].attrs.get("units")

    def read(
        self, field: str, role: Role = "pred", *, selection: Selection | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        reader = self.reader(field, role)
        selected = (tuple(slice(None) for _ in reader.backend.spec.shape)
                    if selection is None else selection)
        return self._read(reader, selected, cancelled)

    def _read(
        self, reader: ArrayReader, selection: Selection,
        cancelled: Callable[[], bool] | None,
    ) -> np.ndarray:
        if cancelled is not None and cancelled():
            raise InterruptedError("Prediction read cancelled")
        # ArrayReader owns admission/tiling. Cancellation is checked around each
        # materialized selection; blocking backend reads are not interruptible.
        value = reader.read(selection, stats=self.stats)
        if cancelled is not None and cancelled():
            raise InterruptedError("Prediction read cancelled")
        return value

    def stream(
        self, field: str, role: Role = "pred", *, selection: Selection | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Generator[tuple[Tile, np.ndarray], None, None]:
        """Yield budgeted spatial/time tiles; release each value before advancing.

        Unlike read(), this permits a selection larger than the memory allowance.
        Cancellation is checked between tiles. Tile destinations index the selected
        output, so consumers can reconstruct strided selections without reordering.
        """
        reader = self.reader(field, role)
        selected = (tuple(slice(None) for _ in reader.backend.spec.shape)
                    if selection is None else selection)
        return reader.stream(selected, stats=self.stats, cancelled=cancelled)

    def time(self, index: int) -> float:
        return float(self._read(self.session.open(self.path, "times"), (index,), None))

    def frame_selection(self, field: str, index: int) -> Selection:
        return (index, *(slice(None) for _ in self.reader(field).backend.spec.shape[1:]))

    def selected_fields(self, fields: Sequence[str] | None) -> tuple[str, ...]:
        names = tuple(self.field_names if fields is None else fields)
        if not names or len(set(names)) != len(names) or any(
            name not in self.field_names for name in names
        ):
            raise ValueError("Select distinct fields present in this prediction store")
        return names

    def error_curves(
        self, field: str, *, buffer_bytes: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, np.ndarray]:
        """Score a stored rollout without reading its NPZ or materializing full fields.

        Returns dimensionless normalized errors/correlation, RMSE and sigma in field
        units, and mean-square curves in squared field units. Times remain available
        through time(). This is not the complete Frontier benchmark or ensemble UQ.
        """
        from arrayroute.error_curves import error_curves

        has_mask = f"{field}_valid" in self.group["fields"]
        if (self.info.get("validity_file") or self.info.get("validity_fields")) and not has_mask:
            raise ValueError(f"{field}: chunked validity is required to score this prediction")
        return error_curves(
            self.reader(field), self.reader(field, "true"),
            validity=self.reader(field, "valid") if has_mask else None,
            buffer_bytes=self.session.buffer_bytes if buffer_bytes is None else buffer_bytes,
            cancelled=cancelled, stats=self.stats,
        )
