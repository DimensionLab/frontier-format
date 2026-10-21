# SPDX-License-Identifier: Apache-2.0
"""Named scientific selections backed by independent array engines."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from arrayroute import ArrayReader, ReadPlan, ReadStats, Tile
from arrayroute.quality import ErrorAccumulator, ErrorAssessment
from arrayroute.session import LocalReadSession

if TYPE_CHECKING:
    from arrayroute.retention import RetentionPin
    from arrayroute.snapshot import SnapshotReadSession
    from frontier_data.arrays import Sample


@dataclass(frozen=True)
class FieldResult:
    values: np.ndarray
    dimensions: tuple[str, ...]
    units: str
    association: str
    snapshot_id: str | None = None


@dataclass
class ArrayFieldReader:
    """Named scientific view over any ArrayRoute backend, including remote connectors.

    The connector supplies dimensions, units and association from its scientific catalog.
    This wrapper preserves that metadata; it does not infer units or geometry identity.
    """

    reader: ArrayReader
    dimensions: tuple[str, ...]
    units: str
    association: str
    snapshot_id: str | None = None
    retention_tag: str | None = None

    def __post_init__(self) -> None:
        if (
            len(self.dimensions) != len(self.reader.backend.spec.shape)
            or len(set(self.dimensions)) != len(self.dimensions)
            or not all(isinstance(d, str) and d.strip() for d in self.dimensions)
        ):
            raise ValueError("Field dimensions must be unique and match the array rank")
        if not self.units.strip() or not self.association.strip():
            raise ValueError("Units and association must be declared; use '-' for unknown units")

    def selection(self, selectors: Mapping[str, int | slice]) -> tuple[int | slice, ...]:
        unknown = selectors.keys() - set(self.dimensions)
        if unknown:
            raise ValueError(f"Unknown dimensions {sorted(unknown)}; choose {self.dimensions}")
        return tuple(selectors.get(dim, slice(None)) for dim in self.dimensions)

    def plan(self, selectors: Mapping[str, int | slice]) -> ReadPlan:
        return self.reader.plan(self.selection(selectors))

    def read(
        self,
        selectors: Mapping[str, int | slice],
        *,
        stats: ReadStats | None = None,
    ) -> FieldResult:
        selection = self.selection(selectors)
        return FieldResult(
            self.reader.read(selection, stats=stats),
            tuple(
                d for d, s in zip(self.dimensions, selection, strict=True) if isinstance(s, slice)
            ),
            self.units,
            self.association,
            self.snapshot_id,
        )

    def stream(
        self,
        selectors: Mapping[str, int | slice],
        *,
        stats: ReadStats | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[Tile, FieldResult]]:
        """Yield disjoint tiles in the selected region, retaining scientific metadata."""
        selection = self.selection(selectors)
        dimensions = tuple(
            d for d, s in zip(self.dimensions, selection, strict=True) if isinstance(s, slice)
        )
        for tile, values in self.reader.stream(selection, stats=stats, cancelled=cancelled):
            yield (
                tile,
                FieldResult(values, dimensions, self.units, self.association, self.snapshot_id),
            )
            del values


class SnapshotFieldReader(ArrayFieldReader):
    """Read field data and scientific metadata from the same committed revision.

    The array declares ``dims``, ``units`` and ``association`` attributes. Missing
    scientific metadata is an error, never filled from another dataset/revision.
    """

    def __init__(
        self,
        session: SnapshotReadSession,
        array_path: str,
        *,
        retention: RetentionPin | None = None,
    ):
        if retention is not None:
            from arrayroute.retention import RetentionManager

            if retention.snapshot != session.snapshot:
                raise ValueError("Retention reference belongs to another field revision")
            RetentionManager(session.snapshot.location).verify(retention)
        reader = session.open(array_path)
        attributes = session.attributes(array_path)
        dimensions = attributes.get("dims")
        units, association = attributes.get("units"), attributes.get("association")
        if not isinstance(dimensions, list) or not all(isinstance(d, str) for d in dimensions):
            raise ValueError("Snapshot field must declare a dims list")
        if not isinstance(units, str) or not isinstance(association, str):
            raise ValueError("Snapshot field must declare units and association")
        self.session = session
        super().__init__(
            reader,
            tuple(dimensions),
            units,
            association,
            session.snapshot.snapshot_id,
            retention.tag if retention is not None else None,
        )


class FieldReader(ArrayFieldReader):
    """Read a locally registered sample using its stored scientific metadata."""

    def __init__(
        self,
        sample: Sample,
        name: str,
        *,
        location: str | None = None,
        lod: int = 0,
        engine: str | None = None,
        buffer_bytes: int = 128 << 20,
        session: LocalReadSession | None = None,
    ):
        location = location or sample.native_location(name)
        array = sample.array(name, location, lod)
        info = next(
            (
                f
                for f in sample.meta.fields
                if f.name == name
                and (
                    location is None
                    or f.location
                    == {"fields": "node", "cell_fields": "cell"}.get(location, location)
                )
            ),
            None,
        )
        if session is not None and engine is not None and session.engine != engine:
            raise ValueError("Requested engine differs from the shared read session")
        self.session = session or LocalReadSession(
            engine or "zarr-python", buffer_bytes=buffer_bytes
        )
        super().__init__(
            self.session.open(sample.store_path, array.path),
            tuple(sample.dims(name, location, lod)),
            info.units if info else "-",
            sample.location if location == "grid" else location,
        )


def parse_selectors(items: list[str]) -> dict[str, int | slice]:
    selectors = {}
    for item in items:
        name, value = item.split("=", 1)
        if not name or name in selectors:
            raise ValueError(f"Empty or duplicate dimension in {item!r}")
        parts = value.split(":")
        if len(parts) == 1:
            selectors[name] = int(value)
        elif len(parts) in (2, 3):
            selectors[name] = slice(*(int(p) if p else None for p in parts))
        else:
            raise ValueError(f"Expected dimension=index or dimension=start:stop[:stride]: {item}")
    return selectors


def compare_fields(
    reference: ArrayFieldReader,
    candidate: ArrayFieldReader,
    selectors: Mapping[str, int | slice],
    *,
    absolute_tolerance: float,
    cancelled: Callable[[], bool] | None = None,
) -> ErrorAssessment:
    """Compare explicitly paired fields in the same geometry/index space, through bounded tiles.

    Units, axes, association, shape and dtype must match. This does not establish mesh
    correspondence or remap geometry. Two reader budgets plus comparison scratch apply.
    """
    for name in ("dimensions", "units", "association"):
        if getattr(reference, name) != getattr(candidate, name):
            raise ValueError(f"Field {name} differ; convert or align fields before comparison")
    if reference.reader.backend.spec.shape != candidate.reader.backend.spec.shape:
        raise ValueError("Field shapes differ; align fields before comparison")
    if reference.reader.backend.dtype != candidate.reader.backend.dtype:
        raise ValueError("Field dtypes differ; convert fields before comparison")
    accumulator = ErrorAccumulator(absolute_tolerance=absolute_tolerance)
    selection = reference.selection(selectors)
    for outer, expected in reference.reader.stream(selection, cancelled=cancelled):
        for inner, actual in candidate.reader.stream(outer.plan.selection, cancelled=cancelled):
            accumulator.update(expected[inner.destination], actual)
            del actual
        del expected
    if not accumulator.result.finite_values and not accumulator.result.nonfinite_mismatches:
        raise ValueError("Selected region contains no comparable finite field values")
    return accumulator.result
