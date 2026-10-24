# SPDX-License-Identifier: Apache-2.0
"""Ordered, bounded training-field statistics with complete-row validity."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import numpy as np

from arrayroute.ordered import OrderedSelection, read_flat
from arrayroute.plan import Selection
from arrayroute.reader import ArrayReader, ReadStats
from arrayroute.statistics import CanonicalChannelMoments

ALGORITHM = "training-field-canonical-4096-v2"


def fit_field(
    reader: ArrayReader,
    selections: Iterable[Selection | None],
    *,
    workspace_bytes: int,
    stats: ReadStats,
) -> dict:
    """Complete finite rows participate; missing components exclude the entire row.

    Canonical leaves span selections. Source buffers/caches are separate from numeric
    scratch. Callers pin source revisions, row order, dtype and runtime for exact replay.
    """
    channels = reader.backend.spec.shape[-1]
    dtype = reader.backend.dtype
    if channels < 1 or dtype.kind not in "fiu" or dtype.itemsize > 8:
        raise ValueError("Normalization requires real channel-last fields of at most 64 bits")
    minimum = channels * (4096 * 128 + 256)
    if workspace_bytes < minimum:
        raise MemoryError(f"Normalization requires at least {minimum} statistics workspace bytes")
    rows = min(
        65536,
        max(1, (workspace_bytes - minimum) // (channels * 96)),
        reader.buffer_bytes // (2 * channels * dtype.itemsize),
    )
    if rows < 1:
        raise MemoryError("Read allowance cannot hold one normalization row")
    moments = CanonicalChannelMoments(channels)
    lower, upper = np.full(channels, np.inf), np.full(channels, -np.inf)
    digest = hashlib.sha256()
    rows_seen = 0
    for selection in selections:
        source = OrderedSelection.create(reader, selection)
        if not source.shape or source.shape[-1] != channels:
            raise ValueError("Normalization selections must retain the complete channel axis")
        count = math.prod(source.shape[:-1])
        for start in range(0, count, rows):
            block = read_flat(
                source, start * channels, min(count, start + rows) * channels, stats, None
            ).reshape(-1, channels)
            digest.update(memoryview(block).cast("B"))
            valid = np.isfinite(block).all(axis=1)
            if valid.all():
                moments.update(block)
                values = block
            else:
                moments.update(block, np.broadcast_to(valid[:, None], block.shape))
                values = block[valid]
            if len(values):
                lower = np.minimum(lower, values.min(axis=0))
                upper = np.maximum(upper, values.max(axis=0))
            rows_seen += len(block)
    summary = moments.snapshot()
    n = int(summary.count[0])
    return {
        "n": n,
        "mean": summary.mean.tolist(),
        "m2": summary.m2.tolist(),
        "min": lower.tolist() if n else [],
        "max": upper.tolist() if n else [],
        "rows_seen": rows_seen,
        "sha256": digest.hexdigest(),
        "algorithm": ALGORITHM,
        "dtype": dtype.str,
        "shape": list(reader.backend.spec.shape),
    }
