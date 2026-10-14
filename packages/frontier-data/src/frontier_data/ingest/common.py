# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the dataset converters."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

ProgressFn = Callable[[str, int, int], None]


def safe_id(text: str) -> str:
    """Sample ids are path segments and URL parts: keep ``[A-Za-z0-9_-]`` only."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def split_counts(total: int, fractions: Sequence[float] = (0.8, 0.1, 0.1)) -> list[int]:
    """Distribute ``total`` trajectories over train/val/test, at least one in val and test
    once there are three."""
    if total <= 0:
        return [0, 0, 0]
    if total < 3:
        return [total, 0, 0]
    n_val = max(1, int(round(total * fractions[1])))
    n_test = max(1, int(round(total * fractions[2])))
    return [total - n_val - n_test, n_val, n_test]


def finish_dataset(
    writer: Any,
    splits: dict[str, list[str]],
    *,
    strategy: dict[str, Any],
    quality: bool,
    progress: ProgressFn | None = None,
) -> Any:
    """Write the manifest with ``splits`` and (optionally) the quality report; return the
    opened :class:`~frontier_data.dataset.Dataset`."""
    from frontier_data.dataset import Dataset

    writer.manifest.splits = {k: list(v) for k, v in splits.items()}
    writer.manifest.split_strategy = strategy
    writer.manifest.split_families = {
        str(strategy.get("name", "source")): {
            "splits": writer.manifest.splits,
            "strategy": strategy,
        }
    }
    writer.finalize()
    ds = Dataset(writer.root)
    if quality:
        from frontier_data.quality import run_quality

        run_quality(ds, progress=progress)
        ds = Dataset(writer.root)
    return ds


def subsample_time(n_steps: int, stride: int = 1, max_steps: int | None = None) -> np.ndarray:
    idx = np.arange(0, n_steps, max(1, stride))
    return idx[:max_steps] if max_steps else idx
