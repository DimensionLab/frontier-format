# SPDX-License-Identifier: Apache-2.0
"""Train/val/test split strategies.

* ``random_split``: seeded shuffle with fractions.
* ``parameter_holdout_split``: hold out the top (or bottom) ``holdout_fraction`` of a geometry
  parameter as the test set so an evaluator can measure extrapolation. The validation set is
  drawn from the in-range population.
* ``group_split``: seeded split of whole groups (trajectory ids), so the samples cut from one
  trajectory never straddle two splits.

``find_leakage`` reports sample ids shared by two splits; ``find_trajectory_leakage`` reports
trajectories that straddle splits, from the ``group`` ids and from content evidence
(:class:`TrajectoryFingerprint`: a shared frame, an identical initial condition, or one sample
continuing where another one stops).

Every strategy returns the split lists plus a ``strategy`` dict that is stored in
``manifest.split_strategy`` so the choice is reproducible and visible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: two frames closer than this (relative L2 distance on the probe points) count as the same frame
SAME_FRAME_RTOL = 1e-6
#: sample B continues sample A when ``|B[0] - A[-1]|`` is at most this many typical step changes
DIVERGED_REL = 0.05  # same first frame but last frames this far apart: a shared-IC ensemble
CONTINUATION_FACTOR = 2.0  # first frame of b within this many typical step changes of a's last
#: ... and never further than this relative distance (unrelated frames are about 1.4 apart)
CONTINUATION_MAX_REL = 0.1  # ... and this close in absolute terms: when frames are further
# apart than 10 % (coarsely saved chaotic data) any two states of one regime look like neighbours
#: above this many fingerprints with the same layout only exact (hash) matches are searched
MAX_PAIRWISE_FINGERPRINTS = 2000
#: at most this many content leaks are listed in the result
MAX_LISTED_LEAKS = 50


@dataclass
class SplitResult:
    splits: dict[str, list[str]]
    strategy: dict[str, Any] = field(default_factory=dict)

    def as_family(self) -> dict[str, Any]:
        return {"splits": self.splits, "strategy": self.strategy}


def _check_fractions(fractions: tuple[float, float, float]) -> None:
    if any(f < 0 for f in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
        raise ValueError(f"fractions must be >= 0 and sum to 1, got {fractions}")


def _counts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    n_val = int(round(n * fractions[1]))
    n_test = int(round(n * fractions[2]))
    # Small datasets still need something to validate and test on: guarantee one sample
    # each once there are at least three samples, so evaluation never runs on an empty split.
    if n >= 3:
        n_val = max(n_val, 1) if fractions[1] > 0 else n_val
        n_test = max(n_test, 1) if fractions[2] > 0 else n_test
    n_train = n - n_val - n_test
    if n_train < 0:
        n_train = 0
    return n_train, n_val, n_test


def random_split(
    ids: Iterable[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> SplitResult:
    _check_fractions(fractions)
    ids = list(ids)
    rng = np.random.default_rng(seed)
    order = [ids[i] for i in rng.permutation(len(ids))]
    n_train, n_val, n_test = _counts(len(ids), fractions)
    splits = {
        "train": sorted(order[:n_train], key=_natural),
        "val": sorted(order[n_train : n_train + n_val], key=_natural),
        "test": sorted(order[n_train + n_val :], key=_natural),
    }
    return SplitResult(
        splits,
        {"name": "random", "fractions": list(fractions), "seed": seed, "n": len(ids)},
    )


def parameter_holdout_split(
    ids: Iterable[str],
    params: Mapping[str, Mapping[str, float]],
    parameter: str,
    *,
    holdout_fraction: float = 0.15,
    side: str = "top",
    val_fraction: float = 0.1,
    seed: int = 0,
) -> SplitResult:
    """Hold out the extreme ``holdout_fraction`` of ``parameter`` as the test split."""
    if side not in ("top", "bottom"):
        raise ValueError("side must be 'top' or 'bottom'")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    ids = list(ids)
    values: list[tuple[str, float]] = []
    missing: list[str] = []
    for i in ids:
        v = params.get(i, {}).get(parameter)
        if v is None or not math.isfinite(float(v)):
            missing.append(i)
        else:
            values.append((i, float(v)))
    if not values:
        raise ValueError(f"no sample has geometry parameter {parameter!r}")
    values.sort(key=lambda t: (t[1], _natural(t[0])), reverse=(side == "top"))
    n_test = max(1, int(round(len(values) * holdout_fraction)))
    test = [i for i, _ in values[:n_test]]
    rest = [i for i, _ in values[n_test:]]
    threshold = values[n_test - 1][1]
    rng = np.random.default_rng(seed)
    rest = [rest[i] for i in rng.permutation(len(rest))]
    n_val = int(round(len(rest) * val_fraction))
    if val_fraction > 0 and len(rest) >= 2:
        n_val = max(n_val, 1)
    val = rest[:n_val]
    train = rest[n_val:] + missing  # samples without the parameter can only be trained on
    splits = {
        "train": sorted(train, key=_natural),
        "val": sorted(val, key=_natural),
        "test": sorted(test, key=_natural),
    }
    rng_vals = [v for _, v in values]
    return SplitResult(
        splits,
        {
            "name": "parameter_holdout",
            "parameter": parameter,
            "side": side,
            "holdout_fraction": holdout_fraction,
            "threshold": threshold,
            "test_range": [
                min(v for i, v in values if i in set(test)),
                max(v for i, v in values if i in set(test)),
            ],
            "full_range": [min(rng_vals), max(rng_vals)],
            "val_fraction": val_fraction,
            "seed": seed,
            "n": len(ids),
            "missing_parameter": missing,
        },
    )


def group_split(
    ids: Iterable[str],
    groups: Mapping[str, str | None],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> SplitResult:
    """Split whole groups: every sample of a group (trajectory id) lands in the same split.

    ``groups`` maps sample id -> group; samples without a group (missing or ``None``) form a
    group of their own. ``fractions`` apply to the number of *groups* (the sample fractions
    match when the groups have equal sizes, the usual case of a trajectory cut into windows);
    with at least three groups the validation and test splits are never empty.
    """
    _check_fractions(fractions)
    ids = list(ids)
    members: dict[str, list[str]] = {}
    for i in ids:
        g = groups.get(i)
        members.setdefault(f"group:{g}" if g is not None else f"sample:{i}", []).append(i)
    keys = sorted(members, key=_natural)
    rng = np.random.default_rng(seed)
    order = [keys[i] for i in rng.permutation(len(keys))]
    n_train, n_val, _ = _counts(len(keys), fractions)
    parts = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    splits = {
        name: sorted((i for k in ks for i in members[k]), key=_natural)
        for name, ks in parts.items()
    }
    return SplitResult(
        splits,
        {
            "name": "group",
            "fractions": list(fractions),
            "seed": seed,
            "n": len(ids),
            "n_groups": len(keys),
            "groups_per_split": {name: len(ks) for name, ks in parts.items()},
        },
    )


def make_split(
    strategy: str,
    ids: Iterable[str],
    params: Mapping[str, Mapping[str, float]] | None = None,
    **kwargs: Any,
) -> SplitResult:
    if strategy == "random":
        return random_split(ids, **kwargs)
    if strategy == "group":
        groups = kwargs.pop("groups", None)
        if groups is None:
            raise ValueError("group split needs groups={sample id: group}")
        return group_split(ids, groups, **kwargs)
    if strategy == "parameter_holdout":
        if params is None:
            raise ValueError("parameter_holdout needs geometry params")
        return parameter_holdout_split(ids, params, **kwargs)
    raise ValueError(f"unknown split strategy {strategy!r}")


def find_leakage(splits: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    """Return ``{"a&b": [ids...]}`` for every pair of splits sharing sample ids."""
    names = list(splits)
    sets = {k: set(splits[k]) for k in names}
    out: dict[str, list[str]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = sorted(sets[a] & sets[b], key=_natural)
            if common:
                out[f"{a}&{b}"] = common
    return out


@dataclass
class TrajectoryFingerprint:
    """Cheap content summary of a trajectory, enough to tell that two samples are pieces of
    the same one. ``first`` / ``last`` are the values of the first time-dependent field at a
    fixed set of probe points (the same points for every sample with the same ``key``) at the
    first and last step; the hashes cover the whole frame."""

    key: str  # field name and per-step shape: only samples with the same key are compared
    first_hash: str
    last_hash: str
    first: np.ndarray
    last: np.ndarray
    step_change: float = 0.0  # median relative change between consecutive steps
    trivial_first: bool = False  # a constant frame (e.g. fluid at rest) is no evidence
    trivial_last: bool = False


def _same_parameters(
    a: Mapping[str, float] | None, b: Mapping[str, float] | None, ndigits: int = 9
) -> bool:
    def norm(p: Mapping[str, float] | None) -> dict[str, float]:
        return {k: float(f"{float(v):.{ndigits}g}") for k, v in (p or {}).items()}

    return norm(a) == norm(b)


def _rel_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``|a_i - b_j| / max(|a_i|, |b_j|)`` for every pair of rows."""
    na = (a * a).sum(axis=1)
    nb = (b * b).sum(axis=1)
    d2 = np.maximum(na[:, None] + nb[None, :] - 2.0 * (a @ b.T), 0.0)
    scale = np.maximum(np.maximum(na[:, None], nb[None, :]), 1e-300)
    return np.sqrt(d2 / scale)


def find_trajectory_leakage(
    splits: Mapping[str, Iterable[str]],
    groups: Mapping[str, str | None] | None = None,
    fingerprints: Mapping[str, TrajectoryFingerprint] | None = None,
    parameters: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Find trajectories that straddle splits.

    * ``group_leaks``: ``{group: {split: [ids]}}`` for every group present in two or more
      splits (always ``fail``).
    * ``content_leaks``: pairs of samples in different splits whose content says they belong
      together. ``shared_frame``: the last frame of ``a`` is the first frame of ``b``
      (``fail``). ``continuation``: the first frame of ``b`` is within
      :data:`CONTINUATION_FACTOR` typical step changes of the last frame of ``a`` (and within
      :data:`CONTINUATION_MAX_REL`).
      ``identical_initial_condition``: both start from the same frame. The last two ``fail``
      when the two samples also have the same ``parameters`` (both empty counts as the same)
      and ``warn`` otherwise: a parameter sweep that starts every run from one initial state
      is a legitimate dataset. Runs with the same first frame whose last frames differ by more
      than :data:`DIVERGED_REL` also only ``warn`` (``diverged``): an ensemble started from one
      state (The Well's turbulent_radiative_layer_2D does this) rather than a duplicated run.
      Constant frames (fluid at rest) are never evidence.

    Returns ``{"status", "group_leaks", "content_leaks", "n_content_leaks", "n_groups",
    "n_fingerprints", "content_mode"}``.
    """
    split_of: dict[str, str] = {}
    for name, members in splits.items():
        for i in members:
            split_of.setdefault(i, name)  # ids in two splits are find_leakage's business

    by_group: dict[str, dict[str, list[str]]] = {}
    for i, name in split_of.items():
        g = (groups or {}).get(i)
        if g is not None:
            by_group.setdefault(str(g), {}).setdefault(name, []).append(i)
    group_leaks = {
        g: {name: sorted(v, key=_natural) for name, v in parts.items()}
        for g, parts in sorted(by_group.items())
        if len(parts) > 1
    }

    leaks: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add(kind: str, a: str, b: str, rel: float) -> None:
        if a == b or split_of[a] == split_of[b]:
            return
        if kind == "identical_initial_condition" and _natural(b) < _natural(a):
            a, b = b, a
        same = _same_parameters((parameters or {}).get(a), (parameters or {}).get(b))
        severity = "fail" if kind == "shared_frame" or same else "warn"
        entry: dict[str, Any] = {
            "kind": kind,
            "a": a,
            "b": b,
            "split_a": split_of[a],
            "split_b": split_of[b],
            "rel_distance": float(rel),
            "same_parameters": same,
        }
        if kind == "identical_initial_condition" and a in fps and b in fps:
            # Runs that share step 0 but end far apart are an ensemble started from one state
            # (chaotic divergence, different seeds), not one trajectory stored twice.
            la = np.asarray(fps[a].last, dtype=np.float64).reshape(-1)
            lb = np.asarray(fps[b].last, dtype=np.float64).reshape(-1)
            if la.shape == lb.shape and la.size:
                end = float(
                    np.linalg.norm(la - lb) / max(np.linalg.norm(la), np.linalg.norm(lb), 1e-300)
                )
                entry["rel_distance_last"] = end
                if end > DIVERGED_REL:
                    entry["diverged"] = True
                    severity = "warn"
        entry["severity"] = severity
        leaks.setdefault((kind, a, b), entry)

    fps = {i: f for i, f in (fingerprints or {}).items() if i in split_of}
    by_key: dict[str, list[str]] = {}
    for i, f in fps.items():
        by_key.setdefault(f.key, []).append(i)
    mode = "none" if not fps else "pairwise"
    for ids in by_key.values():
        # exact matches first: O(n) and independent of the probe points
        first_at: dict[str, list[str]] = {}
        for i in ids:
            if not fps[i].trivial_first:
                first_at.setdefault(fps[i].first_hash, []).append(i)
        for same_ic in first_at.values():
            for k, a in enumerate(same_ic):
                for b in same_ic[k + 1 :]:
                    add("identical_initial_condition", a, b, 0.0)
        for a in ids:
            if not fps[a].trivial_last:
                for b in first_at.get(fps[a].last_hash, []):
                    add("shared_frame", a, b, 0.0)
        if len(ids) > MAX_PAIRWISE_FINGERPRINTS:
            mode = "hash"
            continue
        first = np.stack([np.asarray(fps[i].first, dtype=np.float64).reshape(-1) for i in ids])
        last = np.stack([np.asarray(fps[i].last, dtype=np.float64).reshape(-1) for i in ids])
        step = np.array([fps[i].step_change for i in ids], dtype=np.float64)
        step = np.where(np.isfinite(step), step, 0.0)
        ok_first = np.array([not fps[i].trivial_first for i in ids])
        ok_last = np.array([not fps[i].trivial_last for i in ids])
        # the matrix form loses precision below ~1e-4, so candidates are re-measured exactly
        reach = np.minimum(
            CONTINUATION_FACTOR * np.maximum(step[:, None], step[None, :]), CONTINUATION_MAX_REL
        )
        tol = np.maximum(reach, 1e-3)
        lf = _rel_distances(last, first)
        for ia, ib in zip(
            *np.nonzero((lf <= tol) & ok_last[:, None] & ok_first[None, :]), strict=True
        ):
            if ia == ib:
                continue
            rel = float(
                np.linalg.norm(last[ia] - first[ib])
                / max(np.linalg.norm(last[ia]), np.linalg.norm(first[ib]), 1e-300)
            )
            if rel <= SAME_FRAME_RTOL:
                add("shared_frame", ids[ia], ids[ib], rel)
            elif rel <= reach[ia, ib]:
                add("continuation", ids[ia], ids[ib], rel)
        ff = _rel_distances(first, first)
        for ia, ib in zip(
            *np.nonzero((ff <= 1e-3) & ok_first[:, None] & ok_first[None, :]), strict=True
        ):
            if ia >= ib:
                continue
            rel = float(
                np.linalg.norm(first[ia] - first[ib])
                / max(np.linalg.norm(first[ia]), np.linalg.norm(first[ib]), 1e-300)
            )
            if rel <= SAME_FRAME_RTOL:
                add("identical_initial_condition", ids[ia], ids[ib], rel)

    # a pair that shares a frame is reported once, not also as a continuation
    shared = {(a, b) for kind, a, b in leaks if kind == "shared_frame"}
    content = [
        v for (kind, a, b), v in leaks.items() if kind != "continuation" or (a, b) not in shared
    ]
    content.sort(key=lambda v: (v["severity"] != "fail", _natural(v["a"]), _natural(v["b"])))
    status = "pass"
    if any(v["severity"] == "warn" for v in content):
        status = "warn"
    if group_leaks or any(v["severity"] == "fail" for v in content):
        status = "fail"
    return {
        "status": status,
        "group_leaks": group_leaks,
        "content_leaks": content[:MAX_LISTED_LEAKS],
        "n_content_leaks": len(content),
        "n_groups": len(by_group),
        "n_fingerprints": len(fps),
        "content_mode": mode,
    }


def _natural(s: str) -> tuple[Any, ...]:
    import re

    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))
