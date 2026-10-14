# SPDX-License-Identifier: Apache-2.0
"""Transient ops: subsample, trim (burn-in), align, drop duplicate frames, window average,
continuity check. Every op reads one step at a time where it can and rewrites arrays
wholesale through :class:`SampleEditor`."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import numpy as np
from pydantic import Field

from frontier_data.arrays import time_info
from frontier_data.dataset import Manifest, TimeInfo
from frontier_data.quality_transient import classify_stationarity, mser_truncation
from frontier_prep.base import BaseOp, DatasetRef, SampleOutcome
from frontier_prep.io import SampleEditor
from frontier_prep.op import Assertion, Change, OpWarning, Params, register
from frontier_prep.ops.fields import _subset_steps


def _transient_ids(ref: DatasetRef) -> list[str]:
    return [e.id for e in ref.manifest.samples if e.temporal == "transient"]


def _finish_manifest(ref: DatasetRef, manifest: Manifest, outcomes: list[SampleOutcome]) -> None:
    """Refresh n_steps per entry and manifest.time after a time op."""
    by = {o.sample_id: o for o in outcomes}
    for e in manifest.samples:
        o = by.get(e.id)
        if o and "n_steps" in o.stats:
            e.n_steps = int(o.stats["n_steps"])
    times = [(o.stats.get("dt"), o.stats.get("n_steps")) for o in outcomes if "n_steps" in o.stats]
    if times:
        dts = {round(dt, 12) for dt, _ in times if dt is not None}
        dt = next(iter(dts)) if len(dts) == 1 else None
        manifest.time = TimeInfo(
            n_steps=max(int(n) for _, n in times if n is not None),
            dt=dt,
            uniform=dt is not None,
            units=manifest.time.units if manifest.time else "s",
        )


def _time_stats(ed: SampleEditor) -> dict[str, Any]:
    s = ed.reopen()
    ti = time_info(s.times) if s.times is not None else None
    return {"n_steps": s.n_steps, "dt": ti.dt if ti else None, "t0": ti.t0 if ti else None}


# -- time.subsample ----------------------------------------------------------------------------


class SubsampleParams(Params):
    stride: int | None = Field(default=None, ge=1, description="Keep every k-th step.")
    max_steps: int | None = Field(
        default=None, ge=2, description="Keep at most this many (evenly spaced)."
    )
    target_dt: float | None = Field(
        default=None, gt=0, description="Keep steps closest to a uniform grid of this dt."
    )
    start: int = Field(default=0, ge=0, description="First step kept.")
    end: int | None = Field(default=None, description="Last step (exclusive).")


@register
class Subsample(BaseOp):
    id = "prep.time.subsample"
    version = "1.0.0"
    category = "time"
    summary = (
        "Keep a subset of time steps (stride, budget or target dt) in every time-dependent array; "
        "recorded method."
    )
    when_to_use = (
        "quality reports temporal_subsampling.recommended_stride > 1 (oversampled), or a loader "
        "budget needs fewer frames."
    )
    changes_fields = True
    changes_manifest = True
    Params = SubsampleParams

    def plan_samples(self, ref: DatasetRef, params: SubsampleParams) -> list[str]:
        return _transient_ids(ref)

    def _keep(self, times: np.ndarray, params: SubsampleParams) -> np.ndarray:
        n = len(times)
        end = n if params.end is None else min(params.end, n)
        idx = np.arange(params.start, end)
        if params.stride:
            idx = idx[:: params.stride]
        if params.max_steps and len(idx) > params.max_steps:
            idx = idx[np.unique(np.linspace(0, len(idx) - 1, params.max_steps).astype(int))]
        if params.target_dt:
            grid = np.arange(float(times[idx[0]]), float(times[idx[-1]]) + 1e-12, params.target_dt)
            pos = np.searchsorted(times[idx], grid)
            pos = np.clip(pos, 0, len(idx) - 1)
            idx = idx[np.unique(pos)]
        return idx

    def plan_sample(self, ed: SampleEditor, params: SubsampleParams) -> list[Change]:
        t = ed.sample.times
        if t is None:
            return []
        keep = self._keep(t, params)
        if len(keep) == len(t):
            return []
        return [
            Change(
                target=f"samples/{ed.id}/times",
                action="modify",
                before={"n_steps": int(len(t))},
                after={"n_steps": int(len(keep))},
            )
        ]

    def apply_sample(
        self, ed: SampleEditor, params: SubsampleParams, *, seed: int
    ) -> SampleOutcome:
        t = ed.sample.times
        if t is None:
            return SampleOutcome(sample_id=ed.id, skipped=True)
        keep = self._keep(t, params)
        if len(keep) == len(t):
            return SampleOutcome(sample_id=ed.id, skipped=True, stats=_time_stats(ed))
        _subset_steps(ed, keep)
        ed.update_meta(
            lambda m: m.extra.setdefault("time", {}).update(
                {
                    "subsample": params.model_dump(exclude_none=True),
                    "kept": int(len(keep)),
                    "of": int(len(t)),
                }
            )
        )
        return SampleOutcome(
            sample_id=ed.id,
            changes=[
                Change(
                    target=f"samples/{ed.id}/times",
                    action="modify",
                    before={"n_steps": int(len(t))},
                    after={"n_steps": int(len(keep))},
                )
            ],
            stats=_time_stats(ed),
        )

    def apply_manifest(
        self, ref: DatasetRef, manifest: Manifest, params: Any, outcomes: list[SampleOutcome]
    ) -> tuple[list[Change], dict[str, Any]]:
        _finish_manifest(ref, manifest, outcomes)
        return [Change(target="manifest/time", action="modify")], {}


# -- time.trim ----------------------------------------------------------------------------------


class TrimParams(Params):
    method: Literal["stationarity", "explicit"] = "stationarity"
    start: int = Field(default=0, description="explicit: first kept step.")
    end: int | None = None
    field: str | None = Field(
        default=None,
        description="Field whose spatial mean/RMS drives the burn-in test (default: first field).",
    )
    min_remaining: int = Field(default=16, description="Never trim below this many steps.")
    max_burn_in_frac: float = Field(default=0.5)


@register
class Trim(BaseOp):
    id = "prep.time.trim"
    version = "1.0.0"
    category = "time"
    summary = (
        "Drop the initial transient (burn-in) using the MSER truncation rule on the spatial RMS, "
        "or explicit bounds; steps removed are recorded."
    )
    when_to_use = (
        "Statistically steady targets (time averages, spectra) must not include the start-up; "
        "quality reports stationarity transient-start with recommended_burn_in."
    )
    changes_fields = True
    changes_manifest = True
    Params = TrimParams

    def plan_samples(self, ref: DatasetRef, params: TrimParams) -> list[str]:
        return _transient_ids(ref)

    def _bounds(self, ed: SampleEditor, params: TrimParams) -> tuple[int, int, dict[str, Any]]:
        s = ed.sample
        n = s.n_steps
        if params.method == "explicit":
            return params.start, n if params.end is None else min(params.end, n), {}
        name = params.field or s.field_names()[0]
        loc = "cell" if name in s.field_names("cell") else None
        rms = np.empty(n)
        for t, step in enumerate(ed.iter_steps(name, location=loc)):
            x = step.astype(np.float64)
            rms[t] = float(np.sqrt(np.mean(x * x)))
        k = int(mser_truncation(rms))
        k = min(k, int(params.max_burn_in_frac * n), max(n - params.min_remaining, 0))
        cls = classify_stationarity(rms[k:]) if n - k >= 16 else {}
        return (
            k,
            n,
            {
                "field": name,
                "burn_in": k,
                "stationarity_after": cls.get("class") if isinstance(cls, dict) else None,
            },
        )

    def plan_sample(self, ed: SampleEditor, params: TrimParams) -> list[Change]:
        if params.method == "explicit":
            a, b, _ = self._bounds(ed, params)
            if a == 0 and b == ed.sample.n_steps:
                return []
            return [
                Change(
                    target=f"samples/{ed.id}/times", action="modify", detail={"start": a, "end": b}
                )
            ]
        return [
            Change(
                target=f"samples/{ed.id}/times", action="modify", detail={"method": "stationarity"}
            )
        ]

    def apply_sample(self, ed: SampleEditor, params: TrimParams, *, seed: int) -> SampleOutcome:
        a, b, detail = self._bounds(ed, params)
        n = ed.sample.n_steps
        if a == 0 and b == n:
            return SampleOutcome(sample_id=ed.id, skipped=True, stats={**_time_stats(ed), **detail})
        _subset_steps(ed, np.arange(a, b))
        ed.update_meta(
            lambda m: m.extra.setdefault("time", {}).update(
                {"trim": {"start": a, "end": b, **detail}}
            )
        )
        return SampleOutcome(
            sample_id=ed.id,
            changes=[
                Change(
                    target=f"samples/{ed.id}/times",
                    action="modify",
                    before={"n_steps": n},
                    after={"n_steps": b - a},
                    detail=detail,
                )
            ],
            stats={**_time_stats(ed), **detail},
        )

    def apply_manifest(
        self, ref: DatasetRef, manifest: Manifest, params: Any, outcomes: list[SampleOutcome]
    ) -> tuple[list[Change], dict[str, Any]]:
        _finish_manifest(ref, manifest, outcomes)
        return [Change(target="manifest/time", action="modify")], {
            "burn_in": {o.sample_id: o.stats.get("burn_in") for o in outcomes}
        }


# -- time.align --------------------------------------------------------------------------------


class AlignParams(Params):
    mode: Literal["shift_t0", "common_length", "both"] = "both"
    t0: float = Field(default=0.0, description="shift_t0: new first time stamp.")


@register
class Align(BaseOp):
    id = "prep.time.align"
    version = "1.0.0"
    category = "time"
    summary = (
        "Align trajectories: shift time stamps to a common t0 and/or truncate every trajectory to "
        "the shortest length."
    )
    when_to_use = (
        "Trajectories from restarts or different output settings; loaders that batch trajectories "
        "need equal lengths and comparable time axes."
    )
    changes_fields = True
    changes_manifest = True
    Params = AlignParams

    def plan_samples(self, ref: DatasetRef, params: AlignParams) -> list[str]:
        return _transient_ids(ref)

    def _common(self, ref: DatasetRef) -> int:
        return min(
            (e.n_steps for e in ref.manifest.samples if e.temporal == "transient"), default=0
        )

    def plan_sample(self, ed: SampleEditor, params: AlignParams) -> list[Change]:
        return [
            Change(target=f"samples/{ed.id}/times", action="modify", detail={"mode": params.mode})
        ]

    def apply_sample(self, ed: SampleEditor, params: AlignParams, *, seed: int) -> SampleOutcome:
        from frontier_prep.io import read_manifest

        s = ed.sample
        t = s.times
        if t is None:
            return SampleOutcome(sample_id=ed.id, skipped=True)
        changes = []
        if params.mode in ("common_length", "both"):
            m = read_manifest(ed.dir.parent.parent)
            n = min(e.n_steps for e in m.samples if e.temporal == "transient")
            if s.n_steps > n:
                _subset_steps(ed, np.arange(n))
                changes.append(
                    Change(
                        target=f"samples/{ed.id}/times",
                        action="modify",
                        before={"n_steps": s.n_steps},
                        after={"n_steps": n},
                    )
                )
                t = ed.reopen().times
        if (
            params.mode in ("shift_t0", "both")
            and t is not None
            and abs(float(t[0]) - params.t0) > 0
        ):
            shift = params.t0 - float(t[0])
            ed.write_root_array("times", t + shift, ["time"])
            shifted = t + shift

            def _m(m: Any) -> None:
                m.time = time_info(shifted, m.time.units if m.time else "s")
                m.extra.setdefault("time", {})["shift"] = shift

            ed.update_meta(_m)
            changes.append(
                Change(target=f"samples/{ed.id}/times", action="modify", detail={"shift": shift})
            )
        if not changes:
            return SampleOutcome(sample_id=ed.id, skipped=True, stats=_time_stats(ed))
        return SampleOutcome(sample_id=ed.id, changes=changes, stats=_time_stats(ed))

    def apply_manifest(
        self, ref: DatasetRef, manifest: Manifest, params: Any, outcomes: list[SampleOutcome]
    ) -> tuple[list[Change], dict[str, Any]]:
        _finish_manifest(ref, manifest, outcomes)
        return [Change(target="manifest/time", action="modify")], {}


# -- time.drop_duplicate_frames ---------------------------------------------------------------


def frame_hashes(ed: SampleEditor, fields: list[str] | None = None) -> list[str]:
    s = ed.sample
    names = fields or [
        n
        for n in s.field_names()
        if s.is_time_dependent(n, "cell" if n in s.field_names("cell") else None)
    ]
    hashes = [hashlib.sha256() for _ in range(s.n_steps)]
    for n in names:
        loc = "cell" if n in s.field_names("cell") else None
        for t, step in enumerate(ed.iter_steps(n, location=loc)):
            hashes[t].update(np.ascontiguousarray(step).tobytes())
    return [h.hexdigest()[:16] for h in hashes]


class DropDuplicatesParams(Params):
    fields: list[str] | None = None
    adjacent_only: bool = Field(
        default=True,
        description=(
            "Only drop a frame identical to its predecessor (restart overlaps); False drops any "
            "repeat."
        ),
    )
    keep: Literal["first", "last"] = "first"


@register
class DropDuplicateFrames(BaseOp):
    id = "prep.time.drop_duplicate_frames"
    version = "1.0.0"
    category = "time"
    summary = (
        "Drop bit-identical repeated frames (restart overlaps, duplicated initial frames) found by "
        "hashing every step."
    )
    when_to_use = (
        "quality reports duplicated steps or temporal_continuity frozen frames inside a "
        "trajectory; The Well-style sets that repeat the initial frame."
    )
    changes_fields = True
    changes_manifest = True
    produces_assertions = True
    Params = DropDuplicatesParams

    def plan_samples(self, ref: DatasetRef, params: DropDuplicatesParams) -> list[str]:
        return _transient_ids(ref)

    def apply_sample(
        self, ed: SampleEditor, params: DropDuplicatesParams, *, seed: int
    ) -> SampleOutcome:
        hs = frame_hashes(ed, params.fields)
        n = len(hs)
        drop: list[int] = []
        if params.adjacent_only:
            for t in range(1, n):
                if hs[t] == hs[t - 1]:
                    drop.append(t if params.keep == "first" else t - 1)
        else:
            seen: dict[str, int] = {}
            for t, h in enumerate(hs):
                if h in seen:
                    drop.append(t if params.keep == "first" else seen[h])
                seen[h] = t
        drop = sorted(set(drop))
        detail = {"n_steps": n, "duplicates": drop[:50], "n_duplicates": len(drop)}
        if not drop:
            return SampleOutcome(
                sample_id=ed.id,
                skipped=True,
                stats={**_time_stats(ed), "n_duplicates": 0},
                assertions=[
                    Assertion(
                        check="duplicate_frames",
                        status="pass",
                        target=f"samples/{ed.id}",
                        detail=detail,
                    )
                ],
            )
        keep = np.array([t for t in range(n) if t not in set(drop)])
        _subset_steps(ed, keep)
        ed.update_meta(
            lambda m: m.extra.setdefault("time", {}).update({"dropped_duplicate_frames": drop})
        )
        return SampleOutcome(
            sample_id=ed.id,
            changes=[
                Change(
                    target=f"samples/{ed.id}/times",
                    action="modify",
                    before={"n_steps": n},
                    after={"n_steps": int(len(keep))},
                    detail=detail,
                )
            ],
            assertions=[
                Assertion(
                    check="duplicate_frames",
                    status="warn",
                    target=f"samples/{ed.id}",
                    detail=detail,
                )
            ],
            stats={**_time_stats(ed), "n_duplicates": len(drop)},
        )

    def apply_manifest(
        self, ref: DatasetRef, manifest: Manifest, params: Any, outcomes: list[SampleOutcome]
    ) -> tuple[list[Change], dict[str, Any]]:
        _finish_manifest(ref, manifest, outcomes)
        return [Change(target="manifest/time", action="modify")], {
            "n_duplicates": {o.sample_id: o.stats.get("n_duplicates", 0) for o in outcomes}
        }


# -- time.window_average ----------------------------------------------------------------------


class WindowAverageParams(Params):
    fields: list[str] | None = None
    start: int = Field(default=0, description="First step of the window (after burn-in).")
    end: int | None = None
    suffix: str = "_mean"
    rms: bool = Field(default=False, description="Also write <field>_rms (fluctuation RMS).")


@register
class WindowAverage(BaseOp):
    id = "prep.time.window_average"
    version = "1.0.0"
    category = "time"
    summary = (
        "Time-average fields over a documented window into steady <field>_mean (and _rms) arrays; "
        "the window is recorded in meta.time.averaging_window."
    )
    when_to_use = (
        "To build a steady target from a scale-resolving run (DES/LES) with an explicit, auditable "
        "window instead of a solver-side mean of unknown span."
    )
    changes_fields = True
    Params = WindowAverageParams

    def plan_samples(self, ref: DatasetRef, params: WindowAverageParams) -> list[str]:
        return _transient_ids(ref)

    def plan_sample(self, ed: SampleEditor, params: WindowAverageParams) -> list[Change]:
        s = ed.sample
        names = params.fields or [
            n
            for n in s.field_names()
            if s.is_time_dependent(n, "cell" if n in s.field_names("cell") else None)
        ]
        return [Change(target=ed.target(n + params.suffix), action="add") for n in names]

    def apply_sample(
        self, ed: SampleEditor, params: WindowAverageParams, *, seed: int
    ) -> SampleOutcome:
        s = ed.sample
        n = s.n_steps
        a, b = params.start, n if params.end is None else min(params.end, n)
        if b - a < 1:
            return SampleOutcome(
                sample_id=ed.id,
                skipped=True,
                warnings=[OpWarning(code="window", message=f"{ed.id}: empty window", target=ed.id)],
            )
        names = params.fields or [
            m
            for m in s.field_names()
            if s.is_time_dependent(m, "cell" if m in s.field_names("cell") else None)
        ]
        changes = []
        for name in names:
            loc = ed.location_of(name)
            acc = None
            acc2 = None
            for t, step in enumerate(ed.iter_steps(name, location=loc)):
                if t < a or t >= b:
                    continue
                x = step.astype(np.float64)  # ordered accumulation, float64
                acc = x if acc is None else acc + x
                if params.rms:
                    acc2 = x * x if acc2 is None else acc2 + x * x
            assert acc is not None
            mean = acc / (b - a)
            info = ed.field_info(name)
            dims = [d for d in s.dims(name, loc) if d != "time"]
            ed.write_field(
                name + params.suffix,
                mean.astype(np.float32),
                location=loc,
                dims=dims,
                units=info.units if info else None,
                meta_extra={"derived_from": name, "window": [a, b]},
            )
            changes.append(
                Change(
                    target=ed.target(name + params.suffix, loc),
                    action="add",
                    detail={"window": [a, b]},
                )
            )
            if params.rms and acc2 is not None:
                var = np.maximum(acc2 / (b - a) - mean * mean, 0.0)
                ed.write_field(
                    name + "_rms",
                    np.sqrt(var).astype(np.float32),
                    location=loc,
                    dims=dims,
                    units=info.units if info else None,
                    meta_extra={"derived_from": name, "window": [a, b]},
                )
                changes.append(Change(target=ed.target(name + "_rms", loc), action="add"))
        t = s.times
        window = float(t[b - 1] - t[a]) if t is not None else None

        def _m(m: Any) -> None:
            if m.time is not None:
                m.time.averaging_window = window
            m.extra.setdefault("time", {})["averaging"] = {"start": a, "end": b, "fields": names}

        ed.update_meta(_m)
        return SampleOutcome(sample_id=ed.id, changes=changes, stats={"window": [a, b]})


# -- time.continuity_check --------------------------------------------------------------------


class ContinuityParams(Params):
    fields: list[str] | None = None
    jump_z: float = Field(
        default=6.0, description="Robust z of the log step change vs its neighbours."
    )
    jump_ratio: float = Field(
        default=4.0,
        description="Step change must also exceed this multiple of the neighbours' median.",
    )
    fail_ratio: float = Field(default=20.0, description="Fail when a jump exceeds this multiple.")
    write_signal: bool = Field(
        default=True, description="Store the relative step change as signals/continuity_<field>."
    )


@register
class ContinuityCheck(BaseOp):
    id = "prep.time.continuity_check"
    version = "1.0.0"
    category = "time"
    summary = (
        "Flag pressure/velocity jumps between consecutive steps (robust z of the log step change), "
        "frozen frames and non-monotonic times; stores the step-change signal."
    )
    when_to_use = (
        "Before training rollouts: a single glitch step teaches a model to glitch; flagged steps "
        "are candidates for trim, drop or a judgment (physical vs numerical)."
    )
    read_only = True
    produces_assertions = True
    changes_fields = True
    Params = ContinuityParams

    def plan_samples(self, ref: DatasetRef, params: ContinuityParams) -> list[str]:
        return _transient_ids(ref)

    def apply_sample(
        self, ed: SampleEditor, params: ContinuityParams, *, seed: int
    ) -> SampleOutcome:
        from frontier_data.quality_transient import is_signal_sample, signal_stack

        s = ed.sample
        t = s.times
        # a signal sample's state is its channels (standardised, as one vector): its grid
        # fields only describe them, and a jump in a measured channel is physics, not a glitch
        signal = params.fields is None and is_signal_sample(s)
        names = params.fields or [
            n
            for n in s.field_names()
            if s.is_time_dependent(n, "cell" if n in s.field_names("cell") else None)
        ]
        if signal:
            names = ["signals"]
        assertions = []
        changes = []
        stats: dict[str, Any] = {}
        worst = "pass"
        if t is not None and len(t) > 1 and not np.all(np.diff(t) > 0):
            assertions.append(
                Assertion(check="times_monotonic", status="fail", target=f"samples/{ed.id}/times")
            )
            worst = "fail"
        for name in names:
            prev = None
            r = np.full(s.n_steps, np.nan)
            steps = (
                iter(signal_stack(s)[0])
                if signal
                else ed.iter_steps(name, location=ed.location_of(name))
            )
            for k, step in enumerate(steps):
                x = step.astype(np.float64)
                if prev is not None:
                    den = float(np.sqrt(np.sum(prev * prev)))
                    num = float(np.sqrt(np.sum((x - prev) ** 2)))
                    # an all-zero previous frame (displacement at t=0) has no defined change
                    r[k] = num / den if den > 0 else (0.0 if num == 0 else np.nan)
                prev = x
            jumps, frozen = _find_jumps(r, params.jump_z, params.jump_ratio)
            ratios = [j["ratio"] for j in jumps]
            status = "pass"
            if jumps:
                status = "fail" if max(ratios) >= params.fail_ratio and not signal else "warn"
            if frozen:
                status = "warn" if status == "pass" else status
            detail = {
                "jumps": jumps[:20],
                "n_jumps": len(jumps),
                "frozen_steps": frozen[:20],
                "n_frozen": len(frozen),
                "median_step_change": float(np.nanmedian(r[1:])) if s.n_steps > 1 else None,
            }
            stats[name] = detail
            assertions.append(
                Assertion(
                    check="temporal_continuity",
                    status=status,
                    target=f"samples/{ed.id}/{name}",
                    detail=detail,
                )
            )
            worst = _worse(worst, status)
            if params.write_signal and not signal:  # never mixed into the channels
                ed.write_root_array(
                    f"signals/continuity_{name}",
                    np.nan_to_num(r, nan=0.0).astype(np.float32)[:, None],
                    ["time", "component"],
                )
                changes.append(
                    Change(target=f"samples/{ed.id}/signals/continuity_{name}", action="add")
                )
        ed.update_meta(
            lambda m: m.extra.setdefault("time", {}).update(
                {
                    "continuity": {
                        k: {"n_jumps": v["n_jumps"], "n_frozen": v["n_frozen"]}
                        for k, v in stats.items()
                    }
                }
            )
        )
        return SampleOutcome(
            sample_id=ed.id,
            changes=changes,
            assertions=assertions,
            stats={"status": worst, **stats},
        )


def _worse(a: str, b: str) -> str:
    order = {"pass": 0, "skip": 0, "warn": 1, "fail": 2}
    return a if order[a] >= order[b] else b


def _find_jumps(
    r: np.ndarray, z_thr: float, ratio_thr: float
) -> tuple[list[dict[str, Any]], list[int]]:
    """Jumps: robust z of log(r) against the +-5 neighbours and ratio to their median. Frozen:
    relative change below 1e-7."""
    jumps: list[dict[str, Any]] = []
    frozen = [int(k) for k in np.nonzero(r < 1e-7)[0] if k > 0]
    valid = np.isfinite(r) & (r > 0)
    lr = np.where(valid, np.log(np.where(valid, r, 1.0)), np.nan)
    for k in np.nonzero(valid)[0]:
        lo, hi = max(1, k - 5), min(len(r), k + 6)
        nb = np.concatenate([lr[lo:k], lr[k + 1 : hi]])
        nb = nb[np.isfinite(nb)]
        if len(nb) < 3:
            continue
        med = float(np.median(nb))
        mad = float(np.median(np.abs(nb - med))) * 1.4826
        z = (lr[k] - med) / max(mad, 1e-12)
        ratio = float(np.exp(lr[k] - med))
        if z >= z_thr and ratio >= ratio_thr:
            jumps.append({"step": int(k), "ratio": ratio, "z": float(z), "rel_change": float(r[k])})
    return jumps, frozen
