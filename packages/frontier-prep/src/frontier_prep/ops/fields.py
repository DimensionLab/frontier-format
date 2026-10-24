# SPDX-License-Identifier: Apache-2.0
"""Schema and field ops: rename, canonical mapping, location cast, components, NaN policy,
outlier detection, smoothing, normalisation statistics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import Field

from arrayroute.reader import ReadStats
from arrayroute.session import LocalReadSession
from arrayroute.statistics import ChannelMoments
from frontier_data.dataset import Manifest
from frontier_data.mesh import SurfaceMesh
from frontier_data.query import FieldReader
from frontier_prep import vocabulary as vocab
from frontier_prep.base import BaseOp, DatasetRef, SampleOutcome
from frontier_prep.io import SampleEditor
from frontier_prep.normalization import ALGORITHM as NORMALIZATION_ALGORITHM
from frontier_prep.normalization import fit_field
from frontier_prep.op import (
    Assertion,
    Change,
    Judgment,
    OpWarning,
    Params,
    Plan,
    Result,
    register,
)


def _fields_of(ed: SampleEditor, names: list[str] | None) -> list[str]:
    have = ed.field_names()
    if names is None:
        return have
    return [n for n in names if n in have]


# -- fields.rename -----------------------------------------------------------------------------


class RenameParams(Params):
    map: dict[str, str] = Field(description="old name -> new name", min_length=1)
    units: dict[str, str] = Field(default_factory=dict, description="new name -> unit string")
    missing: Literal["skip", "fail"] = Field(
        default="skip", description="What to do when a sample lacks a source field."
    )


@register
class RenameFields(BaseOp):
    id = "prep.fields.rename"
    version = "1.0.0"
    category = "fields"
    summary = "Rename fields (directory rename, no array is rewritten) and optionally set units."
    description = (
        "Renames `fields/<old>` / `cell_fields/<old>` to the new name in every sample and "
        "updates meta.json and the manifest. Chunks are shared with the previous version."
    )
    when_to_use = (
        "A solver name should become the dataset's convention and you already know the target "
        "name; use prep.fields.map to get proposals from the canonical vocabulary."
    )
    changes_fields = True
    changes_manifest = True
    Params = RenameParams

    def plan_sample(self, ed: SampleEditor, params: RenameParams) -> list[Change]:
        out = []
        for old, new in params.map.items():
            if old in ed.field_names():
                out.append(
                    Change(
                        target=f"samples/{ed.id}/{ed.location_of(old)}:{old}",
                        action="rename",
                        before={"name": old},
                        after={"name": new, "units": params.units.get(new)},
                    )
                )
        return out

    def plan_extra(
        self, ref: DatasetRef, params: RenameParams
    ) -> tuple[list[OpWarning], list[Judgment]]:
        w: list[OpWarning] = []
        have = {f.name for f in ref.manifest.fields}
        for old, new in params.map.items():
            if old not in have:
                w.append(
                    OpWarning(
                        severity="fail" if params.missing == "fail" else "warn",
                        code="field-missing",
                        message=f"no field {old!r} in the manifest",
                        target=old,
                    )
                )
            if new in have and new != old:
                w.append(
                    OpWarning(
                        severity="fail", code="field-exists", message=f"{new!r} exists", target=new
                    )
                )
        return w, []

    def apply_sample(self, ed: SampleEditor, params: RenameParams, *, seed: int) -> SampleOutcome:
        changes: list[Change] = []
        for old, new in params.map.items():
            if old not in ed.field_names():
                continue
            loc = ed.location_of(old)
            ed.rename_field(old, new)
            if new in params.units:
                ed.set_units(new, params.units[new])
            changes.append(
                Change(
                    target=f"samples/{ed.id}/{loc}:{new}",
                    action="rename",
                    before={"name": old},
                    after={"name": new},
                )
            )
        return SampleOutcome(sample_id=ed.id, changes=changes)

    def summarize(self, pr: Plan | Result) -> str:
        n_samples = len({c.target.split("/")[1] for c in pr.changes})
        return f"rename {len(pr.params.get('map', {}))} field(s) in {n_samples} sample(s)"


# -- fields.map --------------------------------------------------------------------------------


class MapParams(Params):
    decisions: dict[str, str] = Field(
        default_factory=dict,
        description="source name -> canonical name for ambiguous or low-confidence proposals.",
    )
    min_confidence: float = Field(default=0.75, ge=0, le=1)
    apply: bool = Field(default=True, description="False: only report the proposal (plan).")
    set_units: bool = Field(
        default=True, description="Set the canonical unit where none is recorded."
    )
    keep_unmapped: bool = True


@register
class MapFields(BaseOp):
    id = "prep.fields.map"
    version = "1.0.0"
    category = "fields"
    summary = (
        "Map solver field names to the canonical physics vocabulary (with units and dimensional "
        "checks)."
    )
    description = (
        "Proposes a canonical name for every field from the vocabulary (aliases, recorded units, "
        "channel count); names that are ambiguous (dynamic vs kinematic pressure) or below "
        "`min_confidence` come back as judgments and are only applied when `decisions` settles "
        "them."
    )
    when_to_use = (
        "Right after ingest of any solver output, before merging datasets or training across "
        "sources."
    )
    changes_fields = True
    changes_manifest = True
    Params = MapParams

    def proposals(
        self, ref: DatasetRef, params: MapParams
    ) -> tuple[dict[str, str], list[Judgment], list[OpWarning]]:
        mapping: dict[str, str] = {}
        judgments: list[Judgment] = []
        warnings: list[OpWarning] = []
        taken = {f.name for f in ref.manifest.fields}
        for f in ref.manifest.fields:
            if f.location == "global":
                continue
            if f.name in params.decisions:
                target = params.decisions[f.name]
                if target and target != f.name:
                    mapping[f.name] = target
                continue
            if f.name in vocab.FIELDS:
                continue
            m = vocab.map_name(f.name, f.units, f.channels)
            if (
                m.canonical
                and m.confidence >= params.min_confidence
                and m.unit_compatible is not False
            ):
                if m.canonical in taken and m.canonical != f.name:
                    warnings.append(
                        OpWarning(
                            code="target-exists",
                            message=f"{f.name} -> {m.canonical} but {m.canonical} already exists",
                            target=f.name,
                        )
                    )
                    continue
                mapping[f.name] = m.canonical
            elif m.options:
                judgments.append(
                    Judgment(
                        question=(
                            f"Which canonical field is {f.name!r} (units {f.units!r}, "
                            f"{f.channels} ch)? {m.reason}"
                        ),
                        target=f"fields/{f.name}",
                        options=m.options + ["(keep)"],
                        prior=[round(1.0 / len(m.options), 3)] * len(m.options) + [0.0],
                        param=f"decisions.{f.name}",
                    )
                )
            else:
                warnings.append(
                    OpWarning(
                        severity="info",
                        code="unmapped",
                        message=f"no canonical name for {f.name!r}",
                        target=f.name,
                    )
                )
        return mapping, judgments, warnings

    def plan_sample(self, ed: SampleEditor, params: MapParams) -> list[Change]:
        return []

    def plan_manifest(self, ref: DatasetRef, params: MapParams) -> list[Change]:
        mapping, _, _ = self.proposals(ref, params)
        if not params.apply:
            return []
        return [
            Change(
                target=f"manifest/fields/{old}",
                action="rename",
                before={"name": old},
                after={
                    "name": new,
                    "units": vocab.FIELDS[new].unit if new in vocab.FIELDS else None,
                },
            )
            for old, new in mapping.items()
        ]

    def plan_extra(
        self, ref: DatasetRef, params: MapParams
    ) -> tuple[list[OpWarning], list[Judgment]]:
        _, judgments, warnings = self.proposals(ref, params)
        return warnings, judgments

    def apply_sample(self, ed: SampleEditor, params: MapParams, *, seed: int) -> SampleOutcome:
        # the mapping is decided from the manifest; each sample applies the same table
        mapping = params.decisions if not params.apply else self._mapping_cache(ed, params)
        changes: list[Change] = []
        for old, new in mapping.items():
            if old not in ed.field_names() or not new or new == old:
                continue
            loc = ed.location_of(old)
            ed.rename_field(old, new)
            cf = vocab.FIELDS.get(new)
            info = ed.field_info(new)
            if params.set_units and cf is not None and info is not None and info.units in ("-", ""):
                ed.set_units(new, cf.unit or "-")
            ed.meta.extra.setdefault("fields", {}).setdefault(new, {})["source_name"] = old
            ed.save_meta()
            changes.append(
                Change(
                    target=f"samples/{ed.id}/{loc}:{new}",
                    action="rename",
                    before={"name": old},
                    after={"name": new},
                )
            )
        return SampleOutcome(sample_id=ed.id, changes=changes)

    def _mapping_cache(self, ed: SampleEditor, params: MapParams) -> dict[str, str]:
        from frontier_prep.io import read_manifest

        root = ed.dir.parent.parent
        ref = DatasetRef(root)
        ref.ds.manifest = read_manifest(root)
        mapping, _, _ = self.proposals(ref, params)
        return mapping

    def summarize(self, pr: Plan | Result) -> str:
        return f"map {len(pr.changes)} field name(s) to the canonical vocabulary"


# -- fields.cast_location ----------------------------------------------------------------------


class CastLocationParams(Params):
    fields: list[str] | None = Field(default=None, description="Fields to convert (default: all).")
    to: Literal["point", "cell"]
    keep_source: bool = Field(
        default=True, description="Keep the native array next to the derived one."
    )
    suffix: str = Field(
        default="",
        description="Name suffix of the derived array (default: same name at the other location).",
    )


@register
class CastLocation(BaseOp):
    id = "prep.fields.cast_location"
    version = "1.0.0"
    category = "fields"
    summary = (
        "Convert surface fields between cell and vertex locations (area-weighted), tagging the "
        "derived copy."
    )
    description = (
        "cell -> point: area-weighted average of incident faces; point -> cell: mean of the three "
        "vertices. The derived array records `derived_from` so trainers can pick the native one."
    )
    when_to_use = (
        "A model needs vertex data but the solver wrote cell data (or vice versa). Never train on "
        "a derived copy when the native location is available."
    )
    changes_fields = True
    Params = CastLocationParams

    def plan_samples(self, ref: DatasetRef, params: CastLocationParams) -> list[str]:
        return [e.id for e in ref.manifest.samples if e.sample_kind == "surface"]

    def plan_sample(self, ed: SampleEditor, params: CastLocationParams) -> list[Change]:
        src_loc = "cell" if params.to == "point" else "point"
        out = []
        for name in _fields_of(ed, params.fields):
            if name in ed.field_names(src_loc) and (name + params.suffix) not in ed.field_names(
                params.to
            ):
                out.append(
                    Change(
                        target=ed.target(name + params.suffix, params.to),
                        action="add",
                        detail={"from": src_loc},
                    )
                )
        return out

    def apply_sample(
        self, ed: SampleEditor, params: CastLocationParams, *, seed: int
    ) -> SampleOutcome:
        s = ed.sample
        src_loc = "cell" if params.to == "point" else "point"
        mesh = SurfaceMesh(points=s.points(), faces=s.cells)
        changes = []
        for name in _fields_of(ed, params.fields):
            if name not in ed.field_names(src_loc) or (name + params.suffix) in ed.field_names(
                params.to
            ):
                continue
            conv = mesh.cell_to_point if params.to == "point" else mesh.point_to_cell
            if s.is_time_dependent(name, src_loc):
                data = np.stack([conv(step) for step in ed.iter_steps(name, location=src_loc)])
            else:
                data = conv(ed.field(name, location=src_loc))
            info = ed.field_info(name)
            ed.write_field(
                name + params.suffix,
                data,
                location=params.to,
                units=info.units if info else None,
                meta_extra={
                    "derived_from": f"{src_loc}:{name}",
                    "method": "area_weighted" if params.to == "point" else "vertex_mean",
                },
            )
            if not params.keep_source and params.suffix:
                ed.delete_field(name, location=src_loc)
            changes.append(
                Change(
                    target=ed.target(name + params.suffix, params.to),
                    action="add",
                    detail={"from": src_loc},
                )
            )
        return SampleOutcome(sample_id=ed.id, changes=changes)


# -- fields.components -------------------------------------------------------------------------


class ComponentsParams(Params):
    field: str
    reorder: list[int] | None = Field(
        default=None, description="New component order, e.g. [0,2,1]."
    )
    signs: list[float] | None = Field(
        default=None, description="Per-component sign/scale, e.g. [1,-1,1]."
    )
    split: bool = Field(default=False, description="Split into <field>_x/_y/_z scalar fields.")
    merge: list[str] | None = Field(
        default=None, description="Merge these scalar fields into `field`."
    )
    magnitude: bool = Field(default=False, description="Also write <field>_magnitude.")


@register
class Components(BaseOp):
    id = "prep.fields.components"
    version = "1.0.0"
    category = "fields"
    summary = "Reorder, flip, split or merge vector components; optionally derive the magnitude."
    when_to_use = (
        "A dataset uses y-up where Frontier uses z-up, a solver wrote components as separate "
        "scalars (U1, U2), or a model needs |U|."
    )
    changes_fields = True
    Params = ComponentsParams

    def plan_sample(self, ed: SampleEditor, params: ComponentsParams) -> list[Change]:
        names = ed.field_names()
        if params.merge:
            if all(n in names for n in params.merge):
                return [
                    Change(
                        target=f"samples/{ed.id}/fields/{params.field}",
                        action="add",
                        detail={"merge": params.merge},
                    )
                ]
            return []
        if params.field not in names:
            return []
        out = []
        if params.reorder or params.signs:
            out.append(
                Change(
                    target=f"samples/{ed.id}/fields/{params.field}",
                    action="modify",
                    detail={"reorder": params.reorder, "signs": params.signs},
                )
            )
        if params.split:
            out.append(
                Change(
                    target=f"samples/{ed.id}/fields/{params.field}_*",
                    action="add",
                    detail={"split": True},
                )
            )
        if params.magnitude:
            out.append(
                Change(target=f"samples/{ed.id}/fields/{params.field}_magnitude", action="add")
            )
        return out

    def apply_sample(
        self, ed: SampleEditor, params: ComponentsParams, *, seed: int
    ) -> SampleOutcome:
        changes = self.plan_sample(ed, params)
        if not changes:
            return SampleOutcome(sample_id=ed.id, skipped=True)
        units = None
        if params.merge:
            parts = [ed.field(n) for n in params.merge]
            loc = ed.location_of(params.merge[0])
            data = np.concatenate([p.reshape(*p.shape[:-1], -1) for p in parts], axis=-1)
            info = ed.field_info(params.merge[0])
            units = info.units if info else None
            ed.write_field(
                params.field,
                data,
                location=loc,
                units=units,
                meta_extra={"merged_from": params.merge},
            )
            return SampleOutcome(sample_id=ed.id, changes=changes)
        loc = ed.location_of(params.field)
        data = ed.field(params.field, location=loc)
        info = ed.field_info(params.field)
        units = info.units if info else None
        if params.reorder:
            data = data[..., params.reorder]
        if params.signs:
            data = data * np.asarray(params.signs, dtype=np.float32)
        if params.reorder or params.signs:
            ed.write_field(
                params.field,
                data,
                location=loc,
                units=units,
                meta_extra={"reorder": params.reorder, "signs": params.signs},
            )
        if params.split:
            for i, ax in enumerate(
                "xyz"[: data.shape[-1]] if data.shape[-1] <= 3 else range(data.shape[-1])
            ):
                ed.write_field(
                    f"{params.field}_{ax}",
                    data[..., i : i + 1],
                    location=loc,
                    units=units,
                    meta_extra={"derived_from": params.field},
                )
        if params.magnitude:
            mag = np.sqrt(np.sum(data.astype(np.float64) ** 2, axis=-1, keepdims=True)).astype(
                np.float32
            )
            ed.write_field(
                f"{params.field}_magnitude",
                mag,
                location=loc,
                units=units,
                meta_extra={"derived_from": params.field},
            )
        return SampleOutcome(sample_id=ed.id, changes=changes)


# -- fields.nan_fix ----------------------------------------------------------------------------


class NanFixParams(Params):
    fields: list[str] | None = None
    policy: Literal["fill_value", "fill_mean", "mask", "drop_steps", "fail"] = "mask"
    value: float = 0.0
    max_frac: float = Field(
        default=0.05, description="Fail when a field has more non-finite values than this fraction."
    )


@register
class NanFix(BaseOp):
    id = "prep.fields.nan_fix"
    version = "1.0.0"
    category = "fields"
    summary = (
        "Apply a recorded policy to NaN/inf values: fill, mask (write static/<field>_valid), drop "
        "steps or fail."
    )
    when_to_use = (
        "quality reports nan_inf failures. Prefer `mask` (keeps evidence) or `drop_steps` for "
        "transient blow-ups; filling is for inputs a model must not see as holes."
    )
    changes_fields = True
    produces_assertions = True
    Params = NanFixParams

    def plan_sample(self, ed: SampleEditor, params: NanFixParams) -> list[Change]:
        return []  # needs array reads; the plan only lists samples

    def apply_sample(self, ed: SampleEditor, params: NanFixParams, *, seed: int) -> SampleOutcome:
        changes: list[Change] = []
        assertions: list[Assertion] = []
        drop: set[int] = set()
        stats: dict[str, Any] = {}
        for name in _fields_of(ed, params.fields):
            loc = ed.location_of(name)
            data = ed.field(name, location=loc)
            bad = ~np.isfinite(data)
            n_bad = int(bad.sum())
            if n_bad == 0:
                continue
            frac = n_bad / data.size
            stats[name] = {"n_nonfinite": n_bad, "frac": frac}
            if frac > params.max_frac or params.policy == "fail":
                assertions.append(
                    Assertion(
                        check="nan_inf",
                        status="fail",
                        target=f"samples/{ed.id}/{name}",
                        detail=stats[name],
                    )
                )
                continue
            target = f"samples/{ed.id}/{'cell_fields' if loc == 'cell' else 'fields'}/{name}"
            if params.policy == "fill_value":
                data = np.where(bad, np.float32(params.value), data)
            elif params.policy == "fill_mean":
                mean = np.nanmean(
                    np.where(np.isinf(data), np.nan, data).astype(np.float64),
                    axis=tuple(range(data.ndim - 1)),
                )
                data = np.where(bad, mean.astype(np.float32), data)
            elif params.policy == "mask":
                valid = (~bad).all(axis=-1) if data.ndim > 1 else ~bad
                if data.ndim == 3:
                    valid = valid.all(axis=0)
                ed.write_field(
                    f"{name}_valid",
                    valid.astype(np.uint8)[..., None],
                    location=loc,
                    units="-",
                    meta_extra={"derived_from": name, "mask": "finite"},
                )
                data = np.where(bad, np.float32(0.0), data)
            elif params.policy == "drop_steps" and data.ndim == 3:
                drop |= set(np.nonzero(bad.reshape(bad.shape[0], -1).any(axis=1))[0].tolist())
                continue
            ed.write_field(
                name, data, location=loc, meta_extra={"nan_policy": params.policy, "n_fixed": n_bad}
            )
            changes.append(
                Change(
                    target=target,
                    action="modify",
                    detail={"policy": params.policy, "n_fixed": n_bad},
                )
            )
            assertions.append(
                Assertion(
                    check="nan_inf",
                    status="warn",
                    target=f"samples/{ed.id}/{name}",
                    detail=stats[name],
                )
            )
        if drop:
            keep = np.array(sorted(set(range(ed.sample.n_steps)) - drop))
            _subset_steps(ed, keep)
            changes.append(
                Change(
                    target=f"samples/{ed.id}/times",
                    action="modify",
                    detail={"dropped_steps": sorted(drop)},
                )
            )
        return SampleOutcome(sample_id=ed.id, changes=changes, assertions=assertions, stats=stats)


def _subset_steps(ed: SampleEditor, keep: np.ndarray) -> None:
    """Keep the time steps ``keep`` in every time-dependent array (used by nan_fix, time.*)."""
    s = ed.sample
    times = s.times
    for name in s.field_names():
        loc = "cell" if name in s.field_names("cell") else None
        if s.is_time_dependent(name, loc):
            info = ed.field_info(name)
            data = s.field(name, keep, location=loc)
            ed.write_field(
                name,
                data,
                location=loc or ("grid" if s.kind == "grid" else "point"),
                units=info.units if info else None,
            )
    g: Any = s.group
    if "points_t" in g:
        ed.write_root_array(
            "points_t", np.asarray(g["points_t"][...])[keep], ["time", "point", "space"]
        )
    for sig in s.signal_names:
        arr = s.signal(sig)
        if arr.ndim == 2:
            ed.write_root_array(f"signals/{sig}", arr[keep], ["time", "component"])
    if times is not None:
        ed.write_root_array("times", times[keep], ["time"])
        ed.set_attrs(n_steps=int(len(keep)))
        from frontier_data.arrays import time_info

        ed.update_meta(
            lambda m: setattr(m, "time", time_info(times[keep], m.time.units if m.time else "s"))
        )


# -- fields.outlier_detect ---------------------------------------------------------------------


class OutlierParams(Params):
    fields: list[str] | None = None
    z: float = Field(default=6.0, description="Robust z threshold (MAD-based) per field.")
    spatial: bool = Field(
        default=True,
        description="Also flag points whose value deviates from their mesh neighbours.",
    )
    spatial_z: float = 6.0
    physics: bool = Field(
        default=True, description="Apply canonical plausibility ranges (e.g. |Cp| <= 3)."
    )
    write_mask: bool = Field(default=True, description="Write static/<field>_outlier masks.")
    max_points: int = Field(
        default=2_000_000, description="Subsample above this for the spatial test."
    )
    expected_scale: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "field -> expected magnitude (p99) in the recorded unit, e.g. {'stress_vm': 3e8} "
            "for steel in Pa; a ratio off by >= 100x raises a unit_plausibility warning with "
            "the implied factor (B3: card vs data)."
        ),
    )


@register
class OutlierDetect(BaseOp):
    id = "prep.fields.outlier_detect"
    version = "1.0.0"
    category = "fields"
    summary = (
        "Flag outliers per field: robust z-score, neighbour-based spatial deviation, physics "
        "plausibility bounds."
    )
    when_to_use = (
        "Before normalisation statistics and before training; the masks are evidence for a "
        "judgment (physical vs numerical) and input to prep.dataset.select."
    )
    changes_fields = True
    produces_assertions = True
    Params = OutlierParams

    def apply_sample(self, ed: SampleEditor, params: OutlierParams, *, seed: int) -> SampleOutcome:
        s = ed.sample
        changes: list[Change] = []
        assertions: list[Assertion] = []
        stats: dict[str, Any] = {}
        neighbours = None
        for name in _fields_of(ed, params.fields):
            loc = ed.location_of(name)
            # transient fields: the last step (initial frames of impact/start-up runs are zero)
            data = (
                ed.field(name, s.n_steps - 1, location=loc)
                if s.is_time_dependent(name, loc)
                else ed.field(name, location=loc)
            )
            x = data.reshape(data.shape[0], -1) if data.ndim >= 2 else data[:, None]
            if s.kind == "grid":
                x = data.reshape(-1, data.shape[-1])
            mag = (
                np.sqrt(np.sum(x.astype(np.float64) ** 2, axis=1))
                if x.shape[1] > 1
                else x[:, 0].astype(np.float64)
            )
            finite = np.isfinite(mag)
            med = float(np.median(mag[finite])) if finite.any() else 0.0
            mad = float(np.median(np.abs(mag[finite] - med))) if finite.any() else 0.0
            scale = (
                1.4826 * mad if mad > 0 else (float(np.std(mag[finite])) if finite.any() else 1.0)
            )
            z = np.abs(mag - med) / max(scale, 1e-30)
            mask = (z > params.z) | ~finite
            detail: dict[str, Any] = {"robust_z": int(mask.sum()), "median": med, "mad": mad}
            if params.spatial and s.kind in ("surface", "mesh") and len(mag) <= params.max_points:
                if neighbours is None:
                    neighbours = _neighbour_index(
                        s.points(),
                        s.cells,
                        loc == "cell",
                        s.points().shape[0] if loc != "cell" else len(s.cells),
                    )
                if neighbours is not None:
                    nb_mean = neighbours @ mag
                    resid = mag - nb_mean
                    rmad = (
                        float(np.median(np.abs(resid[finite] - np.median(resid[finite]))))
                        if finite.any()
                        else 0.0
                    )
                    sz = np.abs(resid) / max(1.4826 * rmad, 1e-30)
                    smask = sz > params.spatial_z
                    detail["spatial"] = int(smask.sum())
                    mask |= smask
            if params.physics:
                cf = vocab.FIELDS.get(name)
                if cf is not None and cf.plausible is not None:
                    lo, hi = cf.plausible
                    pm = (np.abs(mag) > hi) | (np.abs(mag) < lo)
                    detail["physics"] = int(pm.sum())
                    detail["plausible"] = [lo, hi]
                    mask |= pm
            p99 = float(np.percentile(np.abs(mag[finite]), 99)) if finite.any() else 0.0
            if name in params.expected_scale and p99 > 0:
                ratio = p99 / params.expected_scale[name]
                factor = 10.0 ** round(-np.log10(max(ratio, 1e-300)))
                detail["unit_plausibility"] = {"p99": p99, "ratio": ratio, "implied_factor": factor}
                if abs(np.log10(max(ratio, 1e-300))) >= 2:
                    assertions.append(
                        Assertion(
                            check="unit_plausibility",
                            status="warn",
                            target=f"samples/{ed.id}/{name}",
                            detail={
                                "p99": p99,
                                "expected": params.expected_scale[name],
                                "implied_factor": factor,
                                "hint": (
                                    f"values are off by 10^{round(np.log10(factor))}: "
                                    "declared unit is likely wrong"
                                ),
                            },
                        )
                    )
            n = int(mask.sum())
            detail["n_outliers"] = n
            detail["frac"] = n / max(len(mask), 1)
            stats[name] = detail
            status = "pass" if n == 0 else ("warn" if detail["frac"] < 0.01 else "fail")
            assertions.append(
                Assertion(
                    check="field_outliers",
                    status=status,
                    target=f"samples/{ed.id}/{name}",
                    detail=detail,
                )
            )
            if params.write_mask and n:
                arr = mask.astype(np.uint8)
                if s.kind == "grid":
                    arr = arr.reshape(data.shape[:-1])
                ed.write_field(
                    f"{name}_outlier",
                    arr[..., None],
                    location=loc,
                    units="-",
                    meta_extra={"derived_from": name, "mask": "outlier"},
                )
                changes.append(
                    Change(
                        target=ed.target(f"{name}_outlier", loc),
                        action="add",
                        detail=detail,
                    )
                )
        return SampleOutcome(sample_id=ed.id, changes=changes, assertions=assertions, stats=stats)


def _neighbour_index(points: np.ndarray, cells: np.ndarray, per_cell: bool, n: int) -> Any:
    """Row-normalised sparse adjacency (mean over neighbours) for points or cells."""
    from scipy import sparse

    c = np.asarray(cells)
    k = (c >= 0).sum(axis=1)
    if per_cell:
        # cells sharing a vertex are neighbours
        rows, cols = [], []
        for j in range(c.shape[1]):
            v = c[:, j]
            ok = v >= 0
            rows.append(np.nonzero(ok)[0])
            cols.append(v[ok])
        inc = sparse.csr_matrix(
            (np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(c), len(points)),
        )
        adj = (inc @ inc.T).tocsr()
    else:
        rows, cols = [], []
        for i in range(c.shape[1]):
            for j in range(c.shape[1]):
                if i == j:
                    continue
                ok = (c[:, i] >= 0) & (c[:, j] >= 0)
                rows.append(c[ok, i])
                cols.append(c[ok, j])
        adj = sparse.csr_matrix(
            (np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(points), len(points)),
        )
        del k
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg[deg == 0] = 1
    return sparse.diags(1.0 / deg) @ adj


# -- fields.smooth -----------------------------------------------------------------------------


class SmoothParams(Params):
    fields: list[str]
    iterations: int = Field(default=1, ge=1, le=50)
    weight: float = Field(
        default=0.5, gt=0, le=1, description="Laplacian blend: x <- (1-w) x + w mean(neighbours)."
    )
    only_masked: str | None = Field(
        default=None, description="Smooth only where static/<mask> is 1 (e.g. an outlier mask)."
    )
    justification: str = Field(
        description="Why smoothing is legitimate here (recorded in provenance).", min_length=8
    )


@register
class SmoothFields(BaseOp):
    id = "prep.fields.smooth"
    version = "1.0.0"
    category = "fields"
    summary = (
        "Laplacian smoothing over mesh neighbours, optionally only at masked points; requires a "
        "written justification."
    )
    when_to_use = (
        "Rarely: to repair isolated numerical spikes that a judgment classified as artefacts. "
        "Never to make a field look nicer; the justification is recorded."
    )
    changes_fields = True
    Params = SmoothParams

    def plan_sample(self, ed: SampleEditor, params: SmoothParams) -> list[Change]:
        return [
            Change(
                target=f"samples/{ed.id}/fields/{n}",
                action="modify",
                detail={"iterations": params.iterations},
            )
            for n in _fields_of(ed, params.fields)
        ]

    def apply_sample(self, ed: SampleEditor, params: SmoothParams, *, seed: int) -> SampleOutcome:
        s = ed.sample
        if s.kind not in ("surface", "mesh"):
            return SampleOutcome(
                sample_id=ed.id,
                skipped=True,
                warnings=[OpWarning(code="kind", message="smoothing needs a mesh", target=ed.id)],
            )
        changes = []
        cache: dict[bool, Any] = {}
        for name in _fields_of(ed, params.fields):
            loc = ed.location_of(name)
            per_cell = loc == "cell"
            if per_cell not in cache:
                cache[per_cell] = _neighbour_index(s.points(), s.cells, per_cell, 0)
            adj = cache[per_cell]
            data = ed.field(name, location=loc).astype(np.float64)
            where = None
            if params.only_masked and params.only_masked in ed.field_names(loc):
                where = (
                    ed.field(params.only_masked, location=loc).reshape(data.shape[0], -1)[:, 0] > 0
                )
            x = data.reshape(data.shape[0], -1) if data.ndim == 2 else data
            steps = [x] if data.ndim == 2 else list(x)
            out = []
            for st in steps:
                y = st.copy()
                for _ in range(params.iterations):
                    m = adj @ y
                    y_new = (1 - params.weight) * y + params.weight * m
                    y = np.where(where[:, None], y_new, y) if where is not None else y_new
                out.append(y)
            res = out[0] if data.ndim == 2 else np.stack(out)
            info = ed.field_info(name)
            ed.write_field(
                name,
                res.astype(np.float32),
                location=loc,
                units=info.units if info else None,
                meta_extra={
                    "smoothed": {
                        "iterations": params.iterations,
                        "weight": params.weight,
                        "only_masked": params.only_masked,
                        "justification": params.justification,
                    }
                },
            )
            changes.append(
                Change(
                    target=f"samples/{ed.id}/{'cell_fields' if per_cell else 'fields'}/{name}",
                    action="modify",
                    detail={"iterations": params.iterations},
                )
            )
        return SampleOutcome(sample_id=ed.id, changes=changes)


# -- fields.normalize (statistics) -------------------------------------------------------------


class NormalizeParams(Params):
    fields: list[str] | None = None
    split: str = Field(default="train", description="Statistics are fitted on this split only.")
    method: Literal["zscore", "minmax"] = "zscore"
    write: bool = Field(default=True, description="Store under manifest.stats.normalization.")
    max_steps: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Use all training steps, or sample this many evenly spaced steps.",
    )
    read_engine: Literal["zarr-python", "tensorstore", "zarrista"] = "zarr-python"
    read_buffer_mib: int = Field(default=128, ge=1, strict=True)
    statistics_workspace_mib: int = Field(default=32, ge=1, strict=True)
    tensorstore_cache_mib: int = Field(default=0, ge=0, strict=True)


@register
class NormalizeStats(BaseOp):
    id = "prep.fields.normalize"
    version = "2.0.0"
    category = "fields"
    summary = (
        "Fit per-field normalisation statistics on the training split only and record them in the "
        "manifest."
    )
    when_to_use = (
        "Before training; never fit on validation or test data. The trainer reads "
        "manifest.stats.normalization."
    )
    changes_manifest = True
    read_only = True
    Params = NormalizeParams

    def plan_samples(self, ref: DatasetRef, params: NormalizeParams) -> list[str]:
        ids = ref.manifest.splits.get(params.split)
        if not ids:
            raise ValueError(
                f"Normalization split {params.split!r} is missing or empty; create a nonempty "
                "training split before fitting normalization."
            )
        if len(ids) != len(set(ids)) or set(ids) - set(ref.sample_ids()):
            raise ValueError("Normalization split contains duplicate or unknown samples")
        if any(
            set(ids) & set(other)
            for split, other in ref.manifest.splits.items()
            if split != params.split
        ):
            raise ValueError(
                "Normalization split overlaps another split; repair the partition first"
            )
        return sorted(ids, key=_natural)

    def plan_manifest(self, ref: DatasetRef, params: NormalizeParams) -> list[Change]:
        return [
            Change(
                target="manifest/stats/normalization",
                action="modify",
                detail={"split": params.split, "method": params.method},
            )
        ]

    def apply_sample(
        self, ed: SampleEditor, params: NormalizeParams, *, seed: int
    ) -> SampleOutcome:
        s = ed.sample
        session = LocalReadSession(
            params.read_engine, buffer_bytes=params.read_buffer_mib << 20,
            tensorstore_cache_bytes=params.tensorstore_cache_mib << 20,
        )
        acc: dict[str, Any] = {}
        for name in _fields_of(ed, params.fields):
            loc = ed.location_of(name)
            selected_steps = None
            if s.is_time_dependent(name, loc) and params.max_steps is not None:
                count = min(params.max_steps, s.n_steps)
                selected_steps = (
                    [i * (s.n_steps - 1) // (count - 1) for i in range(count)] if count > 1 else [0]
                )
            selections = (
                [None]
                if selected_steps is None
                else [
                    (step, *(slice(None) for _ in s.dims(name, loc)[1:])) for step in selected_steps
                ]
            )
            reader = FieldReader(s, name, location=loc, session=session)
            stats = ReadStats()
            acc[name] = fit_field(
                reader.reader,
                selections,
                workspace_bytes=params.statistics_workspace_mib << 20,
                stats=stats,
            )
            acc[name].update(units=reader.units, selected_steps=selected_steps)
            ed.bytes_read += stats.output_bytes
        return SampleOutcome(sample_id=ed.id, stats=acc)

    def apply_manifest(
        self,
        ref: DatasetRef,
        manifest: Manifest,
        params: NormalizeParams,
        outcomes: list[SampleOutcome],
    ) -> tuple[list[Change], dict[str, Any]]:
        fields: dict[str, Any] = {}
        outcomes = sorted(outcomes, key=lambda o: _natural(o.sample_id))
        for o in outcomes:
            for name, a in o.stats.items():
                channels = len(a["mean"])
                f = fields.setdefault(
                    name,
                    {
                        "moments": ChannelMoments(channels),
                        "min": None,
                        "max": None,
                        "units": a["units"],
                    },
                )
                if len(f["moments"].count) != channels or f["units"] != a["units"]:
                    raise ValueError(
                        f"Inconsistent channels or units for {name!r}; prepare them first"
                    )
                other = ChannelMoments(channels)
                other.count[:] = a["n"]
                other.mean = np.asarray(a["mean"], dtype=np.float64)
                other.m2 = np.asarray(a["m2"], dtype=np.float64)
                f["moments"].merge(other)
                if a["n"]:
                    f["min"] = (
                        np.asarray(a["min"]) if f["min"] is None else np.minimum(f["min"], a["min"])
                    )
                    f["max"] = (
                        np.asarray(a["max"]) if f["max"] is None else np.maximum(f["max"], a["max"])
                    )
        if not fields:
            raise ValueError("No fields selected for normalization; choose existing fields")
        norm: dict[str, Any] = {}
        for name, f in fields.items():
            moments = f["moments"]
            if moments.count[0] == 0:
                raise ValueError(f"No complete finite training rows for field {name!r}")
            if not np.isfinite(moments.mean).all() or not np.isfinite(moments.variance).all():
                raise ValueError(
                    f"Normalization moments overflow for {name!r}; rescale units first"
                )
            norm[name] = {
                "mean": moments.mean.tolist(),
                "std": np.sqrt(moments.variance).tolist(),
                "min": f["min"].tolist(),
                "max": f["max"].tolist(),
                "n": int(moments.count[0]),
                "units": f["units"],
            }
        if params.write:
            manifest.stats["normalization"] = {
                "algorithm": NORMALIZATION_ALGORITHM,
                "method": params.method,
                "split": params.split,
                "fields": norm,
                "samples": [o.sample_id for o in outcomes],
                "source_fields": {
                    o.sample_id: {
                        name: {
                            key: a[key]
                            for key in ("sha256", "rows_seen", "selected_steps", "dtype", "shape")
                        }
                        for name, a in o.stats.items()
                    }
                    for o in outcomes
                },
                "read_policy": {
                    "engine": params.read_engine,
                    "buffer_mib": params.read_buffer_mib,
                    "statistics_workspace_mib": params.statistics_workspace_mib,
                    "tensorstore_cache_mib": params.tensorstore_cache_mib,
                },
            }
        return [
            Change(
                target="manifest/stats/normalization",
                action="modify",
                detail={"fields": sorted(norm)},
            )
        ], {"normalization": norm}


def _natural(s: str) -> tuple[Any, ...]:
    import re

    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))
