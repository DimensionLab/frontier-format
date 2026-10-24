# SPDX-License-Identifier: Apache-2.0
"""Quality checks for a ``frontier-dataset`` (vocabulary: docs/spec/dataset-format.md).

Per-sample checks: ``nan_inf``, ``field_outliers`` (robust z-score of per-sample field statistics
against the dataset), ``degenerate_faces``, ``area_sum`` (triangle area sum vs STL area and vs the
native polygon areas), ``force_consistency`` (drag coefficient integrated from surface pressure
and wall shear stress vs the solver's CSV), ``duplicate_geometry``.
Dataset-level checks: ``parameter_coverage`` (empty bins / gaps in the normalised parameter
space), ``split_leakage``, plus field statistics and a recommended normalisation.

Steady surface samples take the path above. Every other v1.1 sample (mesh, grid, point cloud,
and any transient sample) goes through :mod:`frontier_data.quality_transient`
(``nan_inf_per_step``, ``temporal_continuity``, ``cfl``, ``stationarity``, ``spectra``,
``conservation``, ``temporal_subsampling``); datasets with such samples also get the
dataset-level ``trajectory_split_leakage`` and ``temporal_consistency`` checks and a top-level
``report["transient"]`` summary.

Each check yields ``{"check", "status": "pass|warn|fail|skip", "detail"}``. ``skip`` is used when
a check cannot run for a sample (e.g. no pressure field); it is not in the contract's vocabulary
but is more honest than a silent pass.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from frontier_data.dataset import Dataset, SampleMeta, write_json
from frontier_data.mesh import SurfaceMesh
from frontier_data.splits import TrajectoryFingerprint, find_leakage, find_trajectory_leakage

log = logging.getLogger(__name__)

Status = str  # "pass" | "warn" | "fail" | "skip"
ProgressFn = Callable[[str, int, int], None]

Z_WARN = 3.5
Z_FAIL = 6.0
AREA_WARN = 0.02
AREA_FAIL = 0.10
FORCE_WARN = 0.02
FORCE_FAIL = 0.05
DEGENERATE_FAIL_FRAC = 1e-3
# robust z-scores need enough samples before they can fail: in a parameter study of a dozen
# cases the median and MAD describe the cluster of similar cases, not the dataset
MIN_SAMPLES_FOR_FAIL = 30
COVERAGE_BINS = 10


@dataclass
class CheckResult:
    check: str
    status: Status
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.check, "status": self.status, "detail": self.detail}


@dataclass
class SampleStats:
    sample_id: str
    n_points: int
    n_faces: int
    area: float
    fields: dict[str, dict[str, float]]  # name -> {mean,std,min,max,p01,p99,n_nan,n_inf}
    checks: list[CheckResult]
    cd_integrated: float | None = None
    cd_pressure: float | None = None
    cd_reference: float | None = None
    transient: Any = None  # quality_transient.TransientSampleResult for non-steady-surface samples


def _status_from(value: float, warn: float, fail: float) -> Status:
    if not math.isfinite(value) or value >= fail:
        return "fail"
    if value >= warn:
        return "warn"
    return "pass"


def _worst(statuses: list[Status]) -> Status:
    order = {"pass": 0, "skip": 0, "warn": 1, "fail": 2}
    best = "pass"
    for s in statuses:
        if order[s] > order[best]:
            best = s
    return best


# -- per-sample checks ------------------------------------------------------------------


def field_stats(arr: np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "p01": math.nan,
            "p99": math.nan,
            "n_nan": float(np.isnan(a).sum()),
            "n_inf": float(np.isinf(a).sum()),
        }
    p01, p99 = np.percentile(finite, [1, 99])
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p01": float(p01),
        "p99": float(p99),
        "n_nan": float(np.isnan(a).sum()),
        "n_inf": float(np.isinf(a).sum()),
    }


def check_nan_inf(mesh: SurfaceMesh) -> CheckResult:
    bad: dict[str, int] = {}
    n = int((~np.isfinite(mesh.points)).sum())
    if n:
        bad["points"] = n
    for name, arr in {
        **mesh.cell_fields,
        **{f"point:{k}": v for k, v in mesh.point_fields.items()},
    }.items():
        n = int((~np.isfinite(arr)).sum())
        if n:
            bad[name] = n
    return CheckResult("nan_inf", "fail" if bad else "pass", {"non_finite_counts": bad})


def check_degenerate_faces(mesh: SurfaceMesh) -> CheckResult:
    mask = mesh.degenerate_faces()
    n_bad = int(mask.sum())
    oob = int(((mesh.faces < 0) | (mesh.faces >= mesh.n_points)).any(axis=1).sum())
    frac = n_bad / max(mesh.n_faces, 1)
    status: Status = "pass"
    if oob or frac >= DEGENERATE_FAIL_FRAC:
        status = "fail"
    elif n_bad:
        status = "warn"
    return CheckResult(
        "degenerate_faces",
        status,
        {"n_degenerate": n_bad, "fraction": frac, "n_out_of_bounds": oob, "n_faces": mesh.n_faces},
    )


def check_area_sum(mesh: SurfaceMesh, meta: SampleMeta) -> CheckResult:
    area = mesh.area
    geo_area = float(mesh.face_areas().sum())
    detail: dict[str, Any] = {"triangle_area_sum": area, "geometric_area_sum": geo_area}
    rel: list[float] = []
    if meta.mesh.stl_area:
        d = abs(area - meta.mesh.stl_area) / meta.mesh.stl_area
        detail["stl_area"] = meta.mesh.stl_area
        detail["rel_diff_vs_stl"] = d
        rel.append(d)
    if meta.mesh.native_area_sum:
        d = abs(area - meta.mesh.native_area_sum) / meta.mesh.native_area_sum
        detail["native_area_sum"] = meta.mesh.native_area_sum
        detail["rel_diff_vs_native"] = d
        rel.append(d)
    if not rel:
        return CheckResult("area_sum", "skip", {**detail, "reason": "no reference area"})
    return CheckResult("area_sum", _status_from(max(rel), AREA_WARN, AREA_FAIL), detail)


def integrate_drag(
    mesh: SurfaceMesh,
    pressure: np.ndarray,
    wss: np.ndarray | None,
    *,
    drag_dir: np.ndarray,
    rho: float,
    u_inf: float,
    a_ref: float,
    pressure_kind: str = "kinematic",
) -> tuple[float, float, float]:
    """Return (cd_total, cd_pressure, cd_viscous).

    ``pressure``/``wss`` are per-face values. For OpenFOAM incompressible output the pressure is
    kinematic (p/rho) and ``wallShearStress`` is the kinematic traction *exerted by the wall on the
    fluid*, hence the minus sign on the viscous term (verified against the AhmedML CSVs).
    """
    n = mesh.face_normals()
    a = mesh.areas().astype(np.float64)
    p = np.asarray(pressure, dtype=np.float64).reshape(-1)
    scale = rho if pressure_kind == "kinematic" else 1.0
    f_p = scale * float(np.sum(p * (n @ drag_dir) * a))
    f_v = 0.0
    if wss is not None:
        w = np.asarray(wss, dtype=np.float64).reshape(mesh.n_faces, -1)
        if w.shape[1] == 3:
            f_v = -scale * float(np.sum((w @ drag_dir) * a))
    q = 0.5 * rho * u_inf**2 * a_ref
    return (f_p + f_v) / q, f_p / q, f_v / q


def check_force_consistency(mesh: SurfaceMesh, meta: SampleMeta) -> CheckResult:
    cond = meta.conditions
    p_name = meta.field_roles.get("pressure")
    w_name = meta.field_roles.get("wall_shear_stress")
    cd_ref = meta.forces.get("cd")
    needed = ("u_inf", "rho", "reference_area")
    if p_name is None or cd_ref is None or any(k not in cond for k in needed):
        return CheckResult(
            "force_consistency",
            "skip",
            {
                "reason": "missing pressure field, reference "
                "Cd or reference conditions (u_inf, rho, reference_area)"
            },
        )
    if p_name in mesh.cell_fields:
        p = mesh.cell_fields[p_name]
    elif p_name in mesh.point_fields:
        p = mesh.point_to_cell(mesh.point_fields[p_name])
    else:
        return CheckResult("force_consistency", "skip", {"reason": f"field {p_name} not stored"})
    wss = None
    if w_name:
        if w_name in mesh.cell_fields:
            wss = mesh.cell_fields[w_name]
        elif w_name in mesh.point_fields:
            wss = mesh.point_to_cell(mesh.point_fields[w_name])
    drag_dir = np.asarray(cond.get("drag_direction", [1.0, 0.0, 0.0]), dtype=np.float64)
    cd, cd_p, cd_v = integrate_drag(
        mesh,
        p,
        wss,
        drag_dir=drag_dir,
        rho=float(cond["rho"]),
        u_inf=float(cond["u_inf"]),
        a_ref=float(cond["reference_area"]),
        pressure_kind=str(cond.get("pressure_kind", "kinematic")),
    )
    rel = abs(cd - float(cd_ref)) / max(abs(float(cd_ref)), 1e-12)
    detail = {
        "cd_reference": float(cd_ref),
        "cd_integrated": cd,
        "cd_pressure": cd_p,
        "cd_viscous": cd_v,
        "rel_err": rel,
        "viscous_included": wss is not None,
        "note": "pressure-only Cd is expected to be lower than the CSV value"
        if wss is None
        else "",
    }
    status = (
        _status_from(rel, FORCE_WARN, FORCE_FAIL)
        if wss is not None
        else ("pass" if cd_p <= float(cd_ref) * 1.05 else "warn")
    )
    return CheckResult("force_consistency", status, detail)


def geometry_hash(params: dict[str, float], ndigits: int = 6) -> str:
    key = json.dumps({k: round(float(v), ndigits) for k, v in sorted(params.items())})
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# -- dataset-level ----------------------------------------------------------------------


def robust_z(values: np.ndarray) -> np.ndarray:
    """Robust z-score (median / MAD, consistent with sigma for normal data)."""
    v = np.asarray(values, dtype=np.float64)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    if not np.isfinite(mad) or mad == 0:
        sd = np.nanstd(v)
        if not np.isfinite(sd) or sd == 0:
            return np.zeros_like(v)
        return (v - med) / sd
    return 0.6745 * (v - med) / mad


def check_parameter_coverage(
    params: dict[str, dict[str, float]], bins: int = COVERAGE_BINS
) -> CheckResult:
    names = sorted({k for p in params.values() for k in p})
    ids = list(params)
    if not names or len(ids) < 2:
        return CheckResult(
            "parameter_coverage", "skip", {"reason": "fewer than 2 samples or no parameters"}
        )
    detail: dict[str, Any] = {"n_samples": len(ids), "bins": bins, "parameters": {}}
    empty_total = 0
    normalised = np.zeros((len(ids), len(names)))
    for j, name in enumerate(names):
        col = np.array([float(params[i].get(name, np.nan)) for i in ids])
        lo, hi = np.nanmin(col), np.nanmax(col)
        span = hi - lo if hi > lo else 1.0
        x = (col - lo) / span
        normalised[:, j] = np.nan_to_num(x, nan=0.5)
        hist, _ = np.histogram(x[np.isfinite(x)], bins=bins, range=(0.0, 1.0))
        empty = [int(b) for b in np.flatnonzero(hist == 0)]
        empty_total += len(empty)
        detail["parameters"][name] = {
            "min": float(lo),
            "max": float(hi),
            "histogram": hist.tolist(),
            "empty_bins": empty,
        }
    # nearest-neighbour gaps in the unit hypercube (LHS-style coverage diagnostic)
    d = np.sqrt(((normalised[:, None, :] - normalised[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    detail["nn_distance"] = {
        "mean": float(nn.mean()),
        "max": float(nn.max()),
        "min": float(nn.min()),
    }
    detail["empty_bins_total"] = empty_total
    status: Status = "pass" if empty_total == 0 else "warn"
    return CheckResult("parameter_coverage", status, detail)


def check_split_leakage(
    splits: dict[str, list[str]], hashes: dict[str, str] | None = None
) -> CheckResult:
    if not splits:
        return CheckResult("split_leakage", "skip", {"reason": "no splits"})
    leaks = find_leakage(splits)
    detail: dict[str, Any] = {"id_overlap": leaks, "sizes": {k: len(v) for k, v in splits.items()}}
    status: Status = "fail" if leaks else "pass"
    if hashes:
        by_split = {k: {hashes[i] for i in v if i in hashes} for k, v in splits.items()}
        names = list(by_split)
        geo_leaks: dict[str, int] = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                n = len(by_split[a] & by_split[b])
                if n:
                    geo_leaks[f"{a}&{b}"] = n
        detail["duplicate_geometry_across_splits"] = geo_leaks
        if geo_leaks and status == "pass":
            status = "warn"
    return CheckResult("split_leakage", status, detail)


def check_trajectory_split_leakage(
    splits: dict[str, list[str]],
    groups: dict[str, str | None] | None = None,
    fingerprints: dict[str, TrajectoryFingerprint] | None = None,
    parameters: dict[str, dict[str, float]] | None = None,
) -> CheckResult:
    """Never split inside a trajectory: ``fail`` when samples of one ``group`` sit in different
    splits or when the content of two samples in different splits shows they are pieces of the
    same trajectory (see :func:`frontier_data.splits.find_trajectory_leakage`)."""
    if not splits:
        return CheckResult("trajectory_split_leakage", "skip", {"reason": "no splits"})
    detail = find_trajectory_leakage(splits, groups, fingerprints, parameters)
    status: Status = detail.pop("status")
    return CheckResult("trajectory_split_leakage", status, detail)


# -- driver -----------------------------------------------------------------------------


def _sample_pass(ds: Dataset, sid: str) -> SampleStats:
    meta = ds.meta(sid)
    mesh = ds.surface(sid, 0)
    checks = [
        check_nan_inf(mesh),
        check_degenerate_faces(mesh),
        check_area_sum(mesh, meta),
        check_force_consistency(mesh, meta),
    ]
    fields = {
        name: field_stats(arr) for name, arr in (mesh.cell_fields or mesh.point_fields).items()
    }
    fc = checks[-1].detail
    from frontier_data.quality_transient import check_predictable_fields

    checks.append(
        check_predictable_fields(sorted(n for n, st in fields.items() if st), None),
    )
    return SampleStats(
        sample_id=sid,
        n_points=mesh.n_points,
        n_faces=mesh.n_faces,
        area=mesh.area,
        fields=fields,
        checks=checks,
        cd_integrated=fc.get("cd_integrated"),
        cd_pressure=fc.get("cd_pressure"),
        cd_reference=fc.get("cd_reference"),
    )


MESH_DEGENERATE_WARN = 1e-4  # fraction of cells without measure
MESH_DEGENERATE_FAIL = 1e-2
FREESTREAM_WARN = 0.05  # relative difference between far-field speed and `conditions.u_inf`
FREESTREAM_FAIL = 0.25


def check_mesh_cells(points: np.ndarray, cells: np.ndarray) -> CheckResult:
    """Closure of a volume mesh with its points: every cell refers to existing points, no cell
    repeats a point, and (planar meshes) no cell has zero area. ``cells`` is padded with -1."""
    n = int(points.shape[0])
    valid = cells >= 0
    out_of_range = int(((cells >= n) & valid).sum() + (cells < -1).sum())
    if out_of_range:
        return CheckResult(
            "mesh_cells", "fail", {"out_of_range": out_of_range, "n_cells": len(cells)}
        )
    ordered = np.sort(np.where(valid, cells, -1), axis=1)
    repeated = ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any(axis=1)
    degenerate = repeated.copy()
    if points.shape[1] == 2 and len(cells):
        idx = np.where(valid, cells, cells[:, :1])  # padding repeats the first node: zero area
        xy = points[idx].astype(np.float64)
        nxt = np.roll(xy, -1, axis=1)
        area = 0.5 * np.abs((xy[..., 0] * nxt[..., 1] - nxt[..., 0] * xy[..., 1]).sum(axis=1))
        scale = float(np.median(area)) if len(area) else 0.0
        degenerate |= area <= 1e-12 * max(scale, 1e-300)
    frac = float(degenerate.mean()) if len(cells) else 0.0
    status = _status_from(frac, MESH_DEGENERATE_WARN, MESH_DEGENERATE_FAIL)
    return CheckResult(
        "mesh_cells",
        status,
        {"n_cells": len(cells), "degenerate": int(degenerate.sum()), "degenerate_frac": frac},
    )


def check_freestream(
    points: np.ndarray, velocity: np.ndarray | None, conditions: dict[str, Any]
) -> CheckResult:
    """The speed at the points farthest from the body against ``conditions.u_inf``: a unit or
    scaling error in the velocity field, or a wrong reference speed, shows here. Skipped when
    the sample has no velocity field or no reference speed."""
    u_inf = conditions.get("u_inf")
    if velocity is None or not isinstance(u_inf, int | float) or u_inf <= 0:
        return CheckResult("freestream_consistency", "skip", {"reason": "no velocity or u_inf"})
    r = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    far = r >= np.quantile(r, 0.98)
    speed = float(np.median(np.linalg.norm(velocity.reshape(len(points), -1)[far], axis=1)))
    rel = abs(speed - float(u_inf)) / float(u_inf)
    return CheckResult(
        "freestream_consistency",
        _status_from(rel, FREESTREAM_WARN, FREESTREAM_FAIL),
        {"far_field_speed": speed, "u_inf": float(u_inf), "rel_err": rel},
    )


#: points outside the declared bounds of a field (tolerance: 0.1 % of its span)
BOUNDS_TOL = 1e-3
BOUNDS_WARN = 1e-4
BOUNDS_FAIL = 1e-3


def check_field_bounds(sample: Any, bounds: dict[str, Any]) -> CheckResult:
    """Closure of a steady field with its boundary values: a field the sample declares
    bounded (``meta.extra.field_bounds = {name: [lo, hi]}``, e.g. a temperature between the
    ambient and the hottest wall in a flow without heat sources, by the maximum principle)
    stays inside them. A solver that did not converge, a unit slip or a mixed-up case
    shows as points outside. Skipped when nothing is declared."""
    names = set(sample.field_names())
    detail: dict[str, Any] = {}
    worst = 0.0
    for name, pair in sorted(bounds.items()):
        if name not in names or not isinstance(pair, list | tuple) or len(pair) != 2:
            continue
        lo, hi = float(pair[0]), float(pair[1])
        tol = BOUNDS_TOL * max(hi - lo, 1e-12)
        arr = np.asarray(sample.array(name)[...], float).reshape(-1)
        outside = float(((arr < lo - tol) | (arr > hi + tol)).mean()) if arr.size else 0.0
        detail[name] = {
            "bounds": [lo, hi],
            "min": float(arr.min()),
            "max": float(arr.max()),
            "outside_frac": outside,
        }
        worst = max(worst, outside)
    if not detail:
        return CheckResult("field_bounds", "skip", {"reason": "no declared field bounds"})
    return CheckResult("field_bounds", _status_from(worst, BOUNDS_WARN, BOUNDS_FAIL), detail)


def _declared_bounds(sample: Any) -> dict[str, Any]:
    try:
        extra = sample.meta.extra or {}
    except (OSError, ValueError, AttributeError):
        return {}
    bounds = extra.get("field_bounds") if isinstance(extra, dict) else None
    return bounds if isinstance(bounds, dict) else {}


def _steady_mesh_checks(sample: Any, conditions: dict[str, Any]) -> list[CheckResult]:
    points = np.asarray(sample.points())
    names = sample.field_names()
    velocity = np.asarray(sample.array("velocity")[...]) if "velocity" in names else None
    if velocity is not None and velocity.shape[0] != points.shape[0]:
        velocity = None  # a cell field: not compared here
    return [
        check_mesh_cells(points, np.asarray(sample.cells)),
        check_freestream(points, velocity, conditions),
    ]


def _is_steady_surface(ds: Dataset, sid: str) -> bool:
    e = ds.entry(sid)
    return e.sample_kind == "surface" and e.temporal != "transient"


def _generic_pass(ds: Dataset, sid: str, series_points: int) -> SampleStats:
    """Per-sample pass for everything that is not a steady surface (format v1.1)."""
    from frontier_data.quality_transient import transient_sample_pass

    entry = ds.entry(sid)
    sample = ds.sample(sid)
    group = entry.group
    if group is None:
        try:
            group = sample.meta.group
        except (OSError, ValueError):
            group = None
    res = transient_sample_pass(
        sample, group=group, parameters=dict(entry.parameters), series_points=series_points
    )
    checks = list(res.checks)
    area = 0.0
    if sample.kind == "surface":
        # the geometry of a transient surface is static: its mesh checks still apply
        mesh = ds.surface(sid, 0, fields=False)
        area = mesh.area
        checks += [
            check_degenerate_faces(mesh),
            check_area_sum(mesh, ds.meta(sid)),
            CheckResult(
                "force_consistency",
                "skip",
                {"reason": "transient surface: forces are not integrated per step"},
            ),
        ]
    if sample.kind == "mesh" and not res.is_transient:
        try:
            conditions = dict(sample.meta.conditions or {})
        except (OSError, ValueError, AttributeError):
            conditions = {}
        checks += _steady_mesh_checks(sample, conditions)
    if sample.kind in ("mesh", "point_cloud") and not res.is_transient:
        bounds = _declared_bounds(sample)
        if bounds:
            checks.append(check_field_bounds(sample, bounds))
    return SampleStats(
        sample_id=sid,
        n_points=entry.n_points,
        n_faces=entry.n_faces or entry.n_cells,
        area=area,
        fields=res.fields,
        checks=checks,
        transient=res,
    )


def series_points_for(ds: Dataset) -> int:
    """Length of the per-sample plot series in the report: shorter for datasets with many
    trajectories, to keep the report small. Part of the per-sample statistics, so a caller
    that computes them elsewhere must use the same value."""
    n_generic = sum(
        not (e.sample_kind == "surface" and e.temporal != "transient") for e in ds.manifest.samples
    )
    return 200 if n_generic <= 100 else 50


def compute_sample_stats(ds: Dataset, sid: str, *, series_points: int | None = None) -> SampleStats:
    """The per-sample pass of :func:`run_quality` for one sample: every check and statistic
    that needs only that sample. It reads the sample's arrays and nothing else, so samples can
    be processed in any order and in other processes; hand the results to ``run_quality`` as
    ``precomputed``."""
    if _is_steady_surface(ds, sid):
        return _sample_pass(ds, sid)
    if series_points is None:
        series_points = series_points_for(ds)
    return _generic_pass(ds, sid, series_points)


def run_quality(
    ds: Dataset,
    *,
    out_dir: Path | None = None,
    progress: ProgressFn | None = None,
    write_html: bool = True,
    precomputed: Mapping[str, SampleStats] | None = None,
    mean_outlier_action: Literal["fail", "warn"] = "fail",
    dataset_location: str | Path | None = None,
) -> dict[str, Any]:
    """Run every check, write ``quality/report.json`` (+ ``report.html``) and return the report.

    ``precomputed`` maps sample ids to the result of :func:`compute_sample_stats` on this
    dataset's current content (for example from a worker pool). Samples it does not cover are
    computed here; the dataset-level statistics are merged in ``ds.sample_ids`` order either
    way, so the report is the same as that of a serial run. The entries are copied, not
    modified.
    ``dataset_location`` records the final location when evaluating a private staging
    directory before atomic publication; numerical reads still use ``ds.root``.
    """
    t0 = time.perf_counter()
    ids = ds.sample_ids
    stats: dict[str, SampleStats] = {}
    # keep the report small for datasets with many trajectories
    series_points = series_points_for(ds)
    for k, sid in enumerate(ids):
        ready = precomputed.get(sid) if precomputed is not None else None
        if ready is not None:
            if ready.sample_id != sid:
                raise ValueError(
                    f"precomputed statistics for {sid!r} belong to sample {ready.sample_id!r}"
                )
            # the dataset-level steps below append checks and z-scores to the entry
            stats[sid] = copy.deepcopy(ready)
        else:
            stats[sid] = compute_sample_stats(ds, sid, series_points=series_points)
        if progress:
            progress(f"checked {sid}", k + 1, len(ids))

    # dataset-wide field statistics and outliers
    field_names = sorted({f for s in stats.values() for f in s.fields})
    field_summary: dict[str, Any] = {}
    for name in field_names:
        rows = [s.fields[name] for s in stats.values() if name in s.fields]
        means = np.array([r["mean"] for r in rows])
        stds = np.array([r["std"] for r in rows])
        field_summary[name] = {
            "mean": float(np.nanmean(means)),
            "std": float(np.sqrt(np.nanmean(stds**2) + np.nanvar(means))),
            "min": float(np.nanmin([r["min"] for r in rows])),
            "max": float(np.nanmax([r["max"] for r in rows])),
            "p01_median": float(np.nanmedian([r["p01"] for r in rows])),
            "p99_median": float(np.nanmedian([r["p99"] for r in rows])),
        }
    for name in field_names:
        sids = [s.sample_id for s in stats.values() if name in s.fields]
        for stat in ("mean", "max", "min"):
            z = robust_z(np.array([stats[i].fields[name][stat] for i in sids]))
            for i, zi in zip(sids, z, strict=True):
                stats[i].fields[name][f"z_{stat}"] = float(zi)
    # A sample fails only when its *mean* is far off and the dataset is large enough for the
    # median/MAD estimate to be trustworthy; outlying extrema (single cells) only warn.
    for s in stats.values():
        flagged: dict[str, dict[str, float]] = {}
        worst = 0.0
        worst_mean = 0.0
        for name, st in s.fields.items():
            zs = {k: v for k, v in st.items() if k.startswith("z_")}
            zmax = max((abs(v) for v in zs.values()), default=0.0)
            worst = max(worst, zmax)
            worst_mean = max(worst_mean, abs(zs.get("z_mean", 0.0)))
            if zmax >= Z_WARN:
                flagged[name] = zs
        status: Status = "pass"
        if worst >= Z_WARN:
            status = "warn"
        if (
            worst_mean >= Z_FAIL
            and len(stats) >= MIN_SAMPLES_FOR_FAIL
            and mean_outlier_action == "fail"
        ):
            status = "fail"
        s.checks.append(
            CheckResult(
                "field_outliers",
                status,
                {
                    "max_abs_z": worst,
                    "max_abs_z_mean": worst_mean,
                    "flagged": flagged,
                    "n_samples": len(stats),
                    "mean_outlier_action": mean_outlier_action,
                },
            )
        )

    # duplicate geometry
    params = ds.geometry_params()
    hashes = {sid: geometry_hash(p) for sid, p in params.items() if p}
    groups: dict[str, list[str]] = {}
    for sid, h in hashes.items():
        groups.setdefault(h, []).append(sid)
    for s in stats.values():
        h = hashes.get(s.sample_id)
        dups = [i for i in groups.get(h, []) if i != s.sample_id] if h else []
        s.checks.append(
            CheckResult(
                "duplicate_geometry",
                "fail" if dups else ("skip" if h is None else "pass"),
                {"hash": h, "duplicates_of": dups},
            )
        )

    dataset_checks = [
        check_parameter_coverage({k: v for k, v in params.items() if v}),
        check_split_leakage(ds.manifest.splits, hashes),
    ]
    generic = [s.transient for s in stats.values() if s.transient is not None]
    transient_summary: dict[str, Any] | None = None
    trajectory_of: dict[str, str | None] = {
        e.id: e.group for e in ds.manifest.samples if e.group is not None
    }
    trajectory_of.update({r.sample_id: r.group for r in generic if r.group is not None})
    if trajectory_of or any(r.is_transient for r in generic):
        dataset_checks.append(
            check_trajectory_split_leakage(
                ds.manifest.splits,
                trajectory_of,
                {r.sample_id: r.fingerprint for r in generic if r.fingerprint is not None},
                {r.sample_id: r.parameters for r in generic},
            )
        )
    if any(r.is_transient for r in generic):
        from frontier_data.quality_transient import check_temporal_consistency, summarize_transient

        dataset_checks.append(check_temporal_consistency(generic))
        units = ds.manifest.time.units if ds.manifest.time is not None else "s"
        transient_summary = summarize_transient(generic, dataset_checks, units)

    # normalisation recommendation
    bounds = np.array([list(ds.meta(i).mesh.bounds) for i in ids]) if ids else np.zeros((0, 6))
    centre = (
        ((bounds[:, 0::2] + bounds[:, 1::2]) / 2).mean(axis=0).tolist()
        if len(bounds)
        else [0, 0, 0]
    )
    extent = float((bounds[:, 1::2] - bounds[:, 0::2]).max()) if len(bounds) else 1.0
    normalisation = {
        "points": {"method": "center_scale", "center": centre, "scale": extent},
        "fields": {
            name: {"method": "zscore", "mean": v["mean"], "std": v["std"]}
            for name, v in field_summary.items()
        },
        "geometry_params": {
            name: {
                "method": "minmax",
                "min": float(np.nanmin([p[name] for p in params.values() if name in p])),
                "max": float(np.nanmax([p[name] for p in params.values() if name in p])),
            }
            for name in ds.manifest.geometry_params
            if any(name in p for p in params.values())
        },
    }

    sample_rows = []
    for sid in ids:
        s = stats[sid]
        status = _worst([c.status for c in s.checks])
        sample_rows.append(
            {
                "id": sid,
                "status": status,
                "n_points": s.n_points,
                "n_faces": s.n_faces,
                "area": s.area,
                "cd_reference": s.cd_reference,
                "cd_integrated": s.cd_integrated,
                "cd_pressure": s.cd_pressure,
                "fields": s.fields,
                "checks": [c.to_dict() for c in s.checks],
            }
        )
        if s.transient is not None:
            sample_rows[-1]["kind"] = s.transient.kind
            if s.transient.transient is not None:
                sample_rows[-1]["transient"] = s.transient.transient
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for r in sample_rows:
        counts[r["status"]] += 1
    ds_status = (
        _worst([r["status"] for r in sample_rows] + [c.status for c in dataset_checks])
        if sample_rows
        else "fail"
    )
    report = {
        "dataset": ds.name,
        "root": str(ds.root if dataset_location is None else dataset_location),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.perf_counter() - t0,
        "summary": {
            "n_samples": len(ids),
            "status": ds_status,
            **counts,
            "dataset_checks": {c.check: c.status for c in dataset_checks},
        },
        "dataset_checks": [c.to_dict() for c in dataset_checks],
        "field_statistics": field_summary,
        "recommended_normalization": normalisation,
        "samples": sample_rows,
    }
    if transient_summary is not None:
        # inserted before "samples" so the summary is near the top of report.json
        samples_block = report.pop("samples")
        report["transient"] = transient_summary
        report["samples"] = samples_block
    out_dir = out_dir or (ds.root / "quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "report.json", report)
    if write_html:
        from frontier_data.quality_report import render_html

        tmp = out_dir / "report.html.tmp"  # atomic: versioned files are read-only
        tmp.write_text(render_html(report))
        tmp.replace(out_dir / "report.html")
    ds.manifest.quality = "quality/report.json"
    ds.save_manifest()
    log.info("quality: %s (%s) in %.1fs", ds.name, ds_status, report["elapsed_s"])
    return report
