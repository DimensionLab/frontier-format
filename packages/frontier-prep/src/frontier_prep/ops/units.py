# SPDX-License-Identifier: Apache-2.0
"""Units and derived scalars: ``prep.units.convert`` (pint, dimensional checks, reference
quantities such as density for kinematic -> dynamic pressure) and ``prep.scalars.derive``
(coefficients from forces with reference quantities)."""

from __future__ import annotations

import os
import shutil
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from frontier_data.dataset import Manifest, _compressors
from frontier_data.mesh import SurfaceMesh
from frontier_prep import kernels
from frontier_prep import vocabulary as vocab
from frontier_prep.base import BaseOp, DatasetRef, SampleOutcome
from frontier_prep.io import LOCATION_GROUP, SampleEditor
from frontier_prep.op import Assertion, Change, Judgment, OpWarning, Params, register


def scale_block(x: np.ndarray, factor: float) -> np.ndarray:
    """The element-wise conversion kernel: ``float32(float64(x) * factor)``; ``factor == 1``
    returns the block unchanged apart from the float32 cast of float inputs (what
    ``SampleEditor.write_field`` applies)."""
    if factor != 1.0:
        nk = kernels.native()
        if nk is not None and x.dtype in (np.float32, np.float64) and x.dtype.isnative:
            return nk.scale_block(np.ascontiguousarray(x), float(factor))
        return scale_block_py(x, factor)
    return x.astype(np.float32, copy=False) if x.dtype.kind == "f" else x


def scale_block_py(x: np.ndarray, factor: float) -> np.ndarray:
    """The numpy reference of the scaling in :func:`scale_block` (``factor != 1``)."""
    return (x.astype(np.float64) * factor).astype(np.float32)


def convert_field_chunked(
    ed: SampleEditor,
    name: str,
    location: str,
    factor: float,
    *,
    units: str | None,
    meta_extra: dict[str, Any] | None,
) -> tuple[int, int]:
    """Rewrite ``fields/<name>`` (or ``cell_fields/<name>``) as ``scale_block(x, factor)``
    one slab at a time instead of materialising the array. The destination has the layout
    ``write_field`` would give it (same chunks, dtype, codec, attributes), so the chunk files
    and therefore the content hashes are identical to the whole-array path. Slabs are a
    multiple of the destination chunk rows (every destination chunk is written once) and at
    least one source chunk. Returns ``(bytes_read, bytes_written)``."""
    import zarr

    src = ed.array(name, location)
    dims = list(ed.dims(name, location))
    shape = tuple(int(x) for x in src.shape)
    src_dtype = np.dtype(src.dtype)
    out_dtype = np.float32 if (factor != 1.0 or src_dtype.kind == "f") else src_dtype
    proto = np.broadcast_to(np.zeros((), dtype=out_dtype), shape)  # shape/dtype only
    chunks = SampleEditor._chunks(proto, dims)  # noqa: SLF001  (the write_field layout rule)
    tmp = ed.dir / f".{LOCATION_GROUP[location]}.{name.replace('/', '.')}.converting"
    if tmp.exists():
        shutil.rmtree(tmp)
    dst = zarr.create_array(
        str(tmp), shape=shape, dtype=out_dtype, chunks=chunks, compressors=_compressors()
    )
    n_read = n_written = 0
    if proto.size:
        src_rows = max(int(src.chunks[0]), 1)
        rows = max(chunks[0], src_rows)
        rows += (-rows) % chunks[0]
        for t0 in range(0, shape[0], rows):
            x = np.asarray(src[t0 : min(t0 + rows, shape[0])])
            y = scale_block(x, factor)
            dst[t0 : t0 + len(y)] = y
            n_read += x.nbytes
            n_written += y.nbytes
    dst.attrs["dims"] = dims
    target = ed.array_path(name, location)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.rename(tmp, target)
    ed.mark_lods_stale()  # level 0 was rewritten without write_field
    ed._record_field(name, proto, location, dims, units, None, meta_extra)  # noqa: SLF001
    ed.reopen()
    ed.bytes_read += n_read
    ed.bytes_written += n_written
    return n_read, n_written


class ConvertParams(Params):
    conversions: dict[str, str] = Field(
        description="field -> target unit (pint spelling: Pa, MPa, m/s, mm, ...)", min_length=1
    )
    from_units: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "field -> source unit when meta has none or the card is wrong (recorded as `declared`)."
        ),
    )
    multiply_by: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "field -> {reference: value} factors with units from `reference_units`, e.g. "
            "{'pressure': {'rho': 1.225}}."
        ),
    )
    reference_units: dict[str, str] = Field(
        default_factory=lambda: {"rho": "kg/m**3", "u_inf": "m/s", "length": "m", "area": "m**2"},
    )
    rename: dict[str, str] = Field(
        default_factory=dict,
        description="Rename after converting (e.g. kinematic -> dynamic form).",
    )
    geometry: str | None = Field(
        default=None,
        description=(
            "Also scale points by this length unit conversion, e.g. 'm' when points are in mm "
            "(uses from_units['points'])."
        ),
    )


@register
class ConvertUnits(BaseOp):
    id = "prep.units.convert"
    version = "1.0.0"
    category = "units"
    summary = (
        "Convert field units with dimensional checking (pint); reference quantities bridge "
        "kinematic and dynamic forms."
    )
    description = (
        "Each conversion is `field -> target unit`. The source unit is the field's recorded unit "
        "or `from_units[field]`. Optional `multiply_by[field]` factors (density, velocity, length) "
        "are checked dimensionally: source * factors must have the target's dimension. The "
        "declared "
        "unit, the factor and the method are recorded in meta.extra.fields."
    )
    when_to_use = (
        "Card and data disagree (MPa stored as Pa), mixed datasets (mm vs m), or an incompressible "
        "solver wrote p/rho and tau/rho (multiply by rho to get Pa)."
    )
    changes_fields = True
    changes_geometry = True
    Params = ConvertParams

    def _factor(self, field: str, src: str, dst: str, params: ConvertParams) -> tuple[float, str]:
        reg = vocab.ureg()
        s = vocab.normalize_unit(src)
        d = vocab.normalize_unit(dst)
        if s in ("", "unknown"):
            raise ValueError(f"{field}: source unit unknown; pass from_units")
        q = reg.Quantity(1.0, s)
        desc = f"{s}"
        for ref, value in params.multiply_by.get(field, {}).items():
            ru = params.reference_units.get(ref, "")
            q = q * reg.Quantity(float(value), ru)
            desc += f" * {ref}({value} {ru})"
        if q.dimensionality != reg.Quantity(1.0, d).dimensionality:
            raise ValueError(
                f"{field}: [{desc}] is {q.dimensionality}, target {d} is "
                f"{reg.Quantity(1.0, d).dimensionality}"
            )
        return float(q.to(d).magnitude), f"{desc} -> {d}"

    def plan_sample(self, ed: SampleEditor, params: ConvertParams) -> list[Change]:
        out = []
        for name, dst in params.conversions.items():
            if name not in ed.field_names():
                continue
            info = ed.field_info(name)
            src = params.from_units.get(name) or (info.units if info else "")
            try:
                f, desc = self._factor(name, src, dst, params)
            except ValueError as exc:
                out.append(
                    Change(
                        target=f"samples/{ed.id}/fields/{name}",
                        action="assert",
                        detail={"error": str(exc)},
                    )
                )
                continue
            out.append(
                Change(
                    target=ed.target(name),
                    action="modify",
                    before={"units": src},
                    after={"units": dst, "factor": f, "method": desc},
                )
            )
        if params.geometry and "points" in params.from_units:
            f, desc = self._factor("points", params.from_units["points"], params.geometry, params)
            out.append(
                Change(
                    target=f"samples/{ed.id}/points",
                    action="modify",
                    after={"factor": f, "method": desc},
                )
            )
        return out

    def plan_extra(
        self, ref: DatasetRef, params: ConvertParams
    ) -> tuple[list[OpWarning], list[Judgment]]:
        w: list[OpWarning] = []
        units = {f.name: f.units for f in ref.manifest.fields}
        for name, dst in params.conversions.items():
            src = params.from_units.get(name) or units.get(name, "")
            try:
                self._factor(name, src, dst, params)
            except ValueError as exc:
                w.append(
                    OpWarning(severity="fail", code="dimension", message=str(exc), target=name)
                )
        return w, []

    def apply_sample(self, ed: SampleEditor, params: ConvertParams, *, seed: int) -> SampleOutcome:
        changes = []
        for name, dst in params.conversions.items():
            if name not in ed.field_names():
                continue
            info = ed.field_info(name)
            src = params.from_units.get(name) or (info.units if info else "")
            f, desc = self._factor(name, src, dst, params)
            new_name = params.rename.get(name, name)
            if new_name != name:
                ed.rename_field(name, new_name)  # every location: a point copy follows
            # convert the native array and any derived copy at the other location, one
            # slab at a time (same files as a whole-array write_field)
            for loc in [x for x in ("cell", "point") if new_name in ed.field_names(x)]:
                convert_field_chunked(
                    ed,
                    new_name,
                    loc,
                    f,
                    units=dst,
                    meta_extra={
                        "declared_units": info.units if info else None,
                        "converted": desc,
                        "factor": f,
                        "source_name": name if new_name != name else None,
                    },
                )
                changes.append(
                    Change(
                        target=ed.target(new_name, loc),
                        action="modify",
                        before={"units": src, "name": name},
                        after={"units": dst, "factor": f, "name": new_name},
                    )
                )
        if params.geometry and "points" in params.from_units:
            f, desc = self._factor("points", params.from_units["points"], params.geometry, params)
            s = ed.sample
            if s.kind != "grid" and f != 1.0:
                pts = s.points()
                ed.write_root_array(
                    "points", (pts.astype(np.float64) * f).astype(np.float32), ["point", "space"]
                )
                g: Any = s.group
                if "points_t" in g:
                    ed.write_root_array(
                        "points_t",
                        (np.asarray(g["points_t"][...]).astype(np.float64) * f).astype(np.float32),
                        ["time", "point", "space"],
                    )
                if "cell_area" in g:
                    ed.write_root_array(
                        "cell_area",
                        (np.asarray(g["cell_area"][...]).astype(np.float64) * f * f).astype(
                            np.float32
                        ),
                        ["cell"],
                    )
                ed.set_attrs(
                    bounds=[float(x) for x in _bounds(pts * f)], length_units=params.geometry
                )

                def _m(m: Any) -> None:
                    m.mesh.bounds = [float(x) for x in _bounds(pts * f)]
                    m.mesh.area = float(m.mesh.area * f * f)
                    m.extra.setdefault("geometry", {})["converted"] = desc

                ed.update_meta(_m)
                changes.append(
                    Change(
                        target=f"samples/{ed.id}/points",
                        action="modify",
                        after={"factor": f, "method": desc},
                    )
                )
        return SampleOutcome(sample_id=ed.id, changes=changes)


def _at(ed: SampleEditor, name: str, loc: str, step: int) -> np.ndarray:
    """A field without its time axis (the given step of a transient array)."""
    if ed.sample.is_time_dependent(name, loc if loc == "cell" else None):
        return ed.field(name, step, location=loc)
    return ed.field(name, location=loc)


def _bounds(points: np.ndarray) -> list[float]:
    out = [0.0] * 6
    if points.size:
        lo, hi = points.min(axis=0), points.max(axis=0)
        for i in range(min(points.shape[1], 3)):
            out[2 * i], out[2 * i + 1] = float(lo[i]), float(hi[i])
    return out


# -- scalars.derive ----------------------------------------------------------------------------


def wind_axes(
    drag: list[float], lift: list[float], side: list[float], *, alpha_deg: float, beta_deg: float
) -> dict[str, np.ndarray]:
    """The drag, lift and side directions of the wind axes: the body directions turned by the
    angle of attack (drag towards lift, in their plane) and then by the sideslip (drag towards
    side). At zero angles they are the body directions. A lifting body at incidence carries a
    share of its lift along the body x axis, sin(alpha) times the lift: closing the drag in
    body axes would compare that against the solver's wind-axis drag and fail every wing."""
    d, lt, s = (np.asarray(v, np.float64) for v in (drag, lift, side))
    a, b = np.deg2rad(alpha_deg), np.deg2rad(beta_deg)
    d1 = np.cos(a) * d + np.sin(a) * lt
    l1 = -np.sin(a) * d + np.cos(a) * lt
    d2 = np.cos(b) * d1 + np.sin(b) * s
    s2 = -np.sin(b) * d1 + np.cos(b) * s
    return {"cd": d2, "cl": l1, "cs": s2}


class DeriveParams(Params):
    derive: list[Literal["cd", "cl", "cs", "forces", "cp", "cf"]] = Field(
        default_factory=lambda: ["cd", "cl", "forces"]
    )
    pressure: str = Field(default="pressure", description="Pressure field name.")
    wall_shear_stress: str | None = Field(default="wall_shear_stress")
    pressure_kind: Literal["dynamic", "kinematic", "auto"] = "auto"
    gauge: bool = Field(
        default=True, description="Subtract p_inf before integrating (needed on open surfaces)."
    )
    p_inf: float = 0.0
    rho: float | None = Field(
        default=None, description="Density; default from meta.conditions.rho."
    )
    u_inf: float | None = Field(
        default=None, description="Free-stream speed; default meta.conditions.u_inf."
    )
    reference_area: float | None = Field(
        default=None, description="A_ref; default meta.conditions.reference_area."
    )
    drag_dir: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0])
    lift_dir: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    side_dir: list[float] = Field(default_factory=lambda: [0.0, 1.0, 0.0])
    axes: Literal["wind", "body"] = Field(
        default="wind",
        description=(
            "wind: drag along the free stream and lift normal to it, the directions above "
            "turned by the angle of attack (meta.conditions.alpha, degrees, drag towards lift) "
            "and sideslip (beta, drag towards side); what a solver reports for a lifting body. "
            "body: the directions as given."
        ),
    )
    compare_with: dict[str, str] = Field(
        default_factory=lambda: {"cd": "cd", "cl": "cl"},
        description="derived key -> forces.json key to compare against (closure check).",
    )
    tolerance: float = Field(
        default=0.02, description="Relative closure tolerance for the assertion."
    )
    step: int | None = Field(
        default=None,
        description="Transient surfaces: time step to integrate (default: last step).",
    )
    normals: Literal["auto", "into_body", "outward"] = Field(
        default="auto",
        description=(
            "Face-normal convention: OpenFOAM patches point out of the fluid (into the body), "
            "CAD/Luminary exports point outward; auto decides from the signed volume and "
            "records the choice."
        ),
    )


def _drop_stale_point_copy(ed: SampleEditor, name: str) -> list[Change]:
    """A derived cell field was just rewritten. A point copy of the same name (written by the
    source or by an earlier cast) now describes the old values: it is removed rather than left
    stale; ``prep.fields.cast_location`` makes a fresh one when a point field is wanted."""
    import shutil

    path = ed.array_path(name, "point")
    if not path.exists():
        return []
    shutil.rmtree(path)
    ed.mark_lods_stale()
    ed.reopen()
    return [
        Change(
            target=f"samples/{ed.id}/fields/{name}",
            action="remove",
            detail={"reason": "stale point copy of the rewritten cell field"},
        )
    ]


@register
class DeriveScalars(BaseOp):
    id = "prep.scalars.derive"
    version = "1.0.0"
    category = "scalars"
    summary = (
        "Integrate surface pressure and shear into force coefficients (Cd/Cl/Cs), Cp and Cf, with "
        "reference quantities; closure assertion against reported values."
    )
    description = (
        "Cd = 2 (∫ p n·d dA - ∫ τ·d dA) / (rho U² A_ref). Kinematic fields are multiplied by rho. "
        "The result goes to forces.json under `derived` next to the reported values, and the "
        "relative disagreement is an assertion (`force_closure`)."
    )
    when_to_use = (
        "To verify reported coefficients (closure), to recover coefficients the source did not "
        "ship, or to add Cp/Cf fields for training."
    )
    changes_fields = True
    produces_assertions = True
    tolerance: ClassVar[dict[str, float] | None] = {"integral_rel": 1e-6}
    Params = DeriveParams

    def plan_samples(self, ref: DatasetRef, params: DeriveParams) -> list[str]:
        return [e.id for e in ref.manifest.samples if e.sample_kind == "surface"]

    def plan_sample(self, ed: SampleEditor, params: DeriveParams) -> list[Change]:
        if params.pressure not in ed.field_names():
            return []
        out = [
            Change(
                target=f"samples/{ed.id}/forces.json",
                action="modify",
                detail={"derive": params.derive},
            )
        ]
        if "cp" in params.derive:
            out.append(
                Change(target=f"samples/{ed.id}/cell_fields/pressure_coefficient", action="add")
            )
        if "cf" in params.derive and params.wall_shear_stress:
            out.append(
                Change(
                    target=f"samples/{ed.id}/cell_fields/skin_friction_coefficient", action="add"
                )
            )
        return out

    def _refs(self, ed: SampleEditor, params: DeriveParams) -> tuple[float, float, float]:
        cond = ed.meta.conditions
        rho = params.rho if params.rho is not None else cond.get("rho")
        u = params.u_inf if params.u_inf is not None else cond.get("u_inf")
        a = (
            params.reference_area
            if params.reference_area is not None
            else cond.get("reference_area")
        )
        missing = [k for k, v in (("rho", rho), ("u_inf", u), ("reference_area", a)) if v is None]
        if missing:
            raise ValueError(
                f"{ed.id}: missing reference quantities {missing} "
                "(pass as params or set meta.conditions)"
            )
        assert rho is not None and u is not None and a is not None
        return float(rho), float(u), float(a)

    def apply_sample(self, ed: SampleEditor, params: DeriveParams, *, seed: int) -> SampleOutcome:
        s = ed.sample
        if params.pressure not in ed.field_names():
            return SampleOutcome(sample_id=ed.id, skipped=True)
        try:
            rho, u_inf, a_ref = self._refs(ed, params)
        except ValueError as exc:
            return SampleOutcome(
                sample_id=ed.id,
                skipped=True,
                warnings=[OpWarning(code="missing-reference", message=str(exc), target=ed.id)],
            )
        mesh = SurfaceMesh(points=s.points(), faces=s.cells)
        g: Any = s.group
        if "cell_area" in g:
            mesh.cell_area = np.asarray(g["cell_area"][...], dtype=np.float32)
        loc = ed.location_of(params.pressure)
        step = params.step if params.step is not None else s.n_steps - 1
        p = _at(ed, params.pressure, loc, step).astype(np.float64).reshape(-1)
        if loc != "cell":
            p = mesh.point_to_cell(p).reshape(-1).astype(np.float64)
        info = ed.field_info(params.pressure)
        kind = params.pressure_kind
        if kind == "auto":
            exp = vocab.exponents_of(info.units) if info else None
            kind = "kinematic" if exp is not None and exp.M == 0 else "dynamic"
        scale = rho if kind == "kinematic" else 1.0
        if params.gauge:
            p = p - params.p_inf
        tau = None
        if params.wall_shear_stress and params.wall_shear_stress in ed.field_names():
            tloc = ed.location_of(params.wall_shear_stress)
            tau = _at(ed, params.wall_shear_stress, tloc, step).astype(np.float64)
            if tloc != "cell":
                tau = mesh.point_to_cell(tau).astype(np.float64)
            tau = tau.reshape(mesh.n_faces, -1)
            if tau.shape[1] != 3:
                tau = None
        n = mesh.face_normals()
        a = mesh.areas().astype(np.float64)
        q = 0.5 * rho * u_inf**2
        convention = params.normals
        if convention == "auto":
            pts = s.points().astype(np.float64)
            vol = float(
                np.einsum(
                    "ij,ij->i",
                    pts[mesh.faces[:, 0]],
                    np.cross(pts[mesh.faces[:, 1]], pts[mesh.faces[:, 2]]),
                ).sum()
                / 6.0
            )
            convention = "outward" if vol > 0 else "into_body"
        sign = -1.0 if convention == "outward" else 1.0
        scale = scale * sign
        # fixed-order float64 sums (np.sum pairwise on a contiguous array is deterministic)
        f_p = scale * np.array([np.sum(p * n[:, i] * a) for i in range(3)])
        f_v = (
            -scale * np.array([np.sum(tau[:, i] * a) for i in range(3)])
            if tau is not None
            else np.zeros(3)
        )
        f = f_p + f_v
        derived: dict[str, Any] = {
            "method": "surface_integral",
            "normals": convention,
            "pressure_kind": kind,
            "gauge": params.gauge,
            "p_inf": params.p_inf,
            "rho": rho,
            "u_inf": u_inf,
            "reference_area": a_ref,
            "force_pressure": f_p.tolist(),
            "force_viscous": f_v.tolist(),
            "force": f.tolist(),
        }
        wind = params.axes == "wind"
        cond = ed.meta.conditions
        dirs = wind_axes(
            params.drag_dir,
            params.lift_dir,
            params.side_dir,
            alpha_deg=float(cond.get("alpha") or 0.0) if wind else 0.0,
            beta_deg=float(cond.get("beta") or 0.0) if wind else 0.0,
        )
        derived["axes"] = params.axes
        derived["directions"] = {k: [float(x) for x in v] for k, v in dirs.items()}
        for key in ("cd", "cl", "cs"):
            if key in params.derive:
                d = dirs[key]
                derived[key] = float(f @ d) / (q * a_ref)
                derived[key + "_pressure"] = float(f_p @ d) / (q * a_ref)
                derived[key + "_viscous"] = float(f_v @ d) / (q * a_ref)
        forces = ed.forces()
        assertions = []
        for dk, rk in params.compare_with.items():
            if (
                dk in derived
                and rk in forces
                and isinstance(forces[rk], int | float)
                and forces[rk] != 0
            ):
                rel = abs(derived[dk] - float(forces[rk])) / abs(float(forces[rk]))
                status = (
                    "pass"
                    if rel <= params.tolerance
                    else ("warn" if rel <= 2.5 * params.tolerance else "fail")
                )
                assertions.append(
                    Assertion(
                        check="force_closure",
                        status=status,
                        target=f"samples/{ed.id}/{dk}",
                        detail={
                            "derived": derived[dk],
                            "reported": float(forces[rk]),
                            "rel_err": rel,
                            "tolerance": params.tolerance,
                            "pressure_kind": kind,
                        },
                    )
                )
        forces["derived"] = derived
        ed.save_forces(forces)
        changes = [
            Change(
                target=f"samples/{ed.id}/forces.json",
                action="modify",
                after={k: derived[k] for k in ("cd", "cl", "cs") if k in derived},
            )
        ]
        if "cp" in params.derive:
            cp = (p * scale / q).astype(np.float32)
            ed.write_field(
                "pressure_coefficient",
                cp[:, None],
                location="cell",
                units="-",
                meta_extra={"derived_from": params.pressure, "q": q, "p_inf": params.p_inf},
            )
            changes.append(
                Change(target=f"samples/{ed.id}/cell_fields/pressure_coefficient", action="add")
            )
            changes += _drop_stale_point_copy(ed, "pressure_coefficient")
        if "cf" in params.derive and tau is not None:
            cf = (np.linalg.norm(tau, axis=1) * scale / q).astype(np.float32)
            ed.write_field(
                "skin_friction_coefficient",
                cf[:, None],
                location="cell",
                units="-",
                meta_extra={"derived_from": params.wall_shear_stress, "q": q},
            )
            changes.append(
                Change(
                    target=f"samples/{ed.id}/cell_fields/skin_friction_coefficient", action="add"
                )
            )
            changes += _drop_stale_point_copy(ed, "skin_friction_coefficient")
        ed.update_meta(
            lambda m: m.forces.update(
                {"derived": {k: derived[k] for k in ("cd", "cl", "cs") if k in derived}}
            )
        )
        return SampleOutcome(
            sample_id=ed.id,
            changes=changes,
            assertions=assertions,
            stats={
                "derived": {
                    k: derived[k] for k in ("cd", "cl", "cs", "cd_pressure") if k in derived
                }
            },
        )

    def apply_manifest(
        self,
        ref: DatasetRef,
        manifest: Manifest,
        params: DeriveParams,
        outcomes: list[SampleOutcome],
    ) -> tuple[list[Change], dict[str, Any]]:
        for o in outcomes:
            d = o.stats.get("derived")
            if not d:
                continue
            for e in manifest.samples:
                if e.id == o.sample_id:
                    e.forces.update({f"{k}_derived": float(v) for k, v in d.items()})
        return [], {}
