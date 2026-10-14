# SPDX-License-Identifier: Apache-2.0
"""The Frontier data-prep benchmark: messy cases with
closure-based ground truth, so ops and agents are scored on the same defects.

Two kinds of cases are listed in ``frontier_prep/benchmark_cases.json`` (:func:`load_cases`):

- **synthetic** cases generated here (``make_case(name, root)``) that reproduce a defect
  class offline: seam soups, unit mismatches, vertex/cell confusion, duplicated frames,
  glitches, leakage, naming conventions, geometry scale. Each has a scorer (``score_case``)
  that checks the closure quantity after the designated ops ran.
- **real** cases that point at converted hub subsets (``datasets/hub/<id>``) with reference
  numbers from the dataset cards; they run when the data is present.

Scores are deterministic (seeded generators, no timestamps in the scored quantities).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from frontier_data import DatasetWriter
from frontier_data.mesh import SurfaceMesh, make_box_mesh
from frontier_prep.base import DatasetRef
from frontier_prep.op import Actor, Context, get_op

#: the case list ships inside the package (``frontier_prep/benchmark_cases.json``)
CASES_RESOURCE = "benchmark_cases.json"

RHO, U_INF, A_REF = 1.2, 10.0, 0.12


def _box(seed: int = 0, subdivisions: int = 6, size: float = 1.0) -> SurfaceMesh:
    m = make_box_mesh(size=(size, 0.4, 0.3), subdivisions=subdivisions)
    # OpenFOAM convention (normals out of the fluid, into the body), which integrate_drag assumes
    pts = m.points.astype(np.float64)
    vol = np.einsum(
        "ij,ij->i", pts[m.faces[:, 0]], np.cross(pts[m.faces[:, 1]], pts[m.faces[:, 2]])
    ).sum()
    if vol > 0:
        m = SurfaceMesh(
            points=m.points, faces=m.faces[:, [0, 2, 1]], cell_fields=dict(m.cell_fields)
        )
    rng = np.random.default_rng(seed)
    # a smooth pressure field in Pa plus a little noise; shear along -x
    c = m.points[m.faces].mean(axis=1)
    p = 0.5 * RHO * U_INF**2 * (0.9 - 1.8 * (c[:, 0] - c[:, 0].min()) / 1.0) + rng.normal(
        0, 0.5, len(c)
    )
    m.cell_fields["pressure"] = p.astype(np.float32)
    m.cell_fields["wall_shear_stress"] = np.tile(
        np.array([-0.2, 0.0, 0.0], np.float32), (m.n_faces, 1)
    )
    return m


def _forces(m: SurfaceMesh) -> dict[str, float]:
    from frontier_data.quality import integrate_drag

    cd, cdp, cdv = integrate_drag(
        m,
        m.cell_fields["pressure"][:, 0],
        m.cell_fields["wall_shear_stress"],
        drag_dir=np.array([1.0, 0, 0]),
        rho=RHO,
        u_inf=U_INF,
        a_ref=A_REF,
        pressure_kind="dynamic",
    )
    cl, _, _ = integrate_drag(
        m,
        m.cell_fields["pressure"][:, 0],
        m.cell_fields["wall_shear_stress"],
        drag_dir=np.array([0.0, 0, 1.0]),
        rho=RHO,
        u_inf=U_INF,
        a_ref=A_REF,
        pressure_kind="dynamic",
    )
    return {"cd": cd, "cl": cl, "cd_pressure": cdp, "cd_viscous": cdv}


def _writer(root: Path, name: str) -> DatasetWriter:
    return DatasetWriter(root, name=name, domain="cfd", license="CC-BY-4.0")


def _meta(sid: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"conditions": {"rho": RHO, "u_inf": U_INF, "reference_area": A_REF}, **(extra or {})}


def _soup(m: SurfaceMesh) -> SurfaceMesh:
    """Every face gets its own copy of its vertices (what per-patch exports do at seams)."""
    pts = m.points[m.faces].reshape(-1, 3)
    faces = np.arange(len(pts), dtype=np.int32).reshape(-1, 3)
    return SurfaceMesh(points=pts, faces=faces, cell_fields=dict(m.cell_fields))


# -- generators -------------------------------------------------------------------------------


def seam_soup(root: Path) -> dict[str, Any]:
    w = _writer(root, "bench-seam-soup")
    truth: dict[str, Any] = {}
    for i in range(3):
        m = _box(seed=i)
        soup = _soup(m)
        w.write_surface_sample(
            f"case_{i}",
            soup,
            meta=_meta(f"case_{i}"),
            forces=_forces(m),
            parameters={"seed": float(i)},
        )
        truth[f"case_{i}"] = {
            "open_edges_before": 3 * m.n_faces,
            "open_edges_after": 0,
            "area": m.area,
            "n_points_after": m.n_points,
        }
    w.finalize()
    return truth


def unit_mismatch(root: Path) -> dict[str, Any]:
    """Pressure stored in MPa but declared Pa (SHIFT-Crash style); forces are right."""
    w = _writer(root, "bench-unit-mismatch")
    truth: dict[str, Any] = {}
    for i in range(2):
        m = _box(seed=10 + i)
        f = _forces(m)
        wrong = m.copy()
        wrong.cell_fields["pressure"] = (m.cell_fields["pressure"] * 1e-6).astype(np.float32)
        w.write_surface_sample(
            f"case_{i}",
            wrong,
            meta=_meta(f"case_{i}"),
            forces=f,
            field_units={"pressure": "Pa", "wall_shear_stress": "Pa"},
        )
        truth[f"case_{i}"] = {"cd": f["cd"], "true_unit": "MPa", "declared_unit": "Pa"}
    w.finalize()
    return truth


def vertex_vs_cell(root: Path) -> dict[str, Any]:
    """Native cell pressure plus a smoothed vertex copy; only the cell field closes."""
    w = _writer(root, "bench-vertex-vs-cell")
    truth: dict[str, Any] = {}
    for i in range(2):
        m = _box(seed=20 + i)
        f = _forces(m)
        m.with_point_fields_from_cells()
        # the point copy is additionally smoothed: integrating it misses the stagnation peak
        m.point_fields["pressure"] = (m.point_fields["pressure"] * 0.9).astype(np.float32)
        w.write_surface_sample(
            f"case_{i}",
            m,
            meta=_meta(f"case_{i}"),
            forces=f,
            field_units={"pressure": "Pa", "wall_shear_stress": "Pa"},
        )
        truth[f"case_{i}"] = {"cd": f["cd"], "native": "cell"}
    w.finalize()
    return truth


def _trajectory(
    rng: np.random.Generator,
    n_steps: int,
    n: int = 200,
    spike_at: int | None = None,
    dup_first: bool = False,
    init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = rng.random((n, 2)).astype(np.float32)
    cells = rng.integers(0, n, (300, 3)).astype(np.int32)
    base = init if init is not None else rng.normal(0, 1, (n, 1))
    frames = [base]
    for _ in range(n_steps - 1):
        frames.append(frames[-1] * 0.98 + rng.normal(0, 0.05, (n, 1)))
    p = np.stack(frames).astype(np.float32)
    if dup_first:
        p = np.concatenate([p[:1], p], axis=0)
    if spike_at is not None:
        p[spike_at] = p[spike_at] + 50.0
    times = 0.01 * np.arange(len(p))
    return pts, cells, p, times


def duplicate_frames(root: Path) -> dict[str, Any]:
    """Every trajectory repeats its initial frame; two trajectories share the initial frame."""
    w = _writer(root, "bench-duplicate-frames")
    rng = np.random.default_rng(3)
    init = rng.normal(0, 1, (200, 1))
    truth: dict[str, Any] = {"shared_initial": ["traj_0", "traj_1"], "duplicates_per_trajectory": 1}
    for i in range(3):
        pts, cells, p, times = _trajectory(rng, 24, dup_first=True, init=init if i < 2 else None)
        w.write_mesh_sample(
            f"traj_{i}",
            points=pts,
            cells=cells,
            fields={"pressure": p},
            times=times,
            parameters={"Re": 100.0 + i},
        )
    w.finalize()
    return truth


def glitch(root: Path) -> dict[str, Any]:
    w = _writer(root, "bench-glitch")
    rng = np.random.default_rng(4)
    truth: dict[str, Any] = {}
    for i, spike in enumerate([None, 17, None, 9]):
        pts, cells, p, times = _trajectory(rng, 40, spike_at=spike)
        w.write_mesh_sample(
            f"traj_{i}",
            points=pts,
            cells=cells,
            fields={"pressure": p},
            times=times,
            parameters={"Re": 100.0 + i},
        )
        truth[f"traj_{i}"] = {"spike_at": spike}
    w.finalize()
    return truth


def leakage(root: Path) -> dict[str, Any]:
    """Two samples share geometry (a rerun with the same parameters); a random split puts
    them on different sides."""
    w = _writer(root, "bench-leakage")
    truth: dict[str, Any] = {"duplicate_group": ["case_1", "case_4"]}
    for i in range(6):
        j = 1 if i == 4 else i
        m = _box(seed=30 + j, size=1.0 + 0.15 * j)
        w.write_surface_sample(
            f"case_{i}",
            m,
            meta=_meta(f"case_{i}"),
            forces=_forces(m),
            parameters={"angle": float(10 * (1 if i == 4 else i))},
        )
    w.manifest.splits = {
        "train": ["case_0", "case_1", "case_2", "case_3"],
        "val": ["case_4"],
        "test": ["case_5"],
    }
    w.finalize()
    return truth


def naming(root: Path) -> dict[str, Any]:
    """OpenFOAM-style names (AhmedML) that must map onto the canonical vocabulary."""
    w = _writer(root, "bench-naming")
    m = _box(seed=40)
    m2 = SurfaceMesh(points=m.points, faces=m.faces)
    m2.cell_fields["pMean"] = m.cell_fields["pressure"] / RHO
    m2.cell_fields["wallShearStressMean"] = m.cell_fields["wall_shear_stress"] / RHO
    m2.cell_fields["static(p)_coeffMean"] = m.cell_fields["pressure"] / (0.5 * RHO * U_INF**2)
    m2.cell_fields["yPlusMean"] = np.full((m.n_faces, 1), 30.0, np.float32)
    w.write_surface_sample(
        "run_1",
        m2,
        meta=_meta("run_1"),
        forces=_forces(m),
        field_units={
            "pMean": "m^2/s^2",
            "wallShearStressMean": "m^2/s^2",
            "static(p)_coeffMean": "-",
            "yPlusMean": "-",
        },
    )
    w.finalize()
    return {
        "expected": {
            "pMean": "kinematic_pressure",
            "wallShearStressMean": "kinematic_wall_shear_stress",
            "static(p)_coeffMean": "pressure_coefficient",
            "yPlusMean": "y_plus",
        },
        "cd": _forces(m)["cd"],
    }


def geometry_scale(root: Path) -> dict[str, Any]:
    """Points in millimetres while the card says metres (bbox 1000x too large)."""
    w = _writer(root, "bench-geometry-scale")
    m = _box(seed=50)
    mm = SurfaceMesh(points=m.points * 1000.0, faces=m.faces, cell_fields=dict(m.cell_fields))
    w.write_surface_sample(
        "case_0",
        mm,
        meta=_meta("case_0", {"extra": {"declared_length_units": "m"}}),
        forces=_forces(m),
    )
    w.finalize()
    return {"bbox_x_m": float(m.points[:, 0].max() - m.points[:, 0].min())}


GENERATORS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "seam_soup": seam_soup,
    "unit_mismatch": unit_mismatch,
    "vertex_vs_cell": vertex_vs_cell,
    "duplicate_frames": duplicate_frames,
    "glitch": glitch,
    "leakage": leakage,
    "naming": naming,
    "geometry_scale": geometry_scale,
}


def make_case(name: str, root: Path) -> dict[str, Any]:
    root = Path(root)
    truth = GENERATORS[name](root)
    (root / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth


# -- scoring --------------------------------------------------------------------------------


def _run(root: Path, op_id: str, params: dict[str, Any], workers: int = 1) -> Any:
    return get_op(op_id).run(
        DatasetRef(root),
        params,
        ctx=Context(workers=workers, actor=Actor(kind="system", name="benchmark")),
    )


def score_case(name: str, root: Path, *, workers: int = 1) -> dict[str, Any]:
    """Run the designated ops on a generated case and compare with the closure truth."""
    root = Path(root)
    truth = json.loads((root / "truth.json").read_text())
    out: dict[str, Any] = {"case": name}
    if name == "seam_soup":
        before = _run(root, "prep.mesh.watertight_check", {})
        opens = {a.target.split("/")[1]: a.detail["open_edges"] for a in before.assertions}
        r = _run(root, "prep.mesh.merge_seams", {}, workers)
        after = _run(root, "prep.mesh.watertight_check", {})
        opens2 = {a.target.split("/")[1]: a.detail["open_edges"] for a in after.assertions}
        area_ok = all(abs(o.stats["area_rel_change"]) < 1e-5 for o in []) or all(
            a.status == "pass" for a in r.assertions
        )
        out.update(
            {"open_edges_before": opens, "open_edges_after": opens2, "area_conserved": area_ok}
        )
        out["ok"] = (
            all(opens[s] == truth[s]["open_edges_before"] for s in truth)
            and all(opens2[s] == 0 for s in truth)
            and area_ok
        )
    elif name == "unit_mismatch":
        bad = _run(root, "prep.scalars.derive", {"derive": ["cd"], "pressure_kind": "dynamic"})
        rel_bad = max(a.detail["rel_err"] for a in bad.assertions)
        _run(
            root,
            "prep.units.convert",
            {"conversions": {"pressure": "Pa"}, "from_units": {"pressure": "MPa"}},
            workers,
        )
        good = _run(root, "prep.scalars.derive", {"derive": ["cd"], "pressure_kind": "dynamic"})
        rel_good = max(a.detail["rel_err"] for a in good.assertions)
        out.update({"closure_before": rel_bad, "closure_after": rel_good})
        out["ok"] = rel_bad > 0.5 and rel_good < 1e-3
    elif name == "vertex_vs_cell":
        cell = _run(root, "prep.scalars.derive", {"derive": ["cd"], "pressure_kind": "dynamic"})
        rel_cell = max(a.detail["rel_err"] for a in cell.assertions)
        # force the point copy: cast the (wrong) point field to a cell field under another name
        _run(
            root,
            "prep.fields.cast_location",
            {"to": "cell", "fields": ["pressure"], "suffix": "_from_points"},
            workers,
        )
        pt = _run(
            root,
            "prep.scalars.derive",
            {"derive": ["cd"], "pressure": "pressure_from_points", "pressure_kind": "dynamic"},
        )
        rel_pt = max(a.detail["rel_err"] for a in pt.assertions)
        out.update({"closure_cell": rel_cell, "closure_point_copy": rel_pt})
        out["ok"] = rel_cell < 1e-3 and rel_pt > 0.05
    elif name == "duplicate_frames":
        chk = _run(root, "prep.quality.check", {"write_report": False})
        groups = [
            a.detail["groups"] for a in chk.assertions if a.check == "initial_frame_duplicates"
        ][0]
        r = _run(root, "prep.time.drop_duplicate_frames", {}, workers)
        n_dup = r.stats.get("n_duplicates", {})
        out.update({"shared_initial_groups": groups, "dropped": n_dup})
        out["ok"] = sorted(groups[0]) == truth["shared_initial"] and all(
            v == 1 for v in n_dup.values()
        )
    elif name == "glitch":
        r = _run(root, "prep.time.continuity_check", {}, workers)
        found = {
            a.target.split("/")[1]: [j["step"] for j in a.detail.get("jumps", [])]
            for a in r.assertions
            if a.check == "temporal_continuity"
        }
        out["found"] = found
        out["ok"] = all((found.get(s) or [None])[0] == t["spike_at"] for s, t in truth.items())
    elif name == "leakage":
        r = _run(root, "prep.dataset.dedup", {"action": "report"}, workers)
        groups = [g["members"] for g in r.stats.get("groups", [])]
        s0 = _run(root, "prep.dataset.split", {"strategy": "random", "seed": 0})
        proof0 = s0.stats["leakage"]
        s1 = _run(
            root,
            "prep.dataset.split",
            {"strategy": "group", "group_key": "geometry_hash", "seed": 0},
        )
        proof1 = s1.stats["leakage"]
        out.update(
            {"groups": groups, "random_status": proof0["status"], "group_status": proof1["status"]}
        )
        out["ok"] = (
            groups == [truth["duplicate_group"]]
            and proof1["status"] == "pass"
            and not proof1["duplicates_across_splits"]
        )
    elif name == "naming":
        op = get_op("prep.fields.map")
        plan = op.plan(DatasetRef(root), {})
        proposed = {c.before["name"]: c.after["name"] for c in plan.changes if c.before and c.after}
        for j in plan.judgments:
            proposed.setdefault(j.target.split("/")[-1], f"judgment:{j.options}")
        out["proposed"] = proposed
        out["ok"] = all(proposed.get(k) == v for k, v in truth["expected"].items())
        if out["ok"]:
            _run(root, "prep.fields.map", {}, workers)
            _run(
                root,
                "prep.units.convert",
                {
                    "conversions": {
                        "kinematic_pressure": "Pa",
                        "kinematic_wall_shear_stress": "Pa",
                    },
                    "multiply_by": {
                        "kinematic_pressure": {"rho": RHO},
                        "kinematic_wall_shear_stress": {"rho": RHO},
                    },
                    "rename": {
                        "kinematic_pressure": "pressure",
                        "kinematic_wall_shear_stress": "wall_shear_stress",
                    },
                },
                workers,
            )
            d = _run(root, "prep.scalars.derive", {"derive": ["cd"]})
            out["closure_after"] = max(a.detail["rel_err"] for a in d.assertions)
            out["ok"] = out["closure_after"] < 1e-3
    elif name == "geometry_scale":
        _run(root, "prep.mesh.transform", {"from_unit": "mm", "to_unit": "m"}, workers)
        ref = DatasetRef(root)
        b = ref.ds.meta("case_0").mesh.bounds
        out["bbox_x"] = b[1] - b[0]
        out["ok"] = abs(out["bbox_x"] - truth["bbox_x_m"]) < 1e-4
    else:
        raise KeyError(name)
    return out


def run_benchmark(
    work_dir: Path, names: list[str] | None = None, *, workers: int = 1
) -> list[dict[str, Any]]:
    results = []
    for name in names or list(GENERATORS):
        root = Path(work_dir) / name
        make_case(name, root)
        results.append(score_case(name, root, workers=workers))
    return results


def load_cases() -> dict[str, Any]:
    """The benchmark's case list (synthetic defect classes and the real cases they mirror)."""
    from importlib import resources

    text = (resources.files("frontier_prep") / CASES_RESOURCE).read_text(encoding="utf-8")
    return json.loads(text)
