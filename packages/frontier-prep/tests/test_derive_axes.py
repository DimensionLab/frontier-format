# SPDX-License-Identifier: Apache-2.0
"""Force coefficients are closed in wind axes: a lifting body at incidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frontier_data import DatasetWriter
from frontier_prep.base import DatasetRef
from frontier_prep.benchmarks import A_REF, RHO, U_INF, _box, _forces
from frontier_prep.op import Context, get_op
from frontier_prep.ops.units import wind_axes


def run_op(op: str, root: Path, params: dict[str, Any]) -> Any:
    return get_op(op).run(DatasetRef(root), params, ctx=Context(workers=1))


def forces_of(root: Path, sid: str) -> dict[str, Any]:
    return json.loads((root / "samples" / sid / "forces.json").read_text())


def test_wind_axes_turn_drag_towards_lift_by_the_angle_of_attack():
    d, lt, s = [1.0, 0, 0], [0.0, 0, 1.0], [0.0, 1.0, 0]
    same = wind_axes(d, lt, s, alpha_deg=0.0, beta_deg=0.0)
    assert np.allclose(same["cd"], d) and np.allclose(same["cl"], lt)
    assert np.allclose(same["cs"], s)
    a = wind_axes(d, lt, s, alpha_deg=1.2856, beta_deg=0.0)
    # the share of a wing's lift that lands on the body x axis: sin(alpha) of it
    assert a["cd"][2] == pytest.approx(np.sin(np.deg2rad(1.2856)))
    assert a["cl"][0] == pytest.approx(-np.sin(np.deg2rad(1.2856)))
    assert np.allclose(a["cs"], s)
    ninety = wind_axes(d, lt, s, alpha_deg=90.0, beta_deg=0.0)
    assert np.allclose(ninety["cd"], lt, atol=1e-12)
    assert np.allclose(ninety["cl"], [-1, 0, 0], atol=1e-12)
    b = wind_axes(d, lt, s, alpha_deg=0.0, beta_deg=90.0)
    assert np.allclose(b["cd"], s, atol=1e-12)
    assert np.allclose(b["cs"], [-1, 0, 0], atol=1e-12)


def _body(root: Path, alpha: float) -> str:
    """One box reported at `alpha` degrees of incidence (the same box every time)."""
    m = _box(seed=0)
    w = DatasetWriter(root, name="wing-like", domain="cfd", license="CC-BY-4.0")
    cond = {"rho": RHO, "u_inf": U_INF, "reference_area": A_REF, "alpha": alpha}
    w.write_surface_sample("case_0", m, meta={"conditions": cond}, forces=_forces(m))
    w.finalize()
    return "case_0"


def test_closure_of_a_lifting_body_uses_the_wind_axes(tmp_path: Path) -> None:
    base = {"derive": ["cd", "cl"], "normals": "into_body", "wall_shear_stress": None}
    # the pressure force alone, in body axes: the reference the turned runs are held to
    flat = tmp_path / "flat"
    sid = _body(flat, 0.0)
    run_op("prep.scalars.derive", flat, {**base, "axes": "body"})
    derived = forces_of(flat, sid)["derived"]
    # without an angle the wind axes are the body axes
    res = run_op("prep.scalars.derive", flat, {**base, "compare_with": {}})
    same = forces_of(flat, sid)["derived"]
    assert same["axes"] == "wind" and np.allclose(same["directions"]["cd"], [1, 0, 0])
    assert same["cd"] == pytest.approx(derived["cd"], rel=1e-9)
    assert same["cl"] == pytest.approx(derived["cl"], rel=1e-9)
    assert not [a for a in res.assertions if a.check == "force_closure"]
    # the same body reported at 90 degrees of incidence: what the solver calls drag is the
    # body-axis lift, and the derivation follows the angle instead of failing the wing
    steep = tmp_path / "steep"
    sid = _body(steep, 90.0)
    run_op("prep.scalars.derive", steep, base)
    turned = forces_of(steep, sid)["derived"]
    assert turned["cd"] == pytest.approx(derived["cl"], abs=1e-9)
    assert turned["cl"] == pytest.approx(-derived["cd"], abs=1e-9)
    assert turned["cd_pressure"] == pytest.approx(turned["cd"], rel=1e-9)
    assert turned["cd_viscous"] == 0.0
    # body axes on request: the directions as given, whatever the angle
    run_op("prep.scalars.derive", steep, {**base, "axes": "body"})
    body = forces_of(steep, sid)["derived"]
    assert body["axes"] == "body" and body["cd"] == pytest.approx(derived["cd"], rel=1e-9)
