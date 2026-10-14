# SPDX-License-Identifier: Apache-2.0
"""A parametric shape from a few numbers, and typed cases as a dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from frontier_data.geometry import (
    AhmedBody,
    ParameterSpec,
    ahmed_surface,
    provider,
    providers,
    register,
)
from frontier_data.inputs import InputError, read_table, write_input


def _closed_and_outward(
    points: np.ndarray, faces: np.ndarray, centre: np.ndarray
) -> tuple[bool, int, float]:
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    tri = points[faces].astype(np.float64)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inward = int((np.einsum("ij,ij->i", nrm, tri.mean(axis=1) - centre) < 0).sum())
    volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)
    return bool((counts == 2).all()), inward, volume


def test_the_ahmed_body_is_a_closed_outward_surface_whose_volume_follows_its_parameters():
    body = AhmedBody()
    names = [s.name for s in body.parameters()]
    assert names[:3] == ["slant_angle", "rear_taper", "height"]
    box = body.surface({"slant_angle": 0.0, "rear_taper": 0.0})
    closed, inward, volume = _closed_and_outward(
        box.points, box.faces, np.array([0.522, 0.0, 0.05 + 0.144])
    )
    assert closed and inward == 0
    assert volume == pytest.approx(1.044 * 0.389 * 0.288, rel=1e-3)
    slanted = body.surface({"slant_angle": 25.0, "rear_taper": 0.1})
    closed, inward, less = _closed_and_outward(
        slanted.points, slanted.faces, np.array([0.522, 0.0, 0.194])
    )
    assert closed and inward == 0 and less < volume
    # values are clamped to the range, defaults fill the rest, and the surface is stable
    v = body.values({"slant_angle": 90.0})
    assert v["slant_angle"] == 45.0 and v["length"] == 1.044
    # the names AhmedML models condition on, in millimetres, derived from the body
    c = body.conditions({"slant_angle": 25.0})
    assert c["body-length"] == pytest.approx(1044.0) and c["slant-angle-degrees"] == 25.0
    assert c["slant-angle-length"] == pytest.approx(222.0)
    assert c["slant-angle-height"] == pytest.approx(222.0 * np.tan(np.radians(25.0)))
    assert c["slant-surface-length"] == pytest.approx(np.hypot(222.0, c["slant-angle-height"]))
    a, b = ahmed_surface(length=1.0, width=0.4, height=0.3, slant_deg=20.0, taper=0.05, ride=0.05)
    assert a.dtype == np.float32 and b.dtype == np.int32 and a[:, 2].min() == pytest.approx(0.05)


def test_providers_are_found_by_id_and_can_be_registered():
    assert "ahmed" in providers()
    assert provider("ahmed").length_units == "m"
    with pytest.raises(KeyError, match="no geometry provider"):
        provider("nurbs")

    class Flat:
        id, title, length_units = "flat", "Flat plate", "mm"

        def conditions(self, params: dict[str, float]) -> dict[str, float]:
            return {}

        def parameters(self):
            return [ParameterSpec("side", 1.0, 0.1, 10.0, "mm")]

        def surface(self, params, *, resolution=24):
            from frontier_data.mesh import SurfaceMesh

            s = params.get("side", 1.0)
            return SurfaceMesh(
                points=np.array([[0, 0, 0], [s, 0, 0], [0, s, 0]], np.float32),
                faces=np.array([[0, 1, 2]], np.int32),
            )

    register(Flat())
    assert provider("flat").surface({"side": 2.0}).n_faces == 1


def test_typed_cases_become_an_input_dataset(tmp_path: Path):
    table = tmp_path / "cases.csv"
    table.write_text("slant_angle,reynolds\n20,1e6\n30,1e6\n")
    assert read_table(table) == [
        {"slant_angle": 20.0, "reynolds": 1e6},
        {"slant_angle": 30.0, "reynolds": 1e6},
    ]
    contract = {"inputs": {"parameters": [{"name": "slant_angle"}, {"name": "reynolds"}]}}
    out = write_input(tmp_path / "inp", table, geometry="ahmed", contract=contract)
    m = json.loads((out / "manifest.json").read_text())
    assert m["source"] == {"type": "input", "geometry": "ahmed"}
    assert m["splits"] == {"train": [], "val": [], "test": ["case_0001", "case_0002"]}
    assert [s["geometry_params"]["slant_angle"] for s in m["samples"]] == [20.0, 30.0]
    assert m["samples"][0]["geometry_params"]["reynolds"] == 1e6
    assert m["samples"][0]["geometry_params"]["slant-angle-degrees"] == 20.0
    assert m["samples"][0]["geometry_params"]["body-height"] == pytest.approx(288.0)
    assert "slant_angle" in m["geometry_params"] and "reynolds" in m["geometry_params"]
    meta = json.loads((out / "samples" / "case_0001" / "meta.json").read_text())
    assert meta["extra"]["input"]["provider"] == "ahmed" and meta["sample_kind"] == "surface"
    # a parameter the model needs that the table does not give
    with pytest.raises(InputError, match="does not give the parameter reynolds"):
        write_input(tmp_path / "bad", {"slant_angle": 20.0}, geometry="ahmed", contract=contract)
    # one shape source at a time
    with pytest.raises(InputError, match="where the geometry comes from"):
        write_input(tmp_path / "none", {"slant_angle": 20.0})


def test_surface_files_and_dataset_samples_are_inputs_too(tmp_path: Path):
    import meshio

    stl = tmp_path / "wing.stl"
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    meshio.write(str(stl), meshio.Mesh(pts, [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))]))
    out = write_input(tmp_path / "files", {"aoa": 5.0}, geometry_files=[stl, stl])
    m = json.loads((out / "manifest.json").read_text())
    assert [s["id"] for s in m["samples"]] == ["wing", "wing-2"]
    assert all(s["geometry_params"] == {"aoa": 5.0} for s in m["samples"])
    # from another dataset: the samples' own parameters, overlaid
    again = write_input(
        tmp_path / "again", {"aoa": 7.0}, dataset=out, sample_ids=[m["samples"][0]["id"]]
    )
    s = json.loads((again / "manifest.json").read_text())["samples"]
    assert len(s) == 1 and s[0]["geometry_params"] == {"aoa": 7.0}
    with pytest.raises(InputError, match="rows of parameters for 2 geometries"):
        write_input(
            tmp_path / "mismatch",
            [{"aoa": 1.0}, {"aoa": 2.0}, {"aoa": 3.0}],
            geometry_files=[stl, stl],
        )
