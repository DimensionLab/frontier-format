# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from frontier_data import (
    Dataset,
    DatasetWriter,
    validate_dataset,
    write_grid_sample,
    write_mesh_sample,
)
from frontier_data.mesh import make_box_mesh
from frontier_data.synthetic import (
    advection_diffusion_2d,
    spectral_vorticity_2d,
    vortex_street_mesh,
    write_synthetic_dataset,
)


def test_grid_transient_roundtrip_and_lazy_steps(tmp_path: Path) -> None:
    d = spectral_vorticity_2d(n=16, n_steps=6, dt=0.1, seed=1)
    w = DatasetWriter(tmp_path, "g", license="CC0-1.0", citation="x", hub_id="g")
    e = w.write_grid_sample(
        "a",
        fields={"vorticity": d["vorticity"][..., 0], "velocity": d["velocity"]},
        times=d["times"],
        spacing=d["spacing"],
        boundary_conditions={"x": "periodic", "y": "periodic"},
        static={"mask": np.ones((16, 16), dtype=bool)},
        parameters={"nu": 1e-3},
    )
    m = w.finalize()
    assert (e.sample_kind, e.temporal, e.n_steps, e.shape) == ("grid", "transient", 6, [16, 16])
    assert (m.minor, m.sample_kind, m.temporal) == (1, "grid", "transient")
    assert m.time is not None and m.time.n_steps == 6 and m.time.dt == pytest.approx(0.1)
    assert m.parameters == ["nu"]

    s = Dataset(tmp_path).sample("a")
    assert s.kind == "grid" and s.is_transient and s.n_steps == 6
    assert s.dims("vorticity") == ["time", "x", "y", "component"]
    assert s.array("vorticity").chunks[0] == 1  # one step per chunk
    assert s.field("vorticity", t=2).shape == (16, 16, 1)
    assert s.field("velocity", t=slice(1, 4)).shape == (3, 16, 16, 2)
    assert s.field("velocity", t=[0, 5]).shape == (2, 16, 16, 2)
    np.testing.assert_allclose(s.field("velocity")[3], d["velocity"][3])
    assert s.static("mask").shape == (16, 16) and s.static_names == ["mask"]
    assert s.spacing == pytest.approx(d["spacing"]) and s.boundary_conditions["x"] == "periodic"
    assert s.times is not None and s.times.dtype == np.float64
    assert s.points().shape == (256, 2)
    assert s.meta.parameters == {"nu": 1e-3} and s.meta.time is not None
    rep = validate_dataset(tmp_path)
    assert rep.ok and not rep.warnings, rep.to_dict()


def test_grid_steady_3d(tmp_path: Path) -> None:
    w = DatasetWriter(tmp_path, "g3")
    w.write_grid_sample("a", fields={"T": np.zeros((4, 5, 6))}, ndim=3, origin=[0, 0, 0])
    m = w.finalize()
    s = Dataset(tmp_path).sample("a")
    assert m.temporal == "steady" and m.time is None
    assert s.dims("T") == ["x", "y", "z", "component"] and s.field("T", t=3).shape == (4, 5, 6, 1)
    assert validate_dataset(tmp_path).ok


def test_mesh_transient_roundtrip(tmp_path: Path) -> None:
    d = vortex_street_mesh(n_target=200, n_steps=5)
    n = len(d["points"])
    moving = np.broadcast_to(d["points"], (5, n, 2)) + 0.0
    cells = np.concatenate([d["cells"], np.full((len(d["cells"]), 1), -1)], axis=1)
    cells[0] = [0, 1, 2, 3]  # one quad
    e = write_mesh_sample(
        tmp_path,
        "m",
        points=d["points"],
        cells=cells,
        fields={"velocity": d["velocity"], "pressure": d["pressure"][..., 0]},
        cell_fields={"q": np.zeros((5, len(cells)))},
        static={"node_type": d["node_type"]},
        points_t=moving,
        times=d["times"],
    )
    assert e.n_cells == len(cells) and e.n_points == n
    w = DatasetWriter(tmp_path, "m")
    w.add_entry(e)
    w.finalize()
    s = Dataset(tmp_path).sample("m")
    assert s.dims("velocity") == ["time", "point", "component"]
    assert s.dims("q") == ["time", "cell", "component"]
    assert s.field("pressure", t=1).shape == (n, 1)
    assert s.points(t=3).shape == (n, 2) and s.has_moving_points
    assert set(np.unique(s.cell_types)) == {5, 9}
    assert s.static("node_type").dtype.kind == "i"
    assert s.triangles().shape[1] == 3
    assert validate_dataset(tmp_path).ok


def test_writer_rejects_bad_shapes(tmp_path: Path) -> None:
    pts = np.zeros((4, 2), dtype=np.float32)
    cells = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="steps"):
        write_mesh_sample(
            tmp_path, "a", points=pts, cells=cells, fields={"u": np.zeros((3, 4))}, times=[0, 1]
        )
    with pytest.raises(ValueError, match="increasing"):
        write_mesh_sample(tmp_path, "a", points=pts, cells=cells, times=[0, 0])
    with pytest.raises(ValueError, match="outside"):
        write_mesh_sample(tmp_path, "a", points=pts, cells=np.array([[0, 1, 9]]))
    with pytest.raises(ValueError, match="shape"):
        write_grid_sample(
            tmp_path, "g", fields={"a": np.zeros((4, 4)), "b": np.zeros((4, 5))}, ndim=2
        )


@pytest.mark.parametrize("kind", ["grid", "grid_advection", "mesh", "surface", "point_cloud"])
def test_synthetic_datasets_validate(tmp_path: Path, kind: str) -> None:
    ds = write_synthetic_dataset(tmp_path / kind, kind, n_samples=4, n_steps=6, size=None)
    assert ds.is_transient and len(ds) == 4
    assert set(ds.manifest.splits) == {"train", "val", "test"}
    rep = validate_dataset(ds.root)
    assert rep.ok, rep.to_dict()
    s = ds.sample(ds.sample_ids[0])
    name = s.field_names()[0]
    assert s.field(name, t=0).shape == s.field(name, t=slice(0, 2)).shape[1:]
    assert np.isfinite(s.field(name)).all()


def test_transient_surface_has_lod_fields_and_v1_reader_still_works(tmp_path: Path) -> None:
    from frontier_data.lod import build_lods

    mesh = make_box_mesh(subdivisions=10)
    mesh.point_fields["thickness"] = np.ones(mesh.n_points)
    lods, _ = build_lods(mesh, levels=[0, 3], targets={3: 200})
    times = np.linspace(0, 1, 4)
    p = np.random.default_rng(0).normal(size=(4, mesh.n_points))
    w = DatasetWriter(tmp_path, "s")
    w.write_surface_sample("a", lods, fields={"p": p}, times=times, forces={"cd": 0.3})
    w.finalize()
    ds = Dataset(tmp_path)
    s = ds.sample("a")
    assert s.field("p", t=1, lod=3).shape == (lods[3].n_points, 1)
    assert not s.is_time_dependent("thickness") and s.is_time_dependent("p")
    surf = ds.surface("a")  # v1 API: steady fields only
    assert "thickness" in surf.point_fields and "p" not in surf.point_fields
    assert ds.forces("a")["cd"] == 0.3
    assert validate_dataset(tmp_path).ok


def test_steady_surface_stays_v1_compatible(tmp_path: Path) -> None:
    mesh = make_box_mesh(subdivisions=4)
    mesh.cell_fields["p"] = np.ones(mesh.n_faces)
    w = DatasetWriter(tmp_path, "s")
    w.write_surface_sample("a", mesh, forces={"cd": 0.1}, parameters={"Re": 1e5})
    m = w.finalize()
    assert m.temporal == "steady" and m.sample_kind == "surface" and m.parameters == ["Re"]
    assert Dataset(tmp_path).surface("a").cell_fields["p"].shape == (mesh.n_faces, 1)
    assert validate_dataset(tmp_path).ok


def test_v1_manifest_without_minor_is_read_and_validated(tmp_path: Path) -> None:
    mesh = make_box_mesh(subdivisions=3)
    w = DatasetWriter(tmp_path, "old")
    w.write_surface_sample("a", mesh)
    w.finalize()
    raw = json.loads((tmp_path / "manifest.json").read_text())
    for k in ("minor", "sample_kind", "temporal", "time", "domain", "license", "citation"):
        raw.pop(k, None)
    for s in raw["samples"]:
        for k in ("sample_kind", "temporal", "n_steps", "n_cells", "shape", "parameters", "group"):
            s.pop(k, None)
    (tmp_path / "manifest.json").write_text(json.dumps(raw))
    g: Any = zarr.open_group(str(tmp_path / "samples/a/surface.zarr"), mode="r+")
    ds = Dataset(tmp_path)
    assert ds.manifest.minor == 0 and not ds.is_transient
    del g["fields"]["p"].attrs["dims"]  # v1 stores carry no dims: the default applies
    assert ds.sample("a").dims("p") == ["point", "component"]
    assert validate_dataset(tmp_path).ok
    assert g.attrs["temporal"] == "steady"


def test_validator_catches_contract_violations(tmp_path: Path) -> None:
    ds = write_synthetic_dataset(tmp_path, "grid", n_samples=3, n_steps=5, size=8)
    sid = ds.sample_ids[0]
    g: Any = zarr.open_group(str(ds.sample_dir(sid) / "grid.zarr"), mode="r+")
    del g["fields"]["velocity"].attrs["dims"]
    g["fields"]["vorticity"].attrs["dims"] = ["x", "time", "y", "component"]
    ds.manifest.splits["val"] = list(ds.manifest.splits["train"])
    ds.manifest.domain = "astrology"
    ds.save_manifest()
    (ds.sample_dir(ds.sample_ids[1]) / "mesh.zarr").mkdir()
    rep = validate_dataset(tmp_path)
    msgs = " | ".join(i.message for i in rep.errors)
    assert not rep.ok
    assert "missing `dims`" in msgs and "time must be the first axis" in msgs
    assert "is in splits" in msgs and "domain" in msgs and "exactly one of" in msgs


def test_advection_diffusion_is_exact_shift() -> None:
    d = advection_diffusion_2d(n=32, n_steps=3, dt=2 * np.pi / 32, velocity=(1.0, 0.0), kappa=0.0)
    c = d["concentration"][..., 0]
    np.testing.assert_allclose(np.roll(c[0], 1, axis=0), c[1], atol=1e-4)


def test_cli_validate_and_synth(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from frontier_data.cli import app

    r = CliRunner().invoke(
        app,
        [
            "synth",
            str(tmp_path / "d"),
            "--kind",
            "mesh",
            "--samples",
            "3",
            "--steps",
            "4",
            "--size",
            "150",
        ],
    )
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(app, ["validate", str(tmp_path / "d"), "--json"])
    assert r.exit_code == 0 and json.loads(r.output)["ok"] is True


def test_transient_predictions_roundtrip(tmp_path: Path) -> None:
    from frontier_data.arrays import Sample
    from frontier_data.predictions import (
        is_transient_predictions,
        list_transient_predictions,
        read_transient_prediction,
        write_predictions_summary,
        write_transient_prediction,
    )

    rng = np.random.default_rng(0)
    times = 0.1 * np.arange(5)
    true = {"u": rng.normal(size=(5, 8, 8, 2)), "w": rng.normal(size=(5, 8, 8, 1))}
    pred = {k: v + 0.1 for k, v in true.items()}
    ens = {k: np.stack([v, v + 0.2]) for k, v in true.items()}
    write_transient_prediction(
        tmp_path,
        "a",
        times=times,
        pred=pred,
        true=true,
        kind="grid",
        spacing=[0.1, 0.1],
        ensemble=ens,
        info={"split": "test", "history": 1},
    )
    d = vortex_street_mesh(n_target=120, n_steps=4)
    write_transient_prediction(
        tmp_path,
        "m",
        times=d["times"],
        pred={"velocity": d["velocity"]},
        true={"velocity": d["velocity"]},
        kind="mesh",
        points=d["points"],
        cells=d["cells"],
    )
    write_predictions_summary(tmp_path, ["a", "m"], {"train_horizon": 4})
    assert is_transient_predictions(tmp_path) and list_transient_predictions(tmp_path) == ["a", "m"]
    r = read_transient_prediction(tmp_path, "a")
    assert r.field_names == ["u", "w"] and r.pred["u"].shape == (5, 8, 8, 2)
    assert r.ensemble is not None and r.ensemble["w"].shape == (2, 5, 8, 8, 1)
    assert r.info["split"] == "test" and r.dt == pytest.approx(0.1)
    s = Sample.open_store(tmp_path / "a.zarr")
    assert s.kind == "grid" and s.field_names() == [
        "u_err",
        "u_pred",
        "u_std",
        "u_true",
        "w_err",
        "w_pred",
        "w_std",
        "w_true",
    ]
    assert s.attrs["sample_kind"] == "grid"
    np.testing.assert_allclose(s.field("u_err", t=2), 0.1, atol=1e-6)
    assert s.dims("w_pred") == ["time", "x", "y", "component"]
    sm = Sample.open_store(tmp_path / "m.zarr")
    assert sm.kind == "mesh" and sm.field("velocity_true", t=1).shape == (len(d["points"]), 2)


def test_signals_cell_grids_and_appender(tmp_path: Path) -> None:
    import tarfile

    from frontier_data.predictions import pack_prediction_store, write_transient_prediction

    w = DatasetWriter(tmp_path, "s")
    times = 0.1 * np.arange(6)
    w.write_grid_sample(
        "g",
        fields={"rho": np.zeros((6, 4, 5))},
        times=times,
        spacing=[1.0, 1.0],
        origin=[0.5, 0.5],
        location="cell",
        signals={"cd": np.arange(6.0), "f": np.ones((6, 3))},
    )
    d = vortex_street_mesh(n_target=100, n_steps=5)
    with w.append_sample(
        "m",
        kind="mesh",
        points=d["points"],
        cells=d["cells"],
        static={"node_type": d["node_type"]},
        parameters={"Re": 100.0},
    ) as app:
        for i, t in enumerate(d["times"]):
            app.append(
                float(t),
                fields={"velocity": d["velocity"][i], "pressure": d["pressure"][i, :, 0]},
                signals={"cl": [float(i)]},
            )
        with pytest.raises(ValueError, match="same arrays"):
            app.append(9.0, fields={"velocity": d["velocity"][0]})
        with pytest.raises(ValueError, match="not after"):
            app.append(0.0, fields={"velocity": d["velocity"][0]})
    m = w.finalize()
    assert m.temporal == "transient" and {e.id: e.n_steps for e in m.samples} == {"g": 6, "m": 5}
    ds = Dataset(tmp_path)
    g, s = ds.sample("g"), ds.sample("m")
    assert g.location == "cell" and g.attrs["sample_kind"] == "grid"
    assert g.signal_names == ["cd", "f"] and g.signal("f").shape == (6, 3)
    assert g.signal("cd", t=2).tolist() == [2.0] and g.signal("cd", t=slice(1, 3)).shape == (2, 1)
    assert s.n_steps == 5 and s.meta.time is not None and s.meta.time.n_steps == 5
    np.testing.assert_allclose(s.field("velocity"), d["velocity"])
    np.testing.assert_allclose(s.field("pressure", t=4), d["pressure"][4])
    assert s.array("velocity").chunks[0] == 1 and s.signal("cl")[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert s.meta.parameters == {"Re": 100.0}
    rep = validate_dataset(tmp_path)
    assert rep.ok, rep.to_dict()
    write_transient_prediction(
        tmp_path / "p",
        "m",
        times=d["times"],
        pred={"velocity": d["velocity"]},
        true={"velocity": d["velocity"]},
        kind="mesh",
        points=d["points"],
        cells=d["cells"],
    )
    tar = pack_prediction_store(tmp_path / "p", "m")
    with tarfile.open(tar) as tf:
        names = tf.getnames()
    assert tar.name == "m.zarr.tar" and "m.zarr/zarr.json" in names
    assert any(n.startswith("m.zarr/fields/velocity_pred") for n in names)
