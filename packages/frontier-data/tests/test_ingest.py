# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from frontier_data import validate_dataset
from frontier_data.ingest.common import split_counts
from frontier_data.ingest.meshgraphnets import (
    convert_meshgraphnets,
    count_complete_records,
    cylinder_parameters,
    iter_tfrecord,
    parse_example,
    read_tfrecord_trajectories,
    write_tfrecord,
)
from frontier_data.synthetic import vortex_street_mesh

META = {
    "simulator": "test",
    "dt": 0.01,
    "trajectory_length": 6,
    "features": {
        "cells": {"type": "static", "shape": [1, -1, 3], "dtype": "int32"},
        "mesh_pos": {"type": "static", "shape": [1, -1, 2], "dtype": "float32"},
        "node_type": {"type": "static", "shape": [1, -1, 1], "dtype": "int32"},
        "velocity": {"type": "dynamic", "shape": [6, -1, 2], "dtype": "float32"},
        "pressure": {"type": "dynamic", "shape": [6, -1, 1], "dtype": "float32"},
    },
}


def _traj(seed: int) -> dict[str, np.ndarray]:
    d = vortex_street_mesh(n_target=150, n_steps=6, dt=0.01, seed=seed)
    return {
        "cells": d["cells"][None].astype(np.int32),
        "mesh_pos": d["points"][None].astype(np.float32),
        "node_type": d["node_type"][None, :, None].astype(np.int32),
        "velocity": d["velocity"],
        "pressure": d["pressure"],
    }


def test_tfrecord_roundtrip_and_truncated_prefix(tmp_path: Path) -> None:
    trajs = [_traj(i) for i in range(3)]
    path = tmp_path / "train.tfrecord"
    write_tfrecord(path, trajs)
    assert count_complete_records(path) == (3, path.stat().st_size)
    back = list(read_tfrecord_trajectories(path, META))
    assert len(back) == 3
    for a, b in zip(trajs, back, strict=True):
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
    # a prefix cut in the middle of the third record still yields two trajectories
    cut = tmp_path / "cut.tfrecord"
    cut.write_bytes(path.read_bytes()[: path.stat().st_size - 100])
    assert count_complete_records(cut)[0] == 2
    assert len(list(read_tfrecord_trajectories(cut, META))) == 2
    payload = next(iter_tfrecord(io.BytesIO(path.read_bytes())))
    assert set(parse_example(payload)) == set(META["features"])


def test_convert_meshgraphnets(tmp_path: Path) -> None:
    files: dict[str, str | Path] = {}
    for split, n in (("train", 3), ("valid", 1), ("test", 1)):
        files[split] = tmp_path / f"{split}.tfrecord"
        write_tfrecord(files[split], [_traj(hash(split) % 100 + i) for i in range(n)])
    ds = convert_meshgraphnets(
        files, META, tmp_path / "ds", trajectories={"train": 2}, time_stride=2, quality=False
    )
    assert {k: len(v) for k, v in ds.manifest.splits.items()} == {"train": 2, "val": 1, "test": 1}
    assert ds.manifest.sample_kind == "mesh" and ds.manifest.hub_id == "meshgraphnets-cylinder-flow"
    assert ds.manifest.time is not None and ds.manifest.time.dt == pytest.approx(0.02)
    s = ds.sample("train_0000")
    assert s.n_steps == 3 and s.field("velocity", t=0).shape[1] == 2
    assert s.static("node_type").ndim == 1
    assert {"u_inlet_max", "cylinder_d", "cylinder_x"} <= set(s.meta.parameters)
    assert s.meta.parameters["cylinder_d"] == pytest.approx(0.1, rel=0.15)
    assert s.meta.provenance.license == "CC-BY-4.0"
    assert validate_dataset(ds.root).ok


def test_cylinder_parameters_without_cylinder() -> None:
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    nt = np.array([4, 4, 5, 5])
    out = cylinder_parameters(pts, nt, np.ones((4, 2), dtype=np.float32))
    assert out == {"u_inlet_max": 1.0, "u_inlet_mean": 1.0}


def test_split_counts() -> None:
    assert split_counts(12) == [10, 1, 1] and split_counts(2) == [2, 0, 0]
    assert split_counts(100) == [80, 10, 10] and sum(split_counts(7)) == 7


def _well_file(path: Path, n_traj: int, tcool: float, seed: int) -> None:
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(seed)
    nt, nx, ny = 5, 8, 12
    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "toy_layer"
        f.attrs["n_spatial_dims"] = 2
        f.attrs["n_trajectories"] = n_traj
        f.attrs["grid_type"] = "cartesian"
        d = f.create_group("dimensions")
        d.attrs["spatial_dims"] = ["x", "y"]
        t = d.create_dataset("time", data=np.linspace(0, 1, nt))
        t.attrs["sample_varying"] = False
        d.create_dataset("x", data=np.linspace(0, 1, nx, endpoint=False))
        d.create_dataset("y", data=np.linspace(-1, 2, ny))
        sc = f.create_group("scalars")
        sc.create_dataset("tcool", data=tcool)
        b = f.create_group("boundary_conditions")
        for name, kind, dim in (("x_periodic", "PERIODIC", "x"), ("y_open", "OPEN", "y")):
            g = b.create_group(name)
            g.attrs["bc_type"] = kind
            g.attrs["associated_dims"] = [dim]
        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = ["density", "background"]
        ds = t0.create_dataset("density", data=rng.random((n_traj, nt, nx, ny), dtype=np.float32))
        ds.attrs.update(sample_varying=True, time_varying=True, dim_varying=[True, True])
        bg = t0.create_dataset("background", data=rng.random((ny,), dtype=np.float32))
        bg.attrs.update(sample_varying=False, time_varying=False, dim_varying=[False, True])
        t1 = f.create_group("t1_fields")
        v = t1.create_dataset(
            "velocity", data=rng.random((n_traj, nt, nx, ny, 2), dtype=np.float32)
        )
        v.attrs.update(sample_varying=True, time_varying=True, dim_varying=[True, True])
        f.create_group("t2_fields")


def test_convert_the_well(tmp_path: Path) -> None:
    from frontier_data.ingest.the_well import convert_the_well, read_well_file

    raw = tmp_path / "raw"
    files: dict[str, list[Path]] = {}
    for split, n in (("train", 3), ("valid", 1), ("test", 1)):
        (raw / split).mkdir(parents=True)
        files[split] = []
        for j, tcool in enumerate((0.03, 0.1)):
            p = raw / split / f"toy_layer_tcool_{tcool:.2f}.hdf5"
            _well_file(p, n, tcool, seed=j)
            files[split].append(p)
    first = next(read_well_file(files["train"][0]))
    assert first["fields"]["velocity"].shape == (5, 8, 12, 2)
    assert first["static"]["background"].shape == (8, 12, 1)  # broadcast over x
    assert first["boundary_conditions"] == {"x": "periodic", "y": "open"}
    ds = convert_the_well(
        files,
        tmp_path / "ds",
        name="toy_layer",
        trajectories={"train": 4},
        per_file=2,
        time_stride=2,
        quality=False,
    )
    assert {k: len(v) for k, v in ds.manifest.splits.items()} == {"train": 4, "val": 2, "test": 2}
    assert ds.manifest.sample_kind == "grid" and ds.manifest.temporal == "transient"
    assert ds.manifest.license == "CC-BY-4.0" and ds.manifest.parameters == ["tcool"]
    s = ds.sample("train_tcool_0_03_000")
    assert s.n_steps == 3 and s.shape == (8, 12)
    assert s.dims("density") == ["time", "x", "y", "component"]
    assert s.static_names == ["background"] and s.meta.parameters == {"tcool": 0.03}
    assert s.boundary_conditions["y"] == "open" and s.spacing is not None
    assert s.spacing[0] == pytest.approx(1 / 8)
    rep = validate_dataset(ds.root)
    assert rep.ok, rep.to_dict()
