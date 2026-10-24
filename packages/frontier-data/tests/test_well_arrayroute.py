# SPDX-License-Identifier: Apache-2.0
"""Well imports select time before I/O and admit expanded trajectory values before fields."""

import numpy as np
import pytest

from arrayroute.hdf5 import HDF5Backend, HDF5ReadSession
from frontier_data.ingest.the_well import read_well_file

h5py = pytest.importorskip("h5py")


def fixture(path, *, compressed=False):
    shape = (2, 17, 4, 5, 6)
    pressure = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tensor = np.arange(2 * 17 * 5 * 2 * 2, dtype=np.float64).reshape(2, 17, 5, 2, 2)
    with h5py.File(path, "w") as f:
        f.attrs.update(n_trajectories=2, n_spatial_dims=3, grid_type="cartesian")
        d = f.create_group("dimensions")
        d.attrs["spatial_dims"] = ["x", "y", "z"]
        for name, n in zip(["x", "y", "z"], shape[2:], strict=True):
            d[name] = np.linspace(0, 1, n)
        d["time"] = np.stack([np.arange(17) * 0.25, np.arange(17) * 0.5])
        ds = f.create_dataset(
            "t0_fields/pressure",
            data=pressure,
            **({"chunks": (1, 1, 4, 5, 6), "compression": "gzip"} if compressed else {}),
        )
        ds.attrs.update(sample_varying=True, time_varying=True, dim_varying=[True] * 3)
        ds = f.create_dataset("t0_fields/constant", data=np.float64(7))
        ds.attrs.update(sample_varying=False, time_varying=False, dim_varying=[False] * 3)
        ds = f.create_dataset("t2_fields/tensor", data=tensor)
        ds.attrs.update(sample_varying=True, time_varying=True, dim_varying=[False, True, False])
        f["scalars/parameter"] = np.array([2.0, 4.0])
        f["scalars/scalar"] = 3.0
    return pressure, tensor


@pytest.mark.parametrize("compressed", [False, True])
def test_selected_values_tensor_broadcast_and_parameters(tmp_path, compressed, monkeypatch):
    path = tmp_path / "well.h5"
    pressure, tensor = fixture(path, compressed=compressed)
    calls = []
    original = HDF5Backend.read

    def read(self, selection):
        calls.append((self._dataset.name, selection))
        return original(self, selection)

    monkeypatch.setattr(HDF5Backend, "read", read)
    results = list(read_well_file(path, time_stride=3, max_steps=4, read_buffer_bytes=2048))
    for traj, result in enumerate(results):
        assert result["fields"]["pressure"].tobytes() == pressure[traj, :12:3, ..., None].tobytes()
        expected = (
            np.broadcast_to(tensor[traj, :12:3, None, :, None, :, :], (4, 4, 5, 6, 2, 2))
            .astype(np.float32)
            .reshape(4, 4, 5, 6, 4)
        )
        assert result["fields"]["tensor"].tobytes() == expected.tobytes()
        assert result["static"]["constant"].shape == (4, 5, 6, 1)
        assert np.all(result["static"]["constant"] == 7)
        assert (
            result["times"].tolist()
            == (np.arange(0, 12, 3) * (0.25 if traj == 0 else 0.5)).tolist()
        )
        assert result["parameters"] == {"parameter": 2.0 if traj == 0 else 4.0, "scalar": 3.0}
    for name, selection in calls:
        if name.startswith(("/t0_fields/", "/t2_fields/")):
            assert isinstance(selection[1], slice)
            assert selection[1].step == 3
            assert selection[1].stop <= 12


def test_field_projection_and_budget_reject_before_payload(tmp_path, monkeypatch):
    path = tmp_path / "well.h5"
    fixture(path)
    reads = []
    original = HDF5Backend.read

    def read(self, selection):
        reads.append(self._dataset.name)
        return original(self, selection)

    monkeypatch.setattr(HDF5Backend, "read", read)
    with pytest.raises(MemoryError, match="materialization"):
        next(read_well_file(path, materialization_bytes=100))
    assert not any(n.startswith(("/t0_fields", "/t2_fields")) for n in reads)
    with pytest.raises(ValueError, match="Unknown"):
        next(read_well_file(path, fields=["pressure", "missing"]))
    assert not any(n.startswith(("/t0_fields", "/t2_fields")) for n in reads)
    selected = next(read_well_file(path, fields=["pressure"], max_steps=1))
    assert set(selected["fields"]) == {"pressure"}
    assert "/t2_fields/tensor" not in reads


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_stride": 0},
        {"max_steps": -1},
        {"max_trajectories": 0},
        {"fields": "pressure"},
        {"fields": ["pressure", "pressure"]},
    ],
)
def test_invalid_selection_controls_fail(tmp_path, kwargs):
    with pytest.raises(ValueError):
        next(read_well_file(tmp_path / "nonexistent", **kwargs))


def test_cancel_closes_session(tmp_path, monkeypatch):
    path = tmp_path / "well.h5"
    fixture(path)
    closed = []
    original = HDF5ReadSession.close

    def close(self):
        closed.append(self.path)
        original(self)

    monkeypatch.setattr(HDF5ReadSession, "close", close)
    with pytest.raises(InterruptedError):
        next(read_well_file(path, cancelled=lambda: True))
    assert closed == [path]


def test_links_are_not_followed(tmp_path):
    path = tmp_path / "well.h5"
    fixture(path)
    with h5py.File(path, "a") as f:
        f["t0_fields/external"] = h5py.ExternalLink("missing.h5", "/field")
    with pytest.raises(ValueError, match="external or soft"):
        next(read_well_file(path))


def test_generator_close_releases_file(tmp_path, monkeypatch):
    path = tmp_path / "well.h5"
    fixture(path)
    closed = []
    original = HDF5ReadSession.close

    def close(self):
        closed.append(self.path)
        original(self)

    monkeypatch.setattr(HDF5ReadSession, "close", close)
    reader = read_well_file(path)
    output = next(reader)
    reader.close()
    assert closed == [path]
    assert output["fields"]["pressure"].shape == (17, 4, 5, 6, 1)


def test_local_cli_controls_produce_inspectable_dataset(tmp_path):
    from typer.testing import CliRunner

    from frontier_data import Dataset
    from frontier_data.cli import app

    raw = tmp_path / "raw"
    source = raw / "data" / "train" / "well.hdf5"
    source.parent.mkdir(parents=True)
    pressure, _ = fixture(source)
    out = tmp_path / "converted"
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            "the-well",
            "synthetic_3d",
            "--local",
            "--raw-dir",
            str(raw),
            "--out",
            str(out),
            "--source-splits",
            "train",
            "--trajectories",
            "1",
            "--max-steps",
            "2",
            "--time-stride",
            "3",
            "--fields",
            "pressure",
            "--read-buffer-mib",
            "1",
            "--materialization-mib",
            "2",
            "--no-quality",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    dataset = Dataset(out)
    sample = dataset.sample(dataset.sample_ids[0])
    assert sample.n_steps == 2
    assert sample.field("pressure").tobytes() == pressure[0, :6:3, ..., None].tobytes()
    assert sample.meta.provenance.files[0].sha256


def test_per_trajectory_coordinates_and_axis_aliases(tmp_path):
    path = tmp_path / "coordinates.h5"
    fixture(path)
    with h5py.File(path, "a") as f:
        del f["dimensions/x"]
        coords = f.create_dataset(
            "dimensions/streamwise", data=np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
        )
        coords.attrs["sample_varying"] = True
        f["dimensions"].attrs["spatial_dims"] = ["streamwise", "y", "z"]
        bc = f.create_group("boundary_conditions/inlet")
        bc.attrs.update(bc_type="OPEN", associated_dims=["streamwise"])
    outputs = list(read_well_file(path, max_steps=1))
    assert outputs[0]["coords"]["x"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert outputs[1]["coords"]["x"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert outputs[1]["boundary_conditions"] == {"x": "open"}
    assert outputs[1]["axes"] == ["x", "y", "z"]


def test_conversion_failure_cleans_private_output_and_retry_is_exact(tmp_path, monkeypatch):
    from frontier_data.dataset import DatasetWriter
    from frontier_data.ingest.the_well import convert_the_well

    source, output = tmp_path / "source.h5", tmp_path / "dataset"
    pressure, _ = fixture(source)
    original = DatasetWriter.write_grid_sample
    calls = 0

    def fail_second(writer, *args, **kwargs):
        nonlocal calls
        calls += 1
        assert not output.exists()
        if calls == 2:
            raise OSError("injected disk write failure")
        return original(writer, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(DatasetWriter, "write_grid_sample", fail_second)
        with pytest.raises(OSError, match="injected disk"):
            convert_the_well({"train": [source]}, output, name="test", quality=False)
    assert calls == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".arrayroute-*.partial"))
    # The failed consumer must also close the HDF5 generator before returning.
    with h5py.File(source, "r+") as handle:
        handle.attrs["retry"] = True
    ds = convert_the_well({"train": [source]}, output, name="test", quality=False)
    assert ds.root == output
    assert len(ds.manifest.samples) == 2
    for i, sample in enumerate(ds.manifest.samples):
        actual = ds.sample(sample.id).field("pressure")
        assert actual.tobytes() == pressure[i, ..., None].tobytes()


def test_quality_exception_never_publishes_manifest(tmp_path, monkeypatch):
    from frontier_data import quality
    from frontier_data.ingest.the_well import convert_the_well

    source, output = tmp_path / "source.h5", tmp_path / "dataset"
    fixture(source)

    def fail(ds, **kwargs):
        assert (ds.root / "manifest.json").is_file()
        assert not output.exists()
        raise RuntimeError("injected quality failure")

    monkeypatch.setattr(quality, "run_quality", fail)
    with pytest.raises(RuntimeError, match="injected quality"):
        convert_the_well({"train": [source]}, output, name="test")
    assert not output.exists()
    assert not list(tmp_path.glob(".arrayroute-*.partial"))
