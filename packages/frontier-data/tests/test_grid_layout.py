# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.layout import GridLayoutPolicy
from frontier_data.dataset import Dataset, DatasetWriter
from frontier_data.query import FieldReader


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
@pytest.mark.parametrize("transient", [False, True])
@pytest.mark.parametrize("custom", [False, True])
def test_sharded_grid_reads_are_bit_exact_across_engines(tmp_path, engine, transient, custom):
    if engine != "zarr-python":
        pytest.importorskip(engine)
    shape = ((3,) if transient else ()) + (65, 37, 35, 3)
    values = np.random.default_rng(19).normal(size=shape).astype(np.float32)
    # Lossless compression must retain signed zero and subnormals too.
    values.flat[:4] = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001], np.uint32).view(
        np.float32
    )
    policy = GridLayoutPolicy(chunk_edge=8, shard_edge=16) if custom else GridLayoutPolicy()
    writer = DatasetWriter(tmp_path, "spatial layout fixture")
    writer.write_grid_sample(
        "case",
        fields={"velocity": values},
        times=np.arange(3) * 0.1 if transient else None,
        spacing=[0.2, 0.3, 0.4],
        field_units={"velocity": "m/s"},
        storage_layout=policy,
    )
    writer.finalize()
    sample = Dataset(tmp_path).sample("case")
    array = sample.array("velocity")
    lead = int(transient)
    layout = policy.plan(
        shape,
        itemsize=4,
        spatial_axes=tuple(range(lead, lead + 3)),
        time_axis=0 if transient else None,
    )
    assert array.chunks == layout.chunks
    assert array.shards == layout.shards and layout.shards is not None
    reader = FieldReader(sample, "velocity", engine=engine)
    result = reader.read({})
    assert result.values.tobytes() == values.tobytes()
    assert result.units == "m/s" and result.association == "node"
    selectors = {"x": slice(29, 65, 3), "y": 32, "z": slice(15, 35, 2), "component": 2}
    expected = values[..., 29:65:3, 32, 15:35:2, 2]
    if transient:
        selectors["time"] = 1
        expected = expected[1]
    selected = reader.read(selectors)
    assert selected.dimensions == ("x", "z")
    assert selected.values.tobytes() == expected.tobytes()


def test_grid_layout_limits_decode_work_for_a_small_roi(tmp_path):
    values = np.zeros((2, 128, 128, 128, 3), np.float32)
    writer = DatasetWriter(tmp_path, "region access")
    writer.write_grid_sample(
        "case",
        fields={"velocity": values},
        times=[0, 0.1],
        spacing=[1, 1, 1],
        field_units={"velocity": "m/s"},
    )
    writer.finalize()
    reader = FieldReader(Dataset(tmp_path).sample("case"), "velocity")
    assert reader.reader.backend.spec.chunks == (1, 32, 32, 32, 3)
    plan = reader.plan({"time": 1, "x": slice(0, 16), "y": slice(0, 16), "z": slice(0, 16)})
    assert plan.output_bytes == 16**3 * 3 * 4
    assert plan.chunks_touched == 1


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
@pytest.mark.parametrize("steps", [None, 1, 7])
def test_small_grid_default_layout_preserves_bytes_units_and_sparse_reads(tmp_path, engine, steps):
    if engine != "zarr-python":
        pytest.importorskip(engine)
    values = np.random.default_rng(15).normal(size=(37, 2, 3, 2)).astype("f4")
    values.flat[:4] = np.array([0x80000000, 0x00000001, 0x7FC00011, 0xFF800000], "u4").view("f4")
    times = np.arange(37, dtype="f8") * 0.125
    writer = DatasetWriter(tmp_path, "small-grid-time-layout")
    writer.write_grid_sample(
        "case",
        fields={"pressure": values},
        times=times,
        ndim=2,
        field_units={"pressure": "Pa"},
        steps_per_chunk=steps,
        signals={"inflow": np.arange(37, dtype="f4")},
    )
    writer.finalize()
    sample = Dataset(tmp_path).sample("case")
    array = sample.array("pressure")
    assert array.chunks[0] == (37 if steps is None else steps)
    assert array.attrs["storage_layout"]["requested_steps_per_chunk"] == steps
    assert array.attrs["storage_layout"]["algorithm"] == "bounded-spatiotemporal-grid-v2"
    assert sample.times.tobytes() == times.tobytes()
    reader = FieldReader(sample, "pressure", engine=engine)
    assert reader.read({}).values.tobytes() == values.tobytes()
    result = reader.read({"time": slice(3, 35, 7), "component": 1})
    assert result.values.tobytes() == values[3:35:7, ..., 1].tobytes()
    assert result.units == "Pa"
    objects = list((tmp_path / "samples/case/grid.zarr/fields/pressure").glob("c.*"))
    assert len(objects) == (37 + array.chunks[0] - 1) // array.chunks[0]
