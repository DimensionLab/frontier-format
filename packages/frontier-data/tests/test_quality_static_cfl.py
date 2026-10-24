# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.session import LocalReadSession
from frontier_data import Dataset, DatasetWriter
from frontier_data.quality_transient import check_cfl


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
def test_static_cfl_exceeds_read_allowance_and_preserves_exact_percentiles(tmp_path, engine):
    if engine != "zarr-python":
        pytest.importorskip(engine)
    shape = (80, 81, 82)
    velocity = np.random.default_rng(35).normal(size=(*shape, 3)).astype("f4")
    velocity[0, 0, 0] = np.nan
    writer = DatasetWriter(tmp_path / "flow", name="static-cfl")
    writer.write_grid_sample(
        "case", fields={"pressure": np.ones((2, *shape), dtype="f4")},
        static={"velocity": velocity}, times=np.array([0., 0.1]), spacing=[0.01] * 3,
        field_units={"pressure": "Pa"},
    )
    writer.finalize()
    sample = Dataset(tmp_path / "flow").sample(
        "case", session=LocalReadSession(engine, buffer_bytes=2 << 20),
    )
    with pytest.raises(MemoryError):
        sample.static("velocity")
    report = check_cfl(sample, [], sample.times, statistics_workspace_bytes=4 << 20)
    speed = np.sqrt(np.sum(velocity.astype("f8")**2, axis=-1)).ravel()
    speed = speed[np.isfinite(speed)]
    assert report.detail["u_max"] == speed.max()
    assert report.detail["u_p99"] == np.quantile(speed, .99)
    assert report.detail["cfl_p99"] == np.quantile(speed, .99) * .1 / .01
    assert report.detail["statistics"]["statistics_workspace_bytes"] == 4 << 20
    assert report.detail["statistics"]["finite_count"] == speed.size
    assert report.detail["statistics"]["percentile_scope"] == "full_selection"
    assert report.status == "warn"


def test_nonfinite_static_cfl_skips_without_fabricating_speed(tmp_path):
    writer = DatasetWriter(tmp_path / "data", name="nonfinite-cfl")
    writer.write_grid_sample("case", fields={"p": np.ones((2, 2, 2))}, ndim=2,
                             static={"velocity": np.full((2, 2, 2), np.nan)},
                             times=np.array([0., 1.]))
    writer.finalize()
    sample = Dataset(tmp_path / "data").sample("case")
    result = check_cfl(sample, [], sample.times)
    assert result.status == "skip" and "no finite" in result.detail["reason"]
