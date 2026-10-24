# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.transfer import BlockArray
from frontier_data import Dataset, DatasetWriter


def test_grid_writer_reads_only_bounded_shards_with_correct_units_and_coordinates(tmp_path):
    shape = (5, 17, 13, 9, 3)
    peak = 0
    reads = []

    def read(region):
        nonlocal peak
        starts = [np.arange(s.start, s.stop) for s in region]
        arrays = np.meshgrid(*starts, indexing="ij")
        value = sum(a * (10**i) for i, a in enumerate(arrays)).astype("f8")
        peak = max(peak, value.nbytes)
        reads.append(tuple(s.start for s in region))
        return value

    field = BlockArray(shape, np.dtype("f8"), read)
    writer = DatasetWriter(tmp_path / "data", name="streaming")
    writer.write_grid_sample(
        "case",
        fields={"velocity": field},
        times=np.arange(5) * 0.1,
        spacing=[0.2, 0.3, 0.4],
        field_units={"velocity": "m/s"},
        write_buffer_bytes=8192,
    )
    writer.finalize()
    sample = Dataset(tmp_path / "data").sample("case")
    assert peak * 12 // 8 <= 8192
    assert (
        sample.field("velocity").tobytes()
        == read(tuple(slice(0, n) for n in shape)).astype("f4").tobytes()
    )
    # Exclude the explicit full read used for the independent expected array.
    assert len(reads) > 2 and reads[:-1] == sorted(reads[:-1])
    assert sample.meta.fields[0].units == "m/s"
    assert np.allclose(sample.times, np.arange(5) * 0.1)


def test_cancelled_grid_transfer_remains_unpublished(tmp_path):
    from arrayroute.publication import staged_directory

    out = tmp_path / "dataset"
    cancelled = False

    def read(region):
        nonlocal cancelled
        cancelled = True
        return np.zeros(tuple(s.stop - s.start for s in region), dtype="f4")

    with pytest.raises(InterruptedError), staged_directory(out) as stage:
        writer = DatasetWriter(stage, name="cancelled")
        writer.write_grid_sample(
            "case",
            fields={"pressure": BlockArray((4, 8, 8, 1), np.dtype("f4"), read)},
            times=np.arange(4),
            ndim=2,
            cancelled=lambda: cancelled,
        )
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []
