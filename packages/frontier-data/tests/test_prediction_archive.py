# SPDX-License-Identifier: Apache-2.0
import os
import tarfile

import numpy as np
import pytest

from frontier_data.arrays import Sample
from frontier_data.predictions import pack_prediction_store, write_transient_prediction


def test_repacked_prediction_is_byte_identical_and_keeps_units(tmp_path):
    truth = np.arange(4 * 5 * 6 * 2, dtype="f4").reshape(4, 5, 6, 2)
    write_transient_prediction(
        tmp_path, "case", times=np.arange(4, dtype="f8") * .1,
        pred={"u": truth + 2}, true={"u": truth}, field_units={"u": "m/s"},
    )
    path = pack_prediction_store(tmp_path, "case", buffer_bytes=17)
    original = path.read_bytes()
    store = tmp_path / "case.zarr"
    for p in [store, *store.rglob("*")]:
        os.utime(p, (123, 123))
    assert pack_prediction_store(tmp_path, "case", buffer_bytes=1 << 20).read_bytes() == original
    with tarfile.open(path) as archive:
        archive.extractall(tmp_path / "unpacked", filter="data")
    sample = Sample.open_store(tmp_path / "unpacked/case.zarr")
    np.testing.assert_array_equal(sample.field("u_true"), truth)
    np.testing.assert_array_equal(sample.field("u_err"), np.full_like(truth, 2))
    assert sample.group["fields/u_pred"].attrs["units"] == "m/s"
    np.testing.assert_array_equal(sample.times, np.arange(4) * .1)
    with pytest.raises(MemoryError):
        pack_prediction_store(tmp_path, "case", max_archive_bytes=512)
    assert path.read_bytes() == original
    with pytest.raises(InterruptedError):
        pack_prediction_store(tmp_path, "case", cancelled=lambda: True)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".arrayroute-*.partial"))


@pytest.mark.parametrize("sample", ["../case", "a/b", "a\\b", "", ".", ".."])
def test_prediction_archive_rejects_path_escape(tmp_path, sample):
    with pytest.raises(ValueError, match="component"):
        pack_prediction_store(tmp_path, sample)
