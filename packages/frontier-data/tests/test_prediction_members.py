# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import zarr

from frontier_data.predictions import write_transient_prediction


def test_chunked_ensemble_and_time_indices_preserve_archive_values(tmp_path):
    rng = np.random.default_rng(73)
    truth = rng.normal(size=(5, 4, 3, 2))
    members = rng.normal(size=(3, *truth.shape))
    indices = np.arange(45, 50)
    write_transient_prediction(
        tmp_path, "s", times=np.arange(5) * .1, pred={"u": members.mean(0)},
        true={"u": truth}, ensemble={"u": members}, t_index=indices,
        field_units={"u": "m/s"}, spacing=[.25, .5],
    )
    group = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
    with np.load(tmp_path / "s.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(group["ensemble/u"][:], archive["ensemble"])
        np.testing.assert_array_equal(group["t_index"][:], archive["t_index"])
    assert group["ensemble/u"].dtype == np.dtype("f4")
    assert group["ensemble/u"].chunks[0] == 1
    assert group["ensemble/u"].attrs["units"] == "m/s"


@pytest.mark.parametrize("error", ["missing", "shape", "empty", "indices"])
def test_invalid_members_and_indices_rejected_before_writing_payload(tmp_path, error):
    a = np.ones((2, 3, 4, 1))
    members = {"u": np.stack([a, a])}
    indices = np.arange(2)
    if error == "missing":
        members = {}
    elif error == "shape":
        members = {"u": np.ones((2, 2, 3, 4))}
    elif error == "empty":
        members = {"u": np.ones((0, *a.shape))}
    else:
        indices = np.arange(3)
    with pytest.raises(ValueError):
        write_transient_prediction(tmp_path, "s", times=np.arange(2), pred={"u": a},
                                   true={"u": a}, ensemble=members, t_index=indices)
    assert not list(tmp_path.glob("*.npz"))
