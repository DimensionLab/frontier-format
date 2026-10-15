# SPDX-License-Identifier: Apache-2.0
"""Field units survive every native prediction store and its packed artifact."""

import tarfile
from pathlib import Path

import numpy as np
import pytest

from frontier_data.arrays import Sample
from frontier_data.predictions import (
    pack_prediction_store,
    read_transient_prediction,
    write_transient_prediction,
)


@pytest.mark.parametrize("kind", ["grid", "mesh", "point_cloud"])
@pytest.mark.parametrize("with_units", [False, True])
def test_prediction_units_in_store_and_archive(tmp_path: Path, kind: str, with_units: bool) -> None:
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    shape = (2, 3, 3, 1) if kind == "grid" else (2, 3, 1)
    truth = {"p": np.ones(shape, dtype=np.float32), "phi": np.zeros(shape, dtype=np.float32)}
    pred = {key: value + 2 for key, value in truth.items()}
    ensemble = {key: np.stack([value + 1, value + 3]) for key, value in truth.items()}
    supplied_units = {"p": "lattice pressure"} if with_units else None
    write_transient_prediction(
        tmp_path,
        "sample",
        times=np.array([0.0, 1.0]),
        pred=pred,
        true=truth,
        ensemble=ensemble,
        kind=kind,
        field_units=supplied_units,
        points=points if kind != "grid" else None,
        cells=np.array([[0, 1, 2]], dtype=np.int32) if kind == "mesh" else None,
    )
    prediction = read_transient_prediction(tmp_path, "sample")
    if with_units:
        assert prediction.info["field_units"] == supplied_units
    else:
        assert "field_units" not in prediction.info
    archive = pack_prediction_store(tmp_path, "sample")
    with tarfile.open(archive) as stream:
        stream.extractall(tmp_path / "unpacked", filter="data")
    for location in (tmp_path / "sample.zarr", tmp_path / "unpacked/sample.zarr"):
        sample = Sample.open_store(location)
        for suffix in ("pred", "true", "err", "std"):
            attrs = dict(sample.group[f"fields/p_{suffix}"].attrs)
            assert attrs.get("units") == ("lattice pressure" if with_units else None)
            assert "units" not in sample.group[f"fields/phi_{suffix}"].attrs
        np.testing.assert_allclose(sample.field("p_err"), 2)
        np.testing.assert_allclose(sample.field("p_std"), 1)
    assert supplied_units == ({"p": "lattice pressure"} if with_units else None)
