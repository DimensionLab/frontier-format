# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest

from frontier_data import DatasetWriter
from frontier_prep.op import get_op


def test_quality_recipe_uses_serializable_read_policy(tmp_path):
    pytest.importorskip("tensorstore")
    root = tmp_path / "flow"
    writer = DatasetWriter(root, name="quality-readers")
    writer.write_grid_sample(
        "sample", fields={"velocity": np.ones((4, 8, 8, 8, 3), dtype="f4")}, times=np.arange(4)
    )
    writer.write_grid_sample(
        "steady", fields={"pressure": np.arange(512, dtype="f4").reshape(8, 8, 8)},
        field_units={"pressure": "Pa"},
    )
    writer.finalize()
    op = get_op("prep.quality.check")
    params = {
        "reader": {
            "engine": "tensorstore",
            "tensorstore_threads": 2,
            "tensorstore_cache_bytes": 1 << 20,
        },
        "read_buffer_mib": 8,
        "statistics_workspace_mib": 4,
    }
    restored = op.Params.model_validate(op.Params.model_validate(params).model_dump())
    assert restored.reader.engine == "tensorstore"
    assert restored.statistics_workspace_mib == 4
    op.run(root, params)
    report = json.loads((root / "quality/report.json").read_text())
    policy = report["samples"][0]["read_policy"]
    assert policy["engine"] == "tensorstore"
    assert policy["buffer_bytes"] == 8 << 20
    assert policy["tensorstore_cache_bytes"] == 1 << 20
    steady = next(row for row in report["samples"] if row["id"] == "steady")
    check = next(c for c in steady["checks"] if c["check"] == "nan_inf_per_step")
    assert check["detail"]["field_statistics"]["pressure"]["workspace_bytes"] == 4 << 20


@pytest.mark.parametrize("value", [3, True, 4.5])
def test_quality_recipe_rejects_invalid_statistics_workspace(value):
    with pytest.raises(ValueError):
        get_op("prep.quality.check").Params.model_validate({"statistics_workspace_mib": value})
