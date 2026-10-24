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
    writer.finalize()
    op = get_op("prep.quality.check")
    params = {
        "reader": {
            "engine": "tensorstore",
            "tensorstore_threads": 2,
            "tensorstore_cache_bytes": 1 << 20,
        },
        "read_buffer_mib": 8,
    }
    restored = op.Params.model_validate(op.Params.model_validate(params).model_dump())
    assert restored.reader.engine == "tensorstore"
    op.run(root, params)
    report = json.loads((root / "quality/report.json").read_text())
    policy = report["samples"][0]["read_policy"]
    assert policy["engine"] == "tensorstore"
    assert policy["buffer_bytes"] == 8 << 20
    assert policy["tensorstore_cache_bytes"] == 1 << 20
