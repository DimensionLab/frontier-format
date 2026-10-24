# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from arrayroute.session import LocalReadSession
from frontier_data import Dataset, DatasetWriter
from frontier_data.cli import app
from frontier_data.quality import compute_sample_stats, run_quality


def dataset(path, transient=True):
    values = np.random.default_rng(19).normal(size=(4, 8, 9, 10, 3)).astype("f4")
    writer = DatasetWriter(path, name="quality-read-policy")
    writer.write_grid_sample(
        "flow",
        fields={"velocity": values if transient else values[0]},
        times=np.arange(4) if transient else None,
    )
    writer.finalize()
    return Dataset(path)


def scientific(report):
    result = {k: v for k, v in report.items() if k not in ("elapsed_s", "generated_at")}
    result["samples"] = [
        {k: v for k, v in row.items() if k != "read_policy"} for row in result["samples"]
    ]
    return result


@pytest.mark.parametrize("transient", [True, False])
def test_quality_reader_parity_and_precomputed_provenance(tmp_path, transient):
    pytest.importorskip("tensorstore")
    pytest.importorskip("zarrista")
    ds = dataset(tmp_path / "data", transient)
    baseline = run_quality(ds, write_html=False)
    for engine in ("tensorstore", "zarrista"):
        session = LocalReadSession(
            engine,
            buffer_bytes=8 << 20,
            tensorstore_cache_bytes=1 << 20 if engine == "tensorstore" else 0,
        )
        stats = compute_sample_stats(ds, "flow", session=session)
        # A merger's own settings must not mislabel the supplied worker's reads.
        report = run_quality(ds, precomputed={"flow": stats}, write_html=False)
        assert report["samples"][0]["read_policy"]["engine"] == engine
        assert report["samples"][0]["read_policy"]["buffer_bytes"] == 8 << 20
        assert scientific(report) == scientific(baseline)


def test_quality_cli_applies_and_records_cache_budget(tmp_path):
    pytest.importorskip("tensorstore")
    ds = dataset(tmp_path / "data")
    run = CliRunner().invoke(
        app,
        [
            "quality",
            str(ds.root),
            "--no-html",
            "--engine",
            "tensorstore",
            "--read-buffer-mib",
            "8",
            "--cache-mib",
            "1",
            "--native-threads",
            "2",
        ],
    )
    assert run.exit_code == 0, run.output
    policy = json.loads((ds.root / "quality/report.json").read_text())["samples"][0]["read_policy"]
    assert policy["engine"] == "tensorstore"
    assert policy["tensorstore_threads"] == 2
    assert policy["tensorstore_cache_bytes"] == 1 << 20
    assert policy["immutable"] is False
    bad = CliRunner().invoke(app, ["quality", str(ds.root), "--cache-mib", "1"])
    assert bad.exit_code == 2
    assert "requires the tensorstore engine" in bad.output
