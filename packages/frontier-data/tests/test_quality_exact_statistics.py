# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from arrayroute.quantiles import ALGORITHM
from arrayroute.session import LocalReadSession
from frontier_data import Dataset, DatasetWriter
from frontier_data.cli import app
from frontier_data.quality import compute_sample_stats, field_stats, run_quality


def test_large_steady_field_quality_exceeds_read_allowance_without_materializing(tmp_path):
    values = np.random.default_rng(414).normal(size=(130, 131, 132)).astype("f4")
    values.flat[:4] = [np.nan, np.inf, -np.inf, -0.0]
    root = tmp_path / "data"
    writer = DatasetWriter(root, name="exact-statistics")
    writer.write_grid_sample(
        "case",
        fields={"pressure": values},
        ndim=3,
        field_units={"pressure": "Pa"},
        static={"mask": values},
    )
    writer.finalize()
    ds = Dataset(root)
    session = LocalReadSession(buffer_bytes=2 << 20)
    with pytest.raises(MemoryError):
        ds.sample("case", session=session).field("pressure")
    expected = field_stats(values)
    result = compute_sample_stats(ds, "case", session=session)
    for key, value in expected.items():
        assert result.fields["pressure"][key] == pytest.approx(value, rel=2e-13, abs=1e-14)
    check = next(c for c in result.checks if c.check == "nan_inf_per_step")
    assert check.status == "fail"
    meta = check.detail["field_statistics"]["pressure"]
    assert meta["algorithm"] == ALGORITHM and meta["percentile_scope"] == "full_selection"
    assert meta["passes"] > 1 and len(meta["selected_bytes_sha256"]) == 64
    assert meta["finite_count"] == values.size - 3

    assert next(f.units for f in ds.sample("case").meta.fields if f.name == "pressure") == "Pa"


def test_statistics_workspace_cli_and_worker_provenance(tmp_path):
    writer = DatasetWriter(tmp_path / "data", name="workspace-policy")
    writer.write_grid_sample(
        "case", fields={"pressure": np.arange(27, dtype="f4").reshape(3, 3, 3)},
        field_units={"pressure": "Pa"},
    )
    writer.finalize()
    ds = Dataset(tmp_path / "data")
    run = CliRunner().invoke(app, [
        "quality", str(ds.root), "--no-html", "--statistics-workspace-mib", "4",
    ])
    assert run.exit_code == 0, run.output
    row = json.loads((ds.root / "quality/report.json").read_text())["samples"][0]
    details = next(c["detail"] for c in row["checks"] if c["check"] == "nan_inf_per_step")
    assert details["field_statistics"]["pressure"]["workspace_bytes"] == 4 << 20

    worker = compute_sample_stats(ds, "case", statistics_workspace_bytes=4 << 20)
    report = run_quality(ds, precomputed={"case": worker}, write_html=False)
    merged = next(c for c in report["samples"][0]["checks"] if c["check"] == "nan_inf_per_step")
    assert merged["detail"]["field_statistics"]["pressure"]["workspace_bytes"] == 4 << 20
    assert worker.fields == compute_sample_stats(ds, "case").fields
    bad = CliRunner().invoke(app, [
        "quality", str(ds.root), "--statistics-workspace-mib", "3",
    ])
    assert bad.exit_code == 2


@pytest.mark.parametrize("value", [True, 0, (4 << 20) - 1, 4.5])
def test_invalid_statistics_workspace_rejected_before_dataset_access(value):
    with pytest.raises(ValueError, match="statistics_workspace_bytes"):
        compute_sample_stats(None, "missing", statistics_workspace_bytes=value)
    with pytest.raises(ValueError, match="statistics_workspace_bytes"):
        run_quality(None, statistics_workspace_bytes=value)
