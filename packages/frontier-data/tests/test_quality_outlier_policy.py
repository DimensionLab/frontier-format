# SPDX-License-Identifier: Apache-2.0
from frontier_data.quality import CheckResult, compute_sample_stats, run_quality
from frontier_data.synthetic import write_synthetic_dataset


def test_nonstationary_mean_policy_retains_sanity_failures(tmp_path):
    ds = write_synthetic_dataset(tmp_path / "ds", n_samples=31, n_steps=4, size=4)
    stats = {entry.id: compute_sample_stats(ds, entry.id) for entry in ds.manifest.samples}
    for i, sample in enumerate(stats.values()):
        for values in sample.fields.values():
            values.update(mean=1 + i / 100, min=0, max=2)
    sample = list(stats.values())[-1]
    for values in sample.fields.values():
        values["mean"] = 100
    sample.checks.append(CheckResult("nan_inf", "fail", {"non_finite_counts": {"vorticity": 1}}))
    strict = run_quality(ds, precomputed=stats, write_html=False)
    measured = run_quality(ds, precomputed=stats, write_html=False, mean_outlier_action="warn")
    strict_row = next(row for row in strict["samples"] if row["id"] == sample.sample_id)
    measured_row = next(row for row in measured["samples"] if row["id"] == sample.sample_id)
    assert (
        next(c for c in strict_row["checks"] if c["check"] == "field_outliers")["status"] == "fail"
    )
    outliers = next(c for c in measured_row["checks"] if c["check"] == "field_outliers")
    assert outliers["status"] == "warn" and outliers["detail"]["mean_outlier_action"] == "warn"
    assert next(c for c in measured_row["checks"] if c["check"] == "nan_inf")["status"] == "fail"
    assert measured_row["status"] == "fail"
