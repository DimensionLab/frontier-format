# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from frontier_data import DatasetWriter
from frontier_prep.io import SampleEditor
from frontier_prep.op import Change, Selection, get_op
from frontier_prep.preview import compare_changes


def pair(tmp_path: Path, *, transient: bool = True):
    root = tmp_path / "before"
    w = DatasetWriter(root, name="preview", domain="cfd")
    n = 100_001
    values = np.ones((5, n, 1) if transient else (n, 1), dtype=np.float32)
    w.write_mesh_sample(
        "case",
        points=np.zeros((n, 3), dtype=np.float32),
        cells=np.array([[0, 1, 2]], dtype=np.int32),
        fields={"p": values},
        times=np.arange(5) * 0.1 if transient else None,
        field_units={"p": "Pa"},
    )
    w.finalize()
    target = tmp_path / "after"
    shutil.copytree(root, target)
    return SampleEditor(root / "samples/case"), SampleEditor(target / "samples/case"), values


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
def test_preview_counts_every_selected_value_with_units_and_scope(tmp_path, engine):
    if engine != "zarr-python":
        pytest.importorskip(engine)
    before, after, values = pair(tmp_path)
    values[1, 137, 0] += 100
    values[3, 211, 0] += 500  # Outside the preview's time window.
    after.write_field("p", values, location="point", units="Pa")
    (result,) = compare_changes(
        before,
        after,
        [Change(target="samples/case/fields/p", action="modify")],
        Selection(max_steps=2, percentile_samples=2, read_engine=engine, read_buffer_bytes=4 << 20),
    )
    assert result.n_before == 200_002
    assert result.n_changed == result.n_bitwise_changed == 1
    assert result.max_abs_diff == 100
    assert result.units_before == result.units_after == result.difference_units == "Pa"
    assert result.source_shape_before == [5, 100_001, 1]
    assert result.selection_before is not None
    assert result.selection_before[0] == [0, 2, 1]
    assert result.comparison_scope == "full_selection"
    assert result.before is not None and result.after is not None
    assert result.read_before is not None
    assert result.before["percentile_sample_count"] == 2
    assert result.after["max"] == 101
    assert result.after["percentile_scope"] == "deterministic_sample"
    assert result.read_before["output_bytes"] == result.n_before * 4


def test_units_change_does_not_report_a_physical_delta(tmp_path):
    before, after, values = pair(tmp_path, transient=False)
    after.write_field("p", values * 0.001, location="point", units="kPa")
    (result,) = compare_changes(
        before, after, [Change(target="fields/p", action="modify")], Selection()
    )
    assert result.units_before == "Pa" and result.units_after == "kPa"
    assert result.max_abs_diff is result.rel_l2_diff is result.difference_units is None
    assert result.note is not None and "Units changed" in result.note
    assert result.before is not None and result.after is not None
    assert result.before["mean"] == 1
    assert result.after["mean"] == pytest.approx(0.001)


def test_renamed_field_compares_actual_old_and_new_arrays(tmp_path):
    before, after, _ = pair(tmp_path, transient=False)
    after.rename_field("p", "pressure")
    (result,) = compare_changes(
        before,
        after,
        [
            Change(
                target="samples/case/point:pressure",
                action="rename",
                before={"name": "p"},
                after={"name": "pressure"},
            )
        ],
        Selection(),
    )
    assert result.n_changed == 0
    assert result.units_before == result.units_after == "Pa"


def test_component_pattern_resolves_only_actual_matching_arrays(tmp_path):
    before, after, values = pair(tmp_path, transient=False)
    after.write_field("p_x", values, location="point", units="Pa")
    after.write_field("p_y", values * 2, location="point", units="Pa")
    results = compare_changes(
        before, after, [Change(target="fields/p_*", action="add")], Selection()
    )
    assert [d.target for d in results] == ["samples/case/fields/p_x", "samples/case/fields/p_y"]
    assert all(d.note == "added" and d.n_before is None for d in results)


def test_cancel_missing_array_and_chunk_admission_fail_explicitly(tmp_path):
    before, after, _ = pair(tmp_path)
    changes = [Change(target="fields/p", action="modify")]
    with pytest.raises(InterruptedError):
        compare_changes(before, after, changes, Selection(), cancelled=lambda: True)
    with pytest.raises(MemoryError):
        compare_changes(before, after, changes, Selection(read_buffer_bytes=32))
    with pytest.raises(KeyError):
        compare_changes(
            before, after, [Change(target="fields/absent", action="modify")], Selection()
        )


def test_failed_comparison_is_not_reported_as_a_complete_preview(case, monkeypatch):
    from frontier_prep import base

    root = case("naming")

    def broken(*args, **kwargs):
        raise OSError("corrupt chunk")

    monkeypatch.setattr(base, "compare_changes", broken)
    result = get_op("prep.fields.map").preview(root, {}, sample=Selection(renders=False))
    assert result.truncated
    assert not result.diffs
    assert any("comparison failed: OSError: corrupt chunk" in note for note in result.notes)


def test_geometry_and_time_arrays_share_bounded_reader(tmp_path):
    before, after, _ = pair(tmp_path)
    changes = [Change(target=p, action="modify") for p in ("points", "times")]
    geometry, times = compare_changes(before, after, changes, Selection(max_steps=2))
    assert geometry.n_changed == 0 and geometry.n_before == 300_003
    assert geometry.read_before is not None
    assert geometry.read_before["output_bytes"] == 300_003 * 4
    assert times.n_before == 2 and times.units_before == times.units_after == "s"
    with pytest.raises(MemoryError):
        compare_changes(before, after, changes, Selection(read_buffer_bytes=32))


def test_overlapping_patterns_do_not_duplicate_comparisons(tmp_path):
    before, after, values = pair(tmp_path, transient=False)
    after.write_field("p_x", values, location="point", units="Pa")
    results = compare_changes(before, after, [
        Change(target="fields/p_*", action="add"),
        Change(target="fields/p_x", action="add"),
    ], Selection())
    assert len(results) == 1
