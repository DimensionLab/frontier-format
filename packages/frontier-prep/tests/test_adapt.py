# SPDX-License-Identifier: Apache-2.0
"""Data adapted to a model's contract by a recipe of ordinary ops."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontier_prep.adapt import adapt, recipe_for, unset_values
from frontier_prep.op import get_op


def _dataset(tmp_path: Path) -> Path:
    """Four Ahmed bodies at different slants, 770 points each (the public input writer)."""
    from frontier_data.inputs import write_input

    return write_input(
        tmp_path / "ds", [{"slant_angle": a} for a in (10.0, 20.0, 30.0, 40.0)], geometry="ahmed"
    )


def test_a_verdict_becomes_a_recipe_of_known_ops():
    verdict = {
        "verdict": "Needs adapting: resample, set reynolds, convert mm to m.",
        "problems": [
            {
                "kind": "resolution_out_of_range",
                "want": {"min": 100, "max": 300},
                "have": 770,
                "fix": "resample",
                "line": "too many",
            },
            {
                "kind": "missing_parameter",
                "name": "reynolds",
                "fix": "set_value",
                "line": "needs reynolds",
            },
            {
                "kind": "unit_mismatch",
                "name": "length",
                "want": "m",
                "have": "mm",
                "fix": "convert",
                "line": "mm",
            },
            {"kind": "geometry_kind", "fix": None, "line": "a grid"},
        ],
    }
    recipe = recipe_for({"task": "steady-surface-field"}, verdict)
    assert [s.op for s in recipe.steps] == [
        "prep.mesh.decimate",
        "prep.units.convert",
        "prep.params.set",
    ]
    assert recipe.steps[0].params == {"levels": [1], "targets": {1: 600}}
    assert unset_values(recipe) == ["reynolds"]
    with_value = recipe_for({}, verdict, values={"reynolds": 1e6})
    assert unset_values(with_value) == [] and with_value.steps[-1].params["values"] == {
        "reynolds": 1e6
    }
    for s in recipe.steps:
        assert get_op(s.op).version == s.op_version


def test_the_adapter_resamples_and_sets_parameters_as_a_dataset_version(tmp_path: Path):
    ds = _dataset(tmp_path)
    fixes = [
        {
            "kind": "resolution_out_of_range",
            "want": {"min": 50, "max": 200},
            "have": 770,
            "fix": "resample",
            "line": "too many",
        },
        {
            "kind": "missing_parameter",
            "name": "reynolds",
            "fix": "set_value",
            "line": "needs reynolds",
        },
    ]
    with pytest.raises(ValueError, match="needs a value for reynolds"):
        adapt(ds, {"task": "steady-surface-field"}, tmp_path / "x", fixes=fixes)
    out, run = adapt(
        ds,
        {"task": "steady-surface-field", "trained_on": {"dataset": "t"}},
        tmp_path / "adapted",
        fixes=fixes,
        values={"reynolds": 1e6},
    )
    assert run.ok and [s.status for s in run.steps] == ["ok", "ok"]
    assert run.output_version and run.output_version != run.input_version
    manifest = json.loads((out / "manifest.json").read_text())
    assert "reynolds" in manifest["geometry_params"]
    assert all(s["geometry_params"]["reynolds"] == 1e6 for s in manifest["samples"])
    sid = manifest["samples"][0]["id"]
    meta = json.loads((out / "samples" / sid / "meta.json").read_text())
    assert 1 in meta["lod_levels"]
    assert (out / "samples" / sid / "surface.zarr" / "lod" / "1").is_dir()
    assert (out / "adapted.json").is_file()
    # the source is untouched, and the adapted dataset reads at the new level
    assert not (ds / "samples" / sid / "surface.zarr" / "lod" / "1").exists()
    from frontier_data.dataset import read_surface

    assert read_surface(out / "samples" / sid / "surface.zarr", lod=1).n_points <= 260
    # nothing to fix: no version is made
    out2, run2 = adapt(ds, {}, tmp_path / "same", fixes=[])
    assert run2.ok and not run2.steps
