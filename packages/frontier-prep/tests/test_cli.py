# SPDX-License-Identifier: Apache-2.0
"""The command line reports wrong input as a message, not a traceback."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from frontier_prep.benchmarks import make_case
from frontier_prep.cli import app

runner = CliRunner()


def _out(res) -> str:
    return res.output + (res.stderr if res.stderr_bytes else "")


def test_wrong_param_key_names_the_valid_ones(tmp_path: Path) -> None:
    root = tmp_path / "naming"
    make_case("naming", root)
    for verb in ("plan", "preview", "run"):
        res = runner.invoke(
            app, [verb, "prep.fields.rename", "-d", str(root), "-p", '{"mapping": {"a": "b"}}']
        )
        text = _out(res)
        assert res.exit_code == 1, text
        assert "Traceback" not in text and res.exception.__class__ is SystemExit
        assert "invalid parameters for prep.fields.rename" in text
        assert (
            "mapping: unknown parameter" in text and "valid parameters: map, missing, units" in text
        )
    res = runner.invoke(app, ["plan", "prep.fields.rename", "-d", str(root), "-p", "{not json"])
    assert res.exit_code == 1 and "not valid JSON" in _out(res)
    res = runner.invoke(app, ["plan", "prep.fields.renam", "-d", str(root)])
    assert res.exit_code == 1 and "did you mean: prep.fields.rename" in _out(res)
    ok = runner.invoke(
        app, ["plan", "prep.fields.rename", "-d", str(root), "-p", '{"map": {"pMean": "p"}}']
    )
    assert ok.exit_code == 0, _out(ok)


def test_recipe_persists_success_and_failure_results(tmp_path: Path) -> None:
    dataset = tmp_path / "naming"
    make_case("naming", dataset)
    recipe = tmp_path / "recipe.json"
    result = tmp_path / "result.json"
    for params, expected_ok in [({"map": {"pMean": "p"}}, True), ({"invalid": 1}, False)]:
        recipe.write_text(json.dumps({"name": "rename", "steps": [
            {"op": "prep.fields.rename", "params": params},
        ]}))
        response = runner.invoke(app, ["recipe", "run", str(recipe), "-d", str(dataset),
                                       "--result-file", str(result), "--json"])
        assert response.exit_code == (0 if expected_ok else 2), _out(response)
        report = json.loads(result.read_text())
        assert report == json.loads(response.stdout)
        assert report["ok"] is expected_ok
        assert report["steps"][0]["status"] == ("ok" if expected_ok else "failed")
