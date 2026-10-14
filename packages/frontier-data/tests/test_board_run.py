# SPDX-License-Identifier: Apache-2.0
"""A board's run reports every step as `frontier-event:` lines (pipeline-run events v1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from frontier_data.board_run import EVENT_PREFIX, is_board_document, one_line, run_board


def _events(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.split(EVENT_PREFIX, 1)[1])
        for line in text.splitlines()
        if EVENT_PREFIX in line
    ]


def _states(events: list[dict[str, Any]], step: str) -> list[str]:
    return [e["to"] for e in events if e["type"] == "step.state" and e.get("step") == step]


def _doc(out: Path, steps: dict[str, Any] | None = None, **train: Any) -> dict[str, Any]:
    return {
        "format": "frontier-pipeline",
        "version": 1,
        "name": "demo",
        "nodes": [
            {"id": "src", "kind": "synthetic_source", "params": {"n": 3}, "inputs": []},
            {"id": "read", "kind": "vtk_reader", "params": {}, "inputs": ["src"]},
            {"id": "sink", "kind": "dataset_sink", "params": {"out_dir": ""}, "inputs": ["read"]},
            {"id": "train", "kind": "training_run", "params": train, "inputs": ["sink"]},
            {"id": "report", "kind": "report", "params": {}, "inputs": ["train"]},
        ],
        "run": {
            "id": "prun_t",
            "board": "Demo board",
            "work_dir": str(out / "work"),
            "datasets_dir": str(out / "datasets"),
            "steps": steps or {},
        },
    }


def test_steps_report_running_progress_and_outputs_and_a_failure_skips_what_follows(
    tmp_path: Path, capfd: Any, monkeypatch: Any
) -> None:
    from frontier_data import board_run

    def missing_config(self: Any, step: str, argv: list[str], env: Any = None) -> tuple[int, str]:
        self._tail[step] = ["configuration not found: no-such-config"]
        return 2, self._tail[step][-1]

    monkeypatch.setattr(board_run.BoardRun, "_child", missing_config)
    ok = run_board(_doc(tmp_path, config="no-such-config"), workers=1)
    events = _events(capfd.readouterr().out)
    assert not ok
    for step in ("src", "read", "sink"):
        assert _states(events, step) == ["running", "succeeded"], step
    assert any(e["type"] == "step.progress" and e["step"] == "read" for e in events)
    # the sink binds the dataset it wrote: an empty path on the board is the default place
    done = next(e for e in events if e.get("step") == "sink" and e.get("to") == "succeeded")
    out = done["data"]["outputs"][0]
    assert (out["port"], out["kind"]) == ("dataset", "dataset_version")
    assert out["name"] == str((tmp_path / "datasets" / "demo-board").resolve())
    assert (Path(out["name"]) / "manifest.json").is_file()
    assert _states(events, "train") == ["running", "failed"]
    failed = next(e for e in events if e.get("step") == "train" and e.get("to") == "failed")
    assert "no-such-config" in failed["data"]["error"]["line"]
    assert "\n" not in failed["data"]["error"]["line"]
    assert _states(events, "report") == ["skipped"]
    assert events[-1]["type"] == "run.finished" and events[-1]["data"]["state"] == "failed"
    assert events[-1]["data"]["summary"] == {"ok": 3, "failed": 1, "cached": 0, "skipped": 1}
    assert all(e["at"].endswith("Z") for e in events)


def test_reused_steps_are_not_run_and_their_outputs_feed_what_runs(
    tmp_path: Path, capfd: Any
) -> None:
    plan = {
        "src": {"action": "cached", "outputs": []},
        "read": {"action": "cached", "outputs": []},
        "sink": {
            "action": "cached",
            "outputs": [{"port": "dataset", "kind": "dataset_version", "name": "/nowhere"}],
        },
        "train": {"action": "run", "outputs": []},
        "report": {"action": "skip", "outputs": []},
    }
    ok = run_board(_doc(tmp_path, plan, config="no-such-config"), workers=1)
    events = _events(capfd.readouterr().out)
    assert not ok
    assert {e.get("step") for e in events if e["type"] == "step.state"} == {"train"}
    assert not (tmp_path / "datasets").exists(), "the data steps did not run again"
    assert events[-1]["data"]["summary"] == {"ok": 0, "failed": 1, "cached": 3, "skipped": 1}


def _trained(tmp_path: Path, monkeypatch: Any, **params: Any) -> tuple[list[list[str]], bool]:
    """Run only the training step of the board; returns the commands it started."""
    from frontier_data import board_run

    started: list[list[str]] = []

    def child(self: Any, step: str, argv: list[str], env: Any = None) -> tuple[int, str]:
        started.append(argv[2:])
        self._tail[step] = [
            "error: PhysicsNeMo is not installed, and this model needs it",
            "hint: install it with pip install 'frontier-physicsnemo[physicsnemo]'",
        ]
        return 2, self._tail[step][-1]

    monkeypatch.setattr(board_run.BoardRun, "_child", child)
    plan = {
        "src": {"action": "cached", "outputs": []},
        "read": {"action": "cached", "outputs": []},
        "sink": {
            "action": "cached",
            "outputs": [{"port": "dataset", "kind": "dataset_version", "name": "/data"}],
        },
        "train": {"action": "run", "outputs": []},
        "report": {"action": "skip", "outputs": []},
    }
    return started, run_board(_doc(tmp_path, plan, **params), workers=1)


def test_a_training_step_with_a_base_model_fine_tunes_from_it(
    tmp_path: Path, monkeypatch: Any, capfd: Any
) -> None:
    def line() -> str:
        failed = [e for e in _events(capfd.readouterr().out) if e.get("to") == "failed"]
        return failed[0]["data"]["error"]["line"]

    # from scratch: `train`, no base
    started, _ = _trained(tmp_path, monkeypatch, config="c")
    assert started[0][:2] == ["frontier_train", "train"] and "--base" not in started[0]
    # the tool's error and hint are the one line on the node
    assert line() == (
        "PhysicsNeMo is not installed, and this model needs it. "
        "Install it with pip install 'frontier-physicsnemo[physicsnemo]'."
    )
    # a hub model goes by its id
    started, _ = _trained(tmp_path, monkeypatch, config="c", base_source="hub", base_ref="m-1")
    assert started[0][:2] == ["frontier_train", "finetune"]
    assert started[0][started[0].index("--base") + 1] == "m-1"
    # a run or a package by the file the server found for it
    weights = tmp_path / "best.pt"
    started, _ = _trained(
        tmp_path, monkeypatch, base_source="run", base_ref="r-9", base_dir=str(weights)
    )
    assert started[0][started[0].index("--base") + 1] == str(weights)
    capfd.readouterr()
    # asked for and not chosen, or not on this server: the step says so and starts nothing
    started, ok = _trained(tmp_path, monkeypatch, base_source="hub")
    assert not ok and not started and line() == "No base model is chosen. Choose one on the node."
    started, ok = _trained(tmp_path, monkeypatch, base_source="package", base_ref="wing@1")
    assert not started
    assert line() == "The model package wing@1 is not on this server. Choose another."
    started, ok = _trained(tmp_path, monkeypatch, base_source="elsewhere", base_ref="x")
    assert not started and "is not known" in line()


def test_the_boundaries_of_a_served_board_pass_through(tmp_path: Path, capsys) -> None:
    doc = {
        "format": "frontier-pipeline",
        "version": 1,
        "name": "served",
        "nodes": [
            {"id": "request", "kind": "service.in", "params": {"name": "x"}, "inputs": []},
            {"id": "response", "kind": "service.out", "params": {}, "inputs": ["request"]},
        ],
        "run": {"id": "prun_s", "board": "Served", "work_dir": str(tmp_path), "steps": {}},
    }
    assert is_board_document(doc)
    assert run_board(doc)
    events = _events(capsys.readouterr().out)
    assert _states(events, "request") == ["running", "succeeded"]
    assert _states(events, "response") == ["running", "succeeded"]


def test_input_adapt_and_optimise_steps_run_the_tools_with_what_the_node_set(
    tmp_path: Path, monkeypatch: Any, capfd: Any
) -> None:
    from frontier_data import board_run

    started: list[list[str]] = []

    def child(self: Any, step: str, argv: list[str], env: Any = None) -> tuple[int, str]:
        started.append(argv[2:])
        # the tools leave what the step reads afterwards
        if argv[2] == "frontier_data":
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text("{}")
        if argv[2] == "frontier_train" and argv[3] == "infer":
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text("{}")
        if argv[2] == "frontier_train" and argv[3] == "optimise":
            out = Path(argv[argv.index("--out") + 1])
            (out / "optimum").mkdir(parents=True, exist_ok=True)
            (out / "optimum.json").write_text(
                json.dumps(
                    {
                        "best": {"value": 0.27, "params": {"slant_angle": 12.0}},
                        "n_evaluations": 9,
                        "geometry_dir": str(out / "optimum"),
                    }
                )
            )
        return 0, ""

    def contract(self: Any, step: str, model: str) -> Path:
        path = self._work(step) / "contract.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path

    monkeypatch.setattr(board_run.BoardRun, "_child", child)
    monkeypatch.setattr(board_run.BoardRun, "_contract_file", contract)
    monkeypatch.setattr(
        board_run.subprocess,
        "run",
        lambda *a, **k: type(
            "R",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps(
                    {
                        "problems": [
                            {"kind": "missing_parameter", "name": "reynolds", "fix": "set_value"}
                        ]
                    }
                ),
                "stderr": "",
            },
        )(),
    )
    train_out = {"port": "model", "kind": "run", "id": "r1", "name": str(tmp_path / "run")}
    doc = {
        "format": "frontier-pipeline",
        "version": 1,
        "name": "demo",
        "nodes": [
            {"id": "train", "kind": "training_run", "params": {}, "inputs": []},
            {
                "id": "input",
                "kind": "input",
                "params": {
                    "geometry_source": "shape",
                    "geometry": "ahmed",
                    "fields": '{"slant_angle": 25}',
                    "resolution": 16,
                },
                "inputs": ["train"],
            },
            {
                "id": "adapt",
                "kind": "adapt",
                "params": {"values": "reynolds=1e6"},
                "inputs": ["input", "train"],
            },
            {
                "id": "infer",
                "kind": "inference",
                "params": {},
                "inputs": ["adapt", "train"],
            },
            {
                "id": "opt",
                "kind": "optimise",
                "params": {
                    "geometry": "ahmed",
                    "free": "slant_angle",
                    "budget": 9,
                    "checks": '[{"check": "cp_range"}]',
                    "method": "auto",
                },
                "inputs": ["train"],
            },
        ],
        "run": {
            "id": "prun_t",
            "board": "Demo",
            "work_dir": str(tmp_path / "work"),
            "datasets_dir": str(tmp_path / "ds"),
            "steps": {"train": {"action": "cached", "outputs": [train_out]}},
        },
    }
    ok = run_board(doc, workers=1)
    events = _events(capfd.readouterr().out)
    assert ok, [e for e in events if e.get("to") == "failed"]
    by = {argv[0] + " " + argv[1]: argv for argv in started}
    inputs = by["frontier_data inputs"]
    assert inputs[inputs.index("--geometry") + 1] == "ahmed" and "--contract" in inputs
    assert inputs[inputs.index("--parameters") + 1] == '{"slant_angle": 25}'
    adapt = by["frontier_prep adapt"]
    assert adapt[adapt.index("--set") + 1] == "reynolds=1e6" and "--fixes" in adapt
    # the typed case is predicted without the training gate: it is not training data
    infer = by["frontier_train infer"]
    assert infer[infer.index("--data") + 1].endswith("adapted") and "--allow-ungated" in infer
    optimise = by["frontier_train optimise"]
    assert optimise[optimise.index("--free") + 1] == "slant_angle"
    request = json.loads(optimise[optimise.index("--json") + 1])
    assert request["budget"] == 9 and request["checks"] == [{"check": "cp_range"}]

    # what each step bound
    def outputs(step: str) -> list[dict[str, Any]]:
        return next(
            e["data"]["outputs"]
            for e in events
            if e.get("step") == step and e.get("to") == "succeeded"
        )

    assert outputs("input")[0]["kind"] == "dataset_version"
    assert outputs("adapt")[0]["name"].endswith("adapted")
    ports = {o["port"]: o for o in outputs("opt")}
    assert set(ports) == {"metrics", "optimum", "dataset"} and ports["metrics"]["kind"] == "run"
    assert ports["dataset"]["name"].endswith("optimum")
    best = [e for e in events if e["type"] == "step.metric" and e.get("step") == "opt"]
    assert {m["data"]["key"]: m["data"]["value"] for m in best} == {
        "optimise/best": 0.27,
        "optimise/evaluations": 9,
    }


def test_plain_pipelines_stay_plain_and_error_lines_are_one_line() -> None:
    assert not is_board_document({"nodes": [{"id": "a", "kind": "lod"}]})
    assert is_board_document({"nodes": [{"id": "a", "kind": "training_run"}]})
    assert is_board_document({"nodes": [], "run": {}})
    assert one_line("Traceback\n  File x\nValueError: bad split\n") == "ValueError: bad split"


def test_finished_run_sends_undefined_metrics_as_null(tmp_path, monkeypatch):
    from frontier._server import ServerClient
    from frontier_data.board_run import BoardRun

    updates = []
    monkeypatch.setenv("FRONTIER_SERVER", "http://localhost:8787")
    monkeypatch.setattr(
        ServerClient, "patch_run", lambda self, run, **kwargs: updates.append(kwargs)
    )
    BoardRun(_doc(tmp_path), workers=1)._finish(
        "run", {"metric/r2": float("nan"), "metric/mae": 0.1}
    )
    assert updates == [{"state": "finished", "summary": {"metric/r2": None, "metric/mae": 0.1}}]
