# SPDX-License-Identifier: Apache-2.0
"""A tool writes records into a job's log and what it always printed into a terminal."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import pytest

from frontier_data import joblog
from frontier_data.board_run import BoardRun


@pytest.fixture
def as_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRONTIER_JOB_ID", "job-1")
    monkeypatch.delenv(joblog.FORMAT_ENV, raising=False)
    monkeypatch.setenv("FRONTIER_PIPELINE_STEP", "train")


@pytest.fixture
def in_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FRONTIER_JOB_ID", joblog.FORMAT_ENV, "FRONTIER_PIPELINE_STEP"):
        monkeypatch.delenv(name, raising=False)


def test_a_job_writes_records_and_a_terminal_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FRONTIER_JOB_ID", joblog.FORMAT_ENV):
        monkeypatch.delenv(name, raising=False)
    assert not joblog.job_mode()
    monkeypatch.setenv("FRONTIER_JOB_ID", "job-1")
    assert joblog.job_mode()
    # the setting decides when it is set, either way
    monkeypatch.setenv(joblog.FORMAT_ENV, "text")
    assert not joblog.job_mode()
    monkeypatch.delenv("FRONTIER_JOB_ID")
    monkeypatch.setenv(joblog.FORMAT_ENV, "JSON")
    assert joblog.job_mode()


def test_a_record_is_one_line_with_the_six_keys(as_job: None) -> None:
    line = joblog.record("slow loader\nsecond line", level="warn", logger="t", workers=2)
    assert "\n" not in line
    got = json.loads(line)
    assert list(got) == ["ts", "level", "logger", "step", "message", "fields"]
    assert got["ts"].endswith("Z") and len(got["ts"]) == 24
    assert (got["level"], got["logger"], got["step"]) == ("warn", "t", "train")
    assert got["message"] == "slow loader\nsecond line" and got["fields"] == {"workers": 2}
    # no step outside a pipeline run, no empty fields
    assert "step" not in json.loads(joblog.record("x", step=""))
    assert "fields" not in json.loads(joblog.record("x"))
    # what is a record and what is not
    assert joblog.is_record(line) == got
    assert joblog.message_of(line) == "slow loader\nsecond line"
    for other in (
        'frontier-event: {"type":"step.state"}',
        '{"ok": true, "message": "done"}',
        '{"ts": 1, "level": "info"}',
        "epoch 3 of 10",
        "{not json",
    ):
        assert joblog.is_record(other) is None and joblog.message_of(f" {other} ") == other


def test_logging_becomes_records_in_a_job(as_job: None, capsys: pytest.CaptureFixture[str]) -> None:
    joblog.setup(verbose=False)
    log = logging.getLogger("frontier_train.test")
    try:
        log.debug("not shown")
        log.info("epoch %d of %d", 3, 10, extra={"loss": 0.125})
        log.warning("slow")
        try:
            raise ValueError("bad shape")
        except ValueError:
            log.exception("step failed")
    finally:
        joblog.setup()  # one handler, however often it is called
    handlers = [h for h in logging.getLogger().handlers if getattr(h, "_frontier_joblog", False)]
    assert len(handlers) == 1
    logging.getLogger().removeHandler(handlers[0])
    lines = [json.loads(ln) for ln in capsys.readouterr().err.splitlines()]
    assert [(r["level"], r["message"]) for r in lines] == [
        ("info", "epoch 3 of 10"),
        ("warn", "slow"),
        ("error", "step failed"),
    ]
    assert lines[0]["logger"] == "frontier_train.test" and lines[0]["step"] == "train"
    assert lines[0]["fields"] == {"loss": 0.125}
    assert "ValueError: bad shape" in lines[2]["fields"]["exception"]


def test_logging_stays_readable_in_a_terminal(
    in_terminal: None, capsys: pytest.CaptureFixture[str]
) -> None:
    joblog.setup(console_format="%(name)s %(levelname)s %(message)s")
    try:
        logging.getLogger("frontier_train.test").info("epoch %d", 3)
    finally:
        for h in [h for h in logging.getLogger().handlers if getattr(h, "_frontier_joblog", False)]:
            logging.getLogger().removeHandler(h)
    assert capsys.readouterr().err == "frontier_train.test INFO epoch 3\n"


def test_status_lines_and_consoles(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rich.table import Table

    table = Table("check", "status")
    table.add_row("no_nan", "pass")
    for name in ("FRONTIER_JOB_ID", joblog.FORMAT_ENV, "FRONTIER_PIPELINE_STEP"):
        monkeypatch.delenv(name, raising=False)
    # a terminal: the words themselves, and rich's own console
    joblog.say("attached 3 files", logger="frontier_data.board_run", files=3)
    assert capsys.readouterr().out == "attached 3 files\n"
    assert type(joblog.console("frontier_hub")).__name__ == "Console"
    # a job: every print is one record; a table is the text it would show
    monkeypatch.setenv("FRONTIER_JOB_ID", "job-1")
    joblog.say("attached 3 files", logger="frontier_data.board_run", step="report", files=3)
    said = json.loads(capsys.readouterr().out)
    assert (said["logger"], said["step"], said["fields"]) == (
        "frontier_data.board_run",
        "report",
        {"files": 3},
    )
    out, err = joblog.console("frontier_hub"), joblog.console("frontier_hub", stderr=True)
    out.print("pulling [cyan]ahmedml[/cyan]: 12 files")
    out.print(table)
    out.print("")  # nothing to say: no record
    err.print("[red]no such entry[/red]")
    captured = capsys.readouterr()
    shown = [json.loads(ln) for ln in captured.out.splitlines()]
    assert [r["level"] for r in shown] == ["info", "info"]
    assert (
        shown[0]["message"] == "pulling ahmedml: 12 files" and shown[0]["logger"] == "frontier_hub"
    )
    assert "no_nan" in shown[1]["message"] and "\n" in shown[1]["message"]
    refused = json.loads(captured.err)
    assert (refused["level"], refused["message"]) == ("error", "no such entry")


def test_the_runner_reads_a_tools_last_record_as_one_sentence(
    as_job: None, tmp_path: Any, capfd: pytest.CaptureFixture[str]
) -> None:
    run = BoardRun({"nodes": [], "run": {"work_dir": str(tmp_path)}})
    wrapped = joblog.record("the entry has no converter:\nconvert it by hand", level="error")
    code, last = run._child(
        "import", [sys.executable, "-c", f"print({wrapped!r}); raise SystemExit(3)"]
    )
    assert code == 3 and last == "the entry has no converter: convert it by hand"
    assert run._reason("import", last) == last
    # the command the step started is a record of the runner, for that step
    first = json.loads(capfd.readouterr().out.splitlines()[0])
    assert first["logger"] == "frontier_data.board_run" and first["step"] == "import"
    assert first["message"].startswith("$ ")
    # the lines other programs read are not records
    plain = "error: no dataset\\nhint: pass --data"
    code, last = run._child(
        "train", [sys.executable, "-c", f"print('{plain}'); raise SystemExit(2)"]
    )
    assert run._reason("train", last) == "No dataset. Pass --data."
    assert "error: no dataset\nhint: pass --data\n" in capfd.readouterr().out
