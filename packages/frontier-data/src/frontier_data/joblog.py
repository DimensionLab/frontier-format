# SPDX-License-Identifier: Apache-2.0
"""Log records for job logs.

A tool that runs as a job of a worker writes its log as JSON lines, one record per line::

    {"ts": "2026-10-13T08:35:12.500Z", "level": "info", "logger": "frontier_train",
     "step": "train", "message": "epoch 3 of 10", "fields": {"loss": 0.12}}

so the log of a job can be filtered by level, tool and step without guessing at text
(``docs/spec/api.md``, "Job and step logs as pages"). The same tool in a terminal prints
what it always printed: records are for logs, not for people at a prompt.

Which of the two applies is :func:`job_mode`: the setting ``FRONTIER_LOG_FORMAT``
(``json`` or ``text``) when it is set, else JSON exactly when the process is a job
(``FRONTIER_JOB_ID`` is in its environment, which a worker sets).

What stays as it is in both modes, because other programs read it: ``frontier-event:``
lines, the ``error:`` / ``hint:`` lines a command ends with, a ``frontier-result:`` line,
and the JSON a command prints as its result.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import IO, Any

FORMAT_ENV = "FRONTIER_LOG_FORMAT"
JOB_ENV = "FRONTIER_JOB_ID"
STEP_ENV = "FRONTIER_PIPELINE_STEP"
CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
#: the level words of a record (``docs/spec/api.md``: trace, debug, info, warn, error)
LEVELS = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warn",
    "ERROR": "error",
    "CRITICAL": "error",
}
# attributes every LogRecord has: whatever else is on one was passed as `extra`
_STANDARD = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


def job_mode() -> bool:
    """Whether this process writes its log as records (see the module text)."""
    chosen = os.environ.get(FORMAT_ENV, "").strip().lower()
    if chosen in ("json", "text"):
        return chosen == "json"
    return bool(os.environ.get(JOB_ENV))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def record(
    message: str,
    *,
    level: str = "info",
    logger: str = "frontier",
    step: str | None = None,
    ts: str | None = None,
    **fields: Any,
) -> str:
    """One record as its line: a JSON object without line breaks. ``step`` defaults to the
    pipeline step this process runs as (``FRONTIER_PIPELINE_STEP``)."""
    out: dict[str, Any] = {"ts": ts or _now(), "level": level, "logger": logger}
    step = step if step is not None else os.environ.get(STEP_ENV)
    if step:
        out["step"] = step
    out["message"] = message
    if fields:
        out["fields"] = fields
    return json.dumps(out, separators=(",", ":"), default=str, ensure_ascii=False)


def is_record(line: str) -> dict[str, Any] | None:
    """The record a line holds, or ``None`` when the line is anything else (text, an event,
    the JSON a command printed as its result)."""
    text = line.strip()
    if not (text.startswith('{"ts":') and text.endswith("}")):
        return None
    try:
        found = json.loads(text)
    except ValueError:
        return None
    if isinstance(found, dict) and {"ts", "level", "message"} <= found.keys():
        return found
    return None


def message_of(line: str) -> str:
    """What a line says: the message of a record, else the line."""
    found = is_record(line)
    return str(found["message"]) if found is not None else line.strip()


class JsonLines(logging.Formatter):
    """A ``logging`` record as a line of the job log. Values passed as ``extra`` become
    ``fields``; an exception goes to ``fields.exception`` so the record stays one line."""

    def format(self, record: logging.LogRecord) -> str:
        fields = {k: v for k, v in vars(record).items() if k not in _STANDARD}
        if record.exc_info:
            fields["exception"] = self.formatException(record.exc_info)
        ts = (
            datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        return as_line(
            record.getMessage(),
            level=LEVELS.get(record.levelname, record.levelname.lower()),
            logger=record.name,
            ts=ts,
            **fields,
        )


def setup(verbose: bool = False, *, console_format: str = CONSOLE_FORMAT) -> None:
    """Configure the root logger of a command line: records in a job, ``console_format``
    in a terminal. Replaces handlers a previous call installed."""
    root = logging.getLogger()
    for old in [h for h in root.handlers if getattr(h, "_frontier_joblog", False)]:
        root.removeHandler(old)
    handler = logging.StreamHandler()
    handler._frontier_joblog = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonLines() if job_mode() else logging.Formatter(console_format))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def say(
    message: str,
    *,
    logger: str = "frontier",
    level: str = "info",
    step: str | None = None,
    file: IO[str] | None = None,
    **fields: Any,
) -> None:
    """Print one line of status: a record in a job, the message itself in a terminal."""
    out = file or sys.stdout
    if job_mode():
        line = record(message, level=level, logger=logger, step=step, **fields)
    else:
        line = message
    out.write(line + "\n")
    out.flush()


def console(logger: str, *, stderr: bool = False) -> Any:
    """The console a command line prints its status with: rich's own in a terminal, and
    in a job one whose every ``print`` is a record of ``logger`` (level ``error`` for the
    error console). Tables and panels become the text they would show."""
    from rich.console import Console

    if not job_mode():
        return Console(stderr=stderr)

    class Records(Console):
        def __init__(self) -> None:
            super().__init__(
                stderr=stderr,
                force_terminal=False,
                no_color=True,
                color_system=None,
                width=160,
                highlight=False,
            )

        def print(self, *objects: Any, **options: Any) -> None:  # noqa: A003
            with self.capture() as shown:
                super().print(*objects, **options)
            text = shown.get().rstrip()
            if not text.strip():
                return
            level = "error" if stderr else "info"
            self.file.write(record(text, level=level, logger=logger) + "\n")
            self.file.flush()

    return Records()


__all__ = [
    "CONSOLE_FORMAT",
    "FORMAT_ENV",
    "JsonLines",
    "console",
    "is_record",
    "job_mode",
    "message_of",
    "record",
    "say",
    "setup",
]


#: :func:`record`, by a name a ``logging.Formatter`` method does not shadow
as_line = record
