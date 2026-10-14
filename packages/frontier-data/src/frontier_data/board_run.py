# SPDX-License-Identifier: Apache-2.0
"""Run the steps of a board and report each one as pipeline-run events.

The document is the ``frontier-pipeline`` JSON the canvas exports, with a ``run`` section
the server adds::

    {"nodes": [{"id", "kind", "params", "inputs"}], "edges": [...],
     "run": {"id", "project", "board", "work_dir", "datasets_dir",
             "steps": {"<id>": {"action": "run" | "cached" | "skip", "outputs": [Output]}}}}

Steps of the kinds in :data:`frontier_data.pipeline.REGISTRY` run in-process through
:class:`~frontier_data.pipeline.PipelineRunner`. ``training_run``, ``inference``, ``evaluate``
and ``report`` steps run ``python -m frontier_train train`` / ``finetune`` / ``infer`` and
``python -m frontier_eval`` as child processes, whose own event lines pass through.

A ``training_run`` step with a base model (``base_source`` ``hub``, ``package`` or ``run``
with ``base_ref``; the server resolves a package or a run to ``base_dir``) fine-tunes from it.

An ``inference`` step predicts with the model of the training run wired into it, or with a
run directory or a model package named in its parameters, and binds a ``predictions`` output
held by a run of its own on the server. An ``evaluate`` step scores predictions that reach
it, else the training run that reaches it, else the model its parameters name.

Events are ``frontier-event: {json}`` lines on stdout (``docs/spec/pipeline-run-events.md``,
section 7): the worker forwards a job's output to the server, which numbers and stores
them. Without a server the lines are simply part of the output.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from frontier_data import joblog
from frontier_data.pipeline import REGISTRY, Pipeline, PipelineEvent, PipelineRunner

EVENT_PREFIX = "frontier-event: "
# Progress of one step is reported at most this often.
PROGRESS_INTERVAL_S = 0.25
STEP_KINDS = (
    "import",
    "recipe",
    "training_run",
    "inference",
    "evaluate",
    "report",
    "input",
    "adapt",
    "optimise",
)
# The boundaries of a served board (`service.in`, `service.out`): a board run passes
# through them without doing anything; the service runtime is what consumes them.
BOUNDARY_KINDS = ("service.in", "service.out")
LOGGER = "frontier_data.board_run"


def say(message: str, step: str | None = None, level: str = "info", **fields: Any) -> None:
    """One line of the runner's own in the log: a record in a job, the message in a
    terminal (``frontier_data.joblog``)."""
    joblog.say(message, logger=LOGGER, level=level, step=step, **fields)


NO_MODEL = "No model reaches this step. Wire a training run into it, or choose a model."
BASE_SOURCES = ("hub", "package", "run")


def emit(kind: str, step: str | None = None, **fields: Any) -> None:
    event: dict[str, Any] = {
        "type": kind,
        "at": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    }
    if step is not None:
        event["step"] = step
    event.update({k: v for k, v in fields.items() if v is not None})
    sys.stdout.write(EVENT_PREFIX + json.dumps(event, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


def one_line(text: str, limit: int = 300) -> str:
    """The line of an error message a node can show."""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return (lines[-1] if lines else "The step failed without a message")[:limit]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-").lower() or "board"


class BoardRun:
    def __init__(self, doc: dict[str, Any], workers: int = 4) -> None:
        self.doc = doc
        self.workers = workers
        self.info: dict[str, Any] = doc.get("run") or {}
        self.nodes: list[dict[str, Any]] = [dict(n) for n in doc.get("nodes", [])]
        self.by_id = {n["id"]: n for n in self.nodes}
        plan: dict[str, Any] = self.info.get("steps") or {}
        self.action = {n["id"]: (plan.get(n["id"]) or {}).get("action", "run") for n in self.nodes}
        # outputs by step: reused ones from the plan, the rest as steps finish
        self.outputs: dict[str, list[dict[str, Any]]] = {
            sid: list(p.get("outputs") or []) for sid, p in plan.items()
        }
        self.state: dict[str, str] = {}
        self.started: dict[str, float] = {}
        self._last_progress: dict[str, float] = {}
        # the last two lines a step's tool printed: its `error:` and `hint:` when it failed
        self._tail: dict[str, list[str]] = {}
        self.t0 = time.perf_counter()

    # -- reporting ---------------------------------------------------------------------------
    def running(self, step: str) -> None:
        if self.state.get(step) is None:
            self.state[step] = "running"
            self.started[step] = time.perf_counter()
            emit("step.state", step, to="running", data={"attempt": 1})

    def progress(self, step: str, done: int = 0, total: int = 0, message: str = "") -> None:
        now = time.perf_counter()
        last = done >= total > 0
        if not last and now - self._last_progress.get(step, 0.0) < PROGRESS_INTERVAL_S:
            return
        self._last_progress[step] = now
        data: dict[str, Any] = {}
        if total > 0:
            data.update(fraction=min(1.0, done / total), done=done, total=total)
        if message:
            data["message"] = message[:200]
        elif total > 0:
            data["message"] = f"{done} of {total}"
        if data:
            emit("step.progress", step, data=data)

    def succeeded(self, step: str, outputs: list[dict[str, Any]] | None = None) -> None:
        if self.state.get(step) in ("succeeded", "failed", "skipped"):
            return
        self.running(step)
        self.state[step] = "succeeded"
        if outputs:
            self.outputs[step] = outputs
        seconds = round(time.perf_counter() - self.started[step], 3)
        emit(
            "step.state",
            step,
            to="succeeded",
            data={"outputs": self.outputs.get(step, []), "seconds": seconds},
        )

    def failed(self, step: str, error: str) -> None:
        if self.state.get(step) in ("succeeded", "failed", "skipped"):
            return
        self.running(step)
        self.state[step] = "failed"
        job = os.environ.get("FRONTIER_JOB_ID", "")
        emit(
            "step.state",
            step,
            to="failed",
            cause={"kind": "error"},
            data={"error": {"line": one_line(error), "log_ref": f"jobs/{job}" if job else ""}},
        )

    def skipped(self, step: str, because: str) -> None:
        if self.state.get(step) is None:
            self.state[step] = "skipped"
            emit(
                "step.state",
                step,
                to="skipped",
                cause={"kind": "upstream"},
                data={"because": because},
            )

    def usable(self, step: str) -> bool:
        """The outputs of `step` can be consumed: it succeeded now, or is reused."""
        return self.state.get(step) == "succeeded" or self.action.get(step) == "cached"

    # -- data steps --------------------------------------------------------------------------
    def run_data(self) -> None:
        data_nodes = [n for n in self.nodes if n["kind"] in REGISTRY]
        mine = [n for n in data_nodes if self.action[n["id"]] == "run"]
        if not mine:
            return
        report = {n["id"] for n in mine}
        context: dict[str, Any] = {}
        sinks = [n for n in data_nodes if n["kind"] == "dataset_sink"]
        out_dir = next(
            (n["params"].get("out_dir") for n in sinks if n["params"].get("out_dir")), None
        )
        if not out_dir:
            root = Path(self.info.get("datasets_dir") or "datasets")
            name = next((n["params"].get("name") for n in sinks if n["params"].get("name")), None)
            out_dir = str(root / _slug(str(name or self.info.get("board") or self.doc.get("name"))))
        out_dir = str(Path(out_dir).expanduser().resolve())
        context["out_dir"] = out_dir
        for n in data_nodes:
            # an empty path on the board means "the default", not the working directory
            n["params"] = {k: v for k, v in n["params"].items() if v not in ("", None)}
        pipeline = Pipeline.model_validate(
            {
                "name": self.doc.get("name") or "board",
                "nodes": [
                    {
                        "id": n["id"],
                        "kind": n["kind"],
                        "params": n["params"],
                        "inputs": [i for i in n.get("inputs", []) if i in self.by_id],
                    }
                    for n in data_nodes
                ],
            }
        )
        # steps with a dataset phase end there, not when the last sample passed through
        late = {
            n["id"]
            for n in data_nodes
            if REGISTRY[n["kind"]].scope == "dataset" or REGISTRY[n["kind"]].finalize is not None
        }
        item_errors: dict[str, list[str]] = {}
        sample_done: set[str] = set()

        def dataset_output() -> list[dict[str, Any]]:
            return [
                {"port": "dataset", "kind": "dataset_version", "name": out_dir, **_version(out_dir)}
            ]

        def on_event(ev: PipelineEvent) -> None:
            sid = ev.node_id
            if sid not in report:
                return
            if ev.status == "start":
                self.running(sid)
            elif ev.status == "progress":
                self.running(sid)
                self.progress(sid, ev.done, ev.total, ev.message)
            elif ev.status == "error":
                if ev.item_id is not None:
                    item_errors.setdefault(sid, []).append(f"{ev.item_id}: {ev.message}")
                else:
                    self.failed(sid, ev.message)
            elif ev.status == "done":
                if item_errors.get(sid):
                    errs = item_errors[sid]
                    self.failed(sid, f"{len(errs)} of {ev.total} samples failed ({errs[0]})")
                elif (
                    sid in late
                    and sid not in sample_done
                    and REGISTRY[self.by_id[sid]["kind"]].scope != "dataset"
                ):
                    sample_done.add(sid)  # its dataset phase follows
                else:
                    is_sink = self.by_id[sid]["kind"] == "dataset_sink"
                    self.succeeded(sid, dataset_output() if is_sink else None)

        try:
            result = PipelineRunner(workers=self.workers, on_event=on_event).run(pipeline, context)
        except Exception as exc:  # the source, or the pipeline itself, is wrong
            blamed = next((s for s in report if self.state.get(s) == "running"), None)
            blamed = blamed or next(iter(n["id"] for n in mine))
            self.failed(blamed, f"{type(exc).__name__}: {exc}")
            for n in mine:
                if self.state.get(n["id"]) == "running":
                    self.failed(n["id"], f"{type(exc).__name__}: {exc}")
                self.skipped(n["id"], f"{blamed} failed")
            return
        for err in result.errors:
            if err.get("node_id") in report:
                self.failed(err["node_id"], err["error"])
        failed = [s for s in report if self.state.get(s) == "failed"]
        for n in mine:
            sid = n["id"]
            if self.state.get(sid) == "running" and not failed:
                is_sink = n["kind"] == "dataset_sink"
                self.succeeded(sid, dataset_output() if is_sink else None)
            elif failed and self.state.get(sid) in (None, "running"):
                if self.state.get(sid) == "running":
                    self.failed(sid, f"{failed[0]} failed")
                self.skipped(sid, f"{failed[0]} failed")
        # splits and quality were written after the sink closed: tell the server again so
        # the registered dataset has them
        for n in sinks:
            if self.state.get(n["id"]) == "succeeded":
                emit("step.output", n["id"], data={"output": dataset_output()[0]})

    # -- steps that run another tool ---------------------------------------------------------
    def _input_outputs(self, node: dict[str, Any]) -> list[dict[str, Any]]:
        return [o for i in node.get("inputs", []) for o in self.outputs.get(i, [])]

    def _child(
        self, step: str, argv: list[str], env_extra: dict[str, str] | None = None
    ) -> tuple[int, str]:
        """Run a tool as this step: its event lines pass through and bind outputs; the
        rest of its output is the log. Returns the exit code and the last output line."""
        env = dict(os.environ)
        env["FRONTIER_PIPELINE_STEP"] = step
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("FRONTIER_RUN_ID", None)
        env.update(env_extra or {})
        say("$ " + " ".join(argv[1:] if argv[0] == sys.executable else argv), step)
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1
        )
        last = ""
        tail = self._tail[step] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if EVENT_PREFIX in line:
                try:
                    ev = json.loads(line.split(EVENT_PREFIX, 1)[1])
                except ValueError:
                    continue
                out = (ev.get("data") or {}).get("output")
                if ev.get("type") == "step.output" and isinstance(out, dict):
                    kept = [
                        o for o in self.outputs.get(step, []) if o.get("port") != out.get("port")
                    ]
                    self.outputs[step] = [*kept, out]
            elif line.strip():
                # a record says it in its message; one that wraps is one sentence
                written = joblog.is_record(line)
                last = (
                    line.strip() if written is None else " ".join(str(written["message"]).split())
                )
                tail[:] = [*tail[-1:], last]
        return proc.wait(), last

    def _reason(self, step: str, last: str) -> str:
        """Why a tool failed, in one line: its `error:` line and its `hint:` as sentences
        when it ended with them, else the last line it printed."""
        told = dict(
            ln.split(": ", 1)
            for ln in self._tail.get(step, [])
            if ln.startswith(("error: ", "hint: "))
        )
        if "error" not in told:
            return last
        parts = [told["error"], *([told["hint"]] if "hint" in told else [])]
        return " ".join(
            t[:1].upper() + t[1:] + ("" if t.endswith((".", "!", "?")) else ".") for t in parts
        )

    def _tools(
        self, step: str, what: str, commands: list[tuple[list[str], dict[str, str]]]
    ) -> bool:
        """Run the tools of a step one after the other; the first that fails fails the step."""
        for args, env in commands:
            code, last = self._child(step, [sys.executable, "-m", *args], env)
            if code != 0:
                self.failed(step, self._reason(step, last) or f"{what} ended with exit code {code}")
                return False
        return True

    def _work(self, step: str) -> Path:
        return Path(self.info.get("work_dir") or "pipeline-runs") / step

    def _dataset_of(self, node: dict[str, Any]) -> str | None:
        """The dataset directory wired into a step, else the one its parameters name."""
        wired = next(
            (
                o.get("name")
                for o in self._input_outputs(node)
                if str(o.get("kind", "")).startswith("dataset")
            ),
            None,
        )
        found = wired or node["params"].get("dataset_dir")
        return str(found) if found else None

    def _typed_cases(self, node: dict[str, Any]) -> bool:
        """Whether the dataset wired into a step was typed on the board (an Input step,
        or an Adapt step after one) rather than imported."""
        for i in node.get("inputs", []):
            up = self.by_id.get(i)
            if up is None or up["kind"] not in ("input", "adapt"):
                continue
            if any(str(o.get("kind", "")).startswith("dataset") for o in self.outputs.get(i, [])):
                return up["kind"] == "input" or self._typed_cases(up)
        return False

    def _dataset_in(self, node: dict[str, Any]) -> str | None:
        """The dataset directory that reaches a step: an upstream step's, else a named one.
        The step fails when there is none."""
        found = self._dataset_of(node)
        if not found:
            named = node["params"].get("dataset")
            self.failed(
                node["id"],
                f"Dataset {named} is not on this server. Choose a registered dataset."
                if named
                else "No dataset reaches this step. Wire a dataset into it.",
            )
        return found

    def _dataset_out(self, directory: str) -> list[dict[str, Any]]:
        return [
            {"port": "dataset", "kind": "dataset_version", "name": directory, **_version(directory)}
        ]

    def run_import(self, node: dict[str, Any]) -> None:
        step, p = node["id"], node["params"]
        source, ref = str(p.get("source") or "example"), str(p.get("ref") or "").strip()
        if source == "dataset":
            found = p.get("dataset_dir")
            if not found:
                self.failed(step, f"Dataset {ref or '(none)'} is not on this server. Choose one.")
                return
            self.succeeded(step, self._dataset_out(str(found)))
            return
        if not ref:
            self.failed(step, "Nothing to import. Set what to import in the inspector.")
            return
        root = Path(self.info.get("datasets_dir") or "datasets")
        name = str(p.get("name") or "").strip()
        own = name or Path(ref.rstrip("/")).name
        out = (root / _slug(own if source == "connector" else ref)).resolve()
        if (out / "manifest.json").is_file():
            # imported by an earlier run: the dataset, with its versions, is kept
            self.progress(step, message="already imported")
            self.succeeded(step, self._dataset_out(str(out)))
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        if source == "example":
            commands = [["frontier_examples", "install", ref, "--out", str(out)]]
        elif source == "hub":
            pull = ["frontier_hub", "pull", ref, "--convert", "--out", str(out), "--yes"]
            if p.get("subset"):
                pull += ["--subset", str(p["subset"])]
            commands = [pull]
        elif source == "attach":
            # files the user already has for a hub entry: checked against it, then converted
            files = str(p.get("files") or "").strip()
            if not files:
                self.failed(step, "No files are named. Set where they are in the inspector.")
                return
            attach = ["frontier_hub", "attach", ref, files, "--out", str(out)]
            attach += ["--to", str(root / "_raw" / _slug(ref))]
            if p.get("subset"):
                attach += ["--subset", str(p["subset"])]
            if p.get("mode"):
                attach += ["--mode", str(p["mode"])]
            commands = [attach]
        elif source == "connector":
            cases = [("--cases", p.get("cases")), ("--case-params", p.get("case_params"))]
            extra = [str(x) for flag, value in cases if value for x in (flag, value)]
            if p.get("connector"):
                extra += ["--connector", str(p["connector"])]
            if "://" in ref and not ref.startswith("file://"):
                # an object store or a URL: fetched, then read with the connector
                attach = ["frontier_hub", "attach", "--name", name or out.name, *extra]
                attach += ["--to", str(root / "_raw" / out.name), "--out", str(out), ref]
                if p.get("mode"):
                    attach += ["--mode", str(p["mode"])]
                commands = [attach]
            else:
                convert = ["frontier_connectors", "convert", ref, "--out", str(out), *extra]
                if name:
                    convert += ["--name", name]
                commands = [convert]
        else:
            self.failed(step, f"Import from {source} is not known. Choose another source.")
            return
        if not self._tools(step, "The import", [(c, {}) for c in commands]):
            return
        if not (out / "manifest.json").is_file():
            self.failed(step, f"The import left no dataset in {out}. Open the job for its log.")
            return
        self.succeeded(step, self._dataset_out(str(out)))

    def run_recipe(self, node: dict[str, Any]) -> None:
        step, p = node["id"], node["params"]
        dataset = self._dataset_in(node)
        if not dataset:
            return
        try:
            recipe = json.loads(str(p.get("text") or "").strip() or "{}")
        except ValueError as exc:
            self.failed(step, f"The recipe is not valid JSON ({exc}). Fix it in the node.")
            return
        if not isinstance(recipe, dict) or not recipe.get("steps"):
            self.failed(step, "The recipe has no steps. Add one in the node.")
            return
        recipe.setdefault("name", str(p.get("name") or "recipe"))
        work = self._work(step)
        work.mkdir(parents=True, exist_ok=True)
        path = work / "recipe.json"
        path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        commands: list[tuple[list[str], dict[str, str]]] = []
        if not (Path(dataset) / ".frontier").is_dir():
            commands.append((["frontier_prep", "versions", "init", dataset], {}))
        result = work / "result.json"
        commands.append(
            (
                ["frontier_data.board_run", "recipe", str(path), dataset, str(result)],
                {},
            )
        )
        if not self._tools(step, "The recipe", commands):
            return
        try:
            done = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            done = {}
        stopped = next(
            (s for s in done.get("steps", []) if s.get("status") in ("failed", "blocked")),
            None,
        )
        if not done.get("ok", False):
            why = (stopped or {}).get("error") or "it did not finish"
            op = (stopped or {}).get("op") or "the recipe"
            self.failed(step, f"{op}: {one_line(why)}")
            return
        self.succeeded(step, self._dataset_out(dataset))

    def _base_in(self, node: dict[str, Any]) -> tuple[bool, str | None]:
        """The model a training step starts from: `(asked for one, what to pass as --base)`.
        A hub model goes by its id; a model package or a run by the file the server found
        for it (`base_dir`), else by `base_ref` when that is a path on this machine."""
        p = node["params"]
        source = str(p.get("base_source") or "none")
        if source in ("", "none"):
            return False, None
        ref = str(p.get("base_ref") or "").strip()
        if source not in BASE_SOURCES:
            self.failed(
                node["id"], f"Base model source {source} is not known. Choose hub, package or run."
            )
            return True, None
        if not ref:
            self.failed(node["id"], "No base model is chosen. Choose one on the node.")
            return True, None
        if source == "hub":
            return True, ref
        found = str(p.get("base_dir") or "") or (ref if Path(ref).exists() else "")
        if not found:
            what = "model package" if source == "package" else "run"
            self.failed(node["id"], f"The {what} {ref} is not on this server. Choose another.")
            return True, None
        return True, found

    def run_training(self, node: dict[str, Any]) -> None:
        step, p = node["id"], node["params"]
        dataset = self._dataset_in(node)
        if not dataset:
            return
        finetune, base = self._base_in(node)
        if finetune and base is None:
            return
        argv = ["frontier_train", "finetune" if finetune else "train"]
        argv += ["--config", str(p.get("config") or "surface_pointmlp"), "--data", str(dataset)]
        if base:
            argv += ["--base", base]
        sets: dict[str, Any] = {
            "train.out_dir": str(self._work(step)),
            "logging.run_name": f"{_slug(str(self.info.get('board') or 'board'))}-{step}",
        }
        if self.info.get("project"):
            sets["logging.project"] = self.info["project"]
        if p.get("epochs"):
            sets["train.epochs"] = int(p["epochs"])
        for item in re.split(r"[,\n]+", str(p.get("overrides") or "")):
            if "=" in item:
                argv += ["--set", item.strip()]
        for key, value in sets.items():
            argv += ["--set", f"{key}={json.dumps(value)}"]
        if p.get("allow_ungated"):
            argv.append("--allow-ungated")
        code, last = self._child(step, [sys.executable, "-m", *argv])
        why = self._reason(step, last)
        if code == 0:
            self.succeeded(step)
        elif "train-ready" in why or "not versioned" in why:
            # the gate refused the dataset: say what to do about it on the node
            self.failed(
                step,
                "The dataset has not passed the quality gate. Add a Recipe step in front of "
                "this one, or train without the gate.",
            )
        else:
            self.failed(step, why or f"Training ended with exit code {code}")

    # -- runs on the server that hold what a step made -----------------------------------------
    def _hold(self, step: str, tags: list[str], config: dict[str, Any]) -> str:
        """Create the run that holds a step's results. '' without a server, and when the
        run cannot be created: the files on disk stay the source of truth."""
        server = os.environ.get("FRONTIER_SERVER")
        if not server:
            return ""
        from frontier._server import ServerClient

        try:
            client = ServerClient(server)
            project = self.info.get("project_id") or client.resolve_project(
                str(self.info.get("project") or "frontier-train")
            )
            name = f"{_slug(str(self.info.get('board') or 'board'))}-{step}"
            return str(client.create_run(str(project), name, config, tags)["id"])
        except Exception as exc:  # noqa: BLE001 - best effort, the step goes on
            say(
                f"no run on the server holds the results of {step}: {one_line(str(exc))}",
                step,
                "warn",
            )
            return ""

    def _finish(self, run_id: str, summary: dict[str, Any]) -> None:
        server = os.environ.get("FRONTIER_SERVER")
        if not server or not run_id:
            return
        from frontier._server import ServerClient

        try:
            ServerClient(server).patch_run(run_id, state="finished", summary=summary)
        except Exception as exc:  # noqa: BLE001 - best effort
            say(f"could not mark run {run_id} finished: {one_line(str(exc))}", level="warn")

    def _attach(self, run_id: str) -> dict[str, str]:
        """Results go to this run when there is a server to take them."""
        if run_id and os.environ.get("FRONTIER_SERVER"):
            return {"FRONTIER_RUN_ID": run_id}
        return {}

    # -- inference -----------------------------------------------------------------------------
    def _model_in(self, node: dict[str, Any]) -> str | None:
        """The model a step predicts with: the directory of the training run wired into it
        (`model_source` `step`), or the run directory, checkpoint or package its parameters
        name (`run`, `package`). The step fails when there is none."""
        step, p = node["id"], node["params"]
        source = str(p.get("model_source") or "step")
        if source == "step":
            run = next((o for o in self._input_outputs(node) if o.get("kind") == "run"), None)
            if run is not None and run.get("name"):
                return str(run["name"])
        elif source in ("run", "package"):
            if p.get("model_dir"):
                return str(p["model_dir"])
        else:
            self.failed(step, f"Model source {source} is not known. Choose step, run or package.")
            return None
        self.failed(step, NO_MODEL)
        return None

    def _infer(self, node: dict[str, Any], model: str, out: Path) -> dict[str, Any] | None:
        """Run `frontier_train infer` as this step. Returns the manifest it wrote."""
        step, p = node["id"], node["params"]
        argv = ["frontier_train", "infer", "--model", model]
        dataset = self._dataset_of(node)
        ids = p.get("ids")
        ids = ",".join(str(i) for i in ids) if isinstance(ids, list | tuple) else str(ids or "")
        for flag, value in (
            ("--checkpoint", p.get("checkpoint")),
            ("--data", dataset),
            ("--split", p.get("split")),
            ("--ids", ids.strip()),
            ("--first", int(p["first"]) if p.get("first") else None),
            ("--device", p.get("device")),
        ):
            if value:
                argv += [flag, str(value)]
        if p.get("allow_ungated") or self._typed_cases(node):
            # the quality gate is for training data: cases typed on the board are not
            argv.append("--allow-ungated")
        origin = {"pipeline_run": self.info.get("id"), "board": self.info.get("board")}
        origin = {"step": step, **{k: v for k, v in origin.items() if v}}
        argv += ["--out", str(out), "--json", json.dumps({"provenance": origin})]
        code, last = self._child(step, [sys.executable, "-m", *argv])
        if code != 0:
            why = self._reason(step, last)
            if "train-ready" in why:
                why = (
                    "The dataset has not passed the quality gate. Add a Recipe step in front "
                    "of this one, or predict without the gate."
                )
            self.failed(step, why or f"Inference ended with exit code {code}")
            return None
        try:
            return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.failed(step, f"Inference left no predictions in {out}. Open the job for its log.")
            return None

    @staticmethod
    def _predicted(manifest: dict[str, Any]) -> dict[str, Any]:
        """What a run that holds predictions says about them."""
        return {
            "n_predictions": len(manifest.get("samples") or []),
            "split": manifest.get("split"),
            "dataset": manifest.get("dataset"),
            **(manifest.get("metrics") or {}),
        }

    def run_inference(self, node: dict[str, Any]) -> None:
        step = node["id"]
        model = self._model_in(node)
        if model is None:
            return
        out = (self._work(step) / "predictions").resolve()
        manifest = self._infer(node, model, out)
        if manifest is None:
            return
        # the predictions are held by a run of their own: viewers read them from it, and an
        # evaluation attaches its results to it
        provenance = manifest.get("provenance") or {}
        name = str((provenance.get("model") or {}).get("name") or "")
        run_id = self._hold(step, ["inference", *([name] if name else [])], provenance)
        if run_id:
            from frontier._server import ServerClient

            done = _upload(
                ServerClient(str(os.environ["FRONTIER_SERVER"])), run_id, prediction_files(out)
            )
            say(f"attached {len(done)} files to run {run_id}", step, files=len(done), run=run_id)
            self._finish(run_id, self._predicted(manifest))
        output = {
            "port": "predictions",
            "kind": "predictions",
            "id": run_id,
            "run": run_id,
            "name": str(out),
        }
        emit("step.output", step, data={"output": output})
        self.succeeded(step, [output])

    # -- evaluation ----------------------------------------------------------------------------
    def _trained(
        self,
        node: dict[str, Any],
        run: dict[str, Any] | None = None,
        dataset: str | None = None,
    ) -> tuple[dict[str, Any], Path, str] | None:
        """The run a step scores: its output, the checkpoint and the dataset it trained on
        (or `dataset`). `run` is the training run wired into the step unless given; its
        `name` is the training directory, or a checkpoint file in it."""
        step = node["id"]
        if run is None:
            run = next((o for o in self._input_outputs(node) if o.get("kind") == "run"), None)
        if run is None or not run.get("name"):
            self.failed(step, "No training run reaches this step. Wire a training run into it.")
            return None
        given = Path(str(run["name"]))
        train_dir = given.parent if given.is_file() else given
        if dataset is None:
            try:
                config = json.loads((train_dir / "config.json").read_text(encoding="utf-8"))
                dataset = str(config["data"]["path"])
            except (OSError, ValueError, KeyError, TypeError):
                self.failed(step, f"The training run left no config in {train_dir}. Train again.")
                return None
        last = str(node["params"].get("checkpoint") or "best") == "last"
        checkpoint = given if given.is_file() else train_dir / ("last.pt" if last else "best.pt")
        if not checkpoint.is_file():
            self.failed(step, f"The training run left no checkpoint in {train_dir}. Train again.")
            return None
        return run, checkpoint, dataset

    def _run_env(self, run: dict[str, Any]) -> dict[str, str]:
        """Results go to the training run when there is a server to take them."""
        return self._attach(str(run.get("id") or ""))

    def _calibration(
        self, step: str, checkpoint: Path, dataset: str, predictions: Path, out: Path
    ) -> list[tuple[list[str], dict[str, str]]]:
        """Uncertainty of predictions: the checkpoint predicts the validation split, which
        calibrates the intervals."""
        calibration = self._work(step) / "calibration"
        predict = ["frontier_train", "predict", "--checkpoint", str(checkpoint)]
        predict += ["--dataset", dataset, "--split", "val", "--out", str(calibration)]
        uq = ["frontier_eval", "uq", "--predictions", str(predictions), "--dataset", dataset]
        return [(predict, {}), (uq + ["--calibration", str(calibration), "--out", str(out)], {})]

    def _evaluate(
        self,
        node: dict[str, Any],
        out: Path,
        uq: bool,
        run: dict[str, Any] | None = None,
        dataset: str | None = None,
    ) -> dict[str, Any] | None:
        """Predict with the checkpoint of a training run and score it."""
        step, p = node["id"], node["params"]
        trained = self._trained(node, run, dataset)
        if trained is None:
            return None
        run, checkpoint, dataset = trained
        split = str(p.get("split") or "test")
        evaluate = ["frontier_eval", "evaluate", "--checkpoint", str(checkpoint)]
        evaluate += ["--dataset", dataset, "--split", split, "--out", str(out)]
        for name in str(p.get("benchmarks") or "").split(","):
            if name.strip():
                evaluate += ["--benchmark", name.strip()]
        commands: list[tuple[list[str], dict[str, str]]] = [(evaluate, {})]
        if uq and not _has_split(dataset, "val"):
            say("no uncertainty: the dataset has no validation split", step)
        elif uq:
            commands += self._calibration(step, checkpoint, dataset, out / "predictions", out)
        if self._run_env(run):
            commands.append((["frontier_data.board_run", "upload", str(out)], self._run_env(run)))
        return run if self._tools(step, "The evaluation", commands) else None

    def _benchmarks(self, node: dict[str, Any], dataset: str) -> list[str] | None:
        """The benchmarks a step runs: those its parameters name, else the ones registered
        for the dataset. The step fails when there is none."""
        step = node["id"]
        names = [n.strip() for n in str(node["params"].get("benchmarks") or "").split(",")]
        names = [n for n in names if n]
        if names:
            return names
        if not (Path(dataset) / "manifest.json").is_file():
            self.failed(step, f"No dataset at {dataset}. Wire a dataset into this step.")
            return None
        # asked of the tool, as every other step does: this package does not import it
        listed = subprocess.run(
            [sys.executable, "-m", "frontier_eval", "benchmarks", "list"]
            + ["--dataset", dataset, "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            names = [str(spec["name"]) for spec in json.loads(listed.stdout or "[]")]
        except (ValueError, KeyError, TypeError):
            names = []
        if listed.returncode != 0:
            self.failed(step, one_line(listed.stderr or "The benchmarks could not be listed."))
            return None
        if not names:
            self.failed(step, "No benchmark is registered for this dataset. Name one in the node.")
            return None
        return names

    def _score(
        self, node: dict[str, Any], predictions: Path, out: Path, uq: bool, run_id: str
    ) -> bool:
        """Score predictions that exist already, one benchmark after the other, and attach
        the results to the run that holds them. Nothing is predicted again; uncertainty needs
        the checkpoint the predictions were made with and a validation split."""
        step, p = node["id"], node["params"]
        try:
            manifest = json.loads((predictions / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.failed(step, f"No predictions in {predictions}. Run the inference step again.")
            return False
        dataset = self._dataset_of(node) or str(manifest.get("dataset") or "")
        if not dataset:
            self.failed(
                step,
                "The predictions do not name their dataset. Wire a dataset into this step.",
            )
            return False
        names = self._benchmarks(node, dataset)
        if names is None:
            return False
        split = str(p.get("split") or manifest.get("split") or "test")
        out.mkdir(parents=True, exist_ok=True)
        commands: list[tuple[list[str], dict[str, str]]] = []
        for name in names:
            run = ["frontier_eval", "run", "--benchmark", name, "--predictions", str(predictions)]
            commands.append((run + ["--dataset", dataset, "--split", split, "--out", str(out)], {}))
        if uq:
            model = (manifest.get("provenance") or {}).get("model") or {}
            checkpoint = Path(str(model.get("path") or manifest.get("checkpoint") or ""))
            has_val = _has_split(dataset, "val")
            if model.get("kind") == "package" or not checkpoint.is_file():
                say("no uncertainty: it needs the checkpoint the predictions came from", step)
            elif not has_val:
                say("no uncertainty: the dataset has no validation split", step)
            else:
                commands += self._calibration(step, checkpoint, dataset, predictions, out)
        if self._attach(run_id):
            commands.append((["frontier_data.board_run", "upload", str(out)], self._attach(run_id)))
        return self._tools(step, "The evaluation", commands)

    def _evaluation(
        self, node: dict[str, Any], out: Path, uq: bool
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Score what reaches a step, in this order: predictions (of an inference step), a
        training run, the model the parameters name (`model_source` `run` or `package` with
        `model_dir`). Returns the run the results are attached to ('' when none) and the
        training run when one is known; None when the step failed."""
        step, p = node["id"], node["params"]
        inputs = self._input_outputs(node)
        predictions = next((o for o in inputs if o.get("kind") == "predictions"), None)
        trained = next((o for o in inputs if o.get("kind") == "run"), None)
        if predictions is not None and predictions.get("name"):
            run_id = str(predictions.get("run") or predictions.get("id") or "")
            ok = self._score(node, Path(str(predictions["name"])), out, uq, run_id)
            return (run_id, trained) if ok else None
        if trained is not None:
            run = self._evaluate(node, out, uq)
            return (str(run.get("id") or ""), run) if run is not None else None
        source, model = str(p.get("model_source") or ""), str(p.get("model_dir") or "")
        if source == "run" and model:
            known = str(p.get("model_run") or "")
            holder = known or self._hold(step, ["evaluation"], {"model": model, "step": step})
            run = {"port": "model", "kind": "run", "id": holder, "name": model}
            if self._evaluate(node, out, uq, run, self._dataset_of(node)) is None:
                return None
            if not known:
                self._finish(holder, {"model": model})
            return holder, (run if known else None)
        if source == "package" and model:
            manifest = self._infer(node, model, (out / "predictions").resolve())
            if manifest is None:
                return None
            provenance = manifest.get("provenance") or {}
            holder = self._hold(step, ["inference", "evaluation"], provenance)
            ok = self._score(node, (out / "predictions").resolve(), out, uq, holder)
            self._finish(holder, self._predicted(manifest))
            return (holder, None) if ok else None
        self.failed(
            step,
            "Nothing to evaluate reaches this step. Wire an inference step or a training run "
            "into it, or choose a model.",
        )
        return None

    def run_evaluate(self, node: dict[str, Any]) -> None:
        step = node["id"]
        out = self._work(step)
        done = self._evaluation(node, out, bool(node["params"].get("include_uq", True)))
        if done is None:
            return
        run_id, trained = done
        outputs = [{"port": "report", "kind": "artifact", "run": run_id, "name": str(out)}]
        if trained is not None:
            outputs.append({**trained, "port": "model"})
        self.succeeded(step, outputs)

    def run_report(self, node: dict[str, Any]) -> None:
        step = node["id"]
        scored = next((o for o in self._input_outputs(node) if o.get("kind") == "artifact"), None)
        trained = next((o for o in self._input_outputs(node) if o.get("kind") == "run"), None)
        if scored is not None and scored.get("name"):
            evald = Path(str(scored["name"]))
            # an evaluation of predictions has no training run: its results, and the report,
            # are on the run that holds the predictions
            run_id = str((trained or {}).get("id") or scored.get("run") or "")
        else:
            # fed by a training run or by predictions directly: score first (no uncertainty)
            evald = self._work(step)
            done = self._evaluation(node, evald, False)
            if done is None:
                return
            run_id, trained = done
        title = f"{self.info.get('board') or 'Board'}: credibility report"
        commands: list[tuple[list[str], dict[str, str]]] = [
            (
                ["frontier_eval", "report", "--eval", str(evald), "--out", str(evald)]
                + ["--title", title],
                {},
            )
        ]
        if self._attach(run_id):
            commands.append(
                (["frontier_data.board_run", "upload", str(evald)], self._attach(run_id))
            )
        if not self._tools(step, "The report", commands):
            return
        report = {"port": "report", "kind": "report", "run": run_id, "name": "report.html"}
        self.succeeded(step, [report, {**trained, "port": "model"}] if trained else [report])

    # -- input, adapt, optimise ------------------------------------------------------------------
    def _model_file(self, node: dict[str, Any]) -> str | None:
        """The model a step uses for its contract: the wired training run's directory, or
        the file the server found for the model its parameters name (`model_dir`)."""
        p = node["params"]
        if str(p.get("model_source") or "step") == "step":
            run = next((o for o in self._input_outputs(node) if o.get("kind") == "run"), None)
            return str(run["name"]) if run and run.get("name") else None
        return str(p["model_dir"]) if p.get("model_dir") else None

    def _contract_file(self, step: str, model: str) -> Path | None:
        """The model's contract, written for the step's tools; None when the model has none."""
        work = self._work(step)
        work.mkdir(parents=True, exist_ok=True)
        path = work / "contract.json"
        got = subprocess.run(
            [sys.executable, "-m", "frontier_train", "contract", model],
            capture_output=True,
            text=True,
            check=False,
        )
        if got.returncode != 0:
            say(one_line(got.stderr or got.stdout), step, "warn")
            return None
        path.write_text(got.stdout, encoding="utf-8")
        return path

    def run_input(self, node: dict[str, Any]) -> None:
        """Typed parameters and geometry become an input dataset (`frontier_data inputs`)."""
        step, p = node["id"], node["params"]
        out = (self._work(step) / "input").resolve()
        argv = [
            "frontier_data",
            "inputs",
            "--out",
            str(out),
            "--name",
            str(p.get("name") or "input"),
        ]
        fields = p.get("fields")
        if isinstance(fields, dict | list):
            argv += ["--parameters", json.dumps(fields)]
        elif str(p.get("table") or "").strip():
            argv += ["--parameters", str(p["table"]).strip()]
        elif isinstance(fields, str) and fields.strip():
            argv += ["--parameters", fields.strip()]
        source = str(p.get("geometry_source") or "shape")
        if source == "shape":
            argv += ["--geometry", str(p.get("geometry") or "ahmed")]
            if p.get("resolution"):
                argv += ["--resolution", str(int(p["resolution"]))]
        elif source == "files":
            files = [
                f.strip() for f in str(p.get("geometry_files") or "").splitlines() if f.strip()
            ]
            if not files:
                self.failed(step, "No geometry file is named. Say where the surfaces are.")
                return
            for f in files:
                argv += ["--file", f]
        else:
            dataset = self._dataset_of(node)
            if not dataset:
                self.failed(step, "No dataset reaches this step. Wire one in, or choose a shape.")
                return
            argv += ["--dataset", dataset]
            if str(p.get("ids") or "").strip():
                argv += ["--ids", str(p["ids"]).strip()]
        model = self._model_file(node)
        contract = self._contract_file(step, model) if model else None
        if contract:
            argv += ["--contract", str(contract)]
        if not self._tools(step, "The input", [(argv, {})]):
            return
        self.succeeded(step, self._dataset_out(str(out)))

    def run_adapt(self, node: dict[str, Any]) -> None:
        """Adapt the dataset wired in to the model's contract (`frontier_prep adapt`)."""
        step, p = node["id"], node["params"]
        dataset = self._dataset_in(node)
        if not dataset:
            return
        model = self._model_file(node)
        if not model:
            self.failed(step, NO_MODEL)
            return
        contract = self._contract_file(step, model)
        if contract is None:
            self.failed(step, "The model has no contract to adapt to. Train or package it again.")
            return
        work = self._work(step)
        fixes = work / "fixes.json"
        given = p.get("fixes")
        if isinstance(given, str) and given.strip():
            fixes.write_text(given, encoding="utf-8")
        elif isinstance(given, list) and given:
            fixes.write_text(json.dumps(given), encoding="utf-8")
        else:
            got = subprocess.run(
                [sys.executable, "-m", "frontier_train", "compat", model, dataset],
                capture_output=True,
                text=True,
                check=False,
            )
            if got.returncode != 0:
                self.failed(step, one_line(got.stderr or got.stdout))
                return
            fixes.write_text(got.stdout, encoding="utf-8")
            verdict = json.loads(got.stdout)
            if not verdict.get("problems"):
                say("the data fits the model as it is: nothing to adapt", step)
                self.succeeded(step, self._dataset_out(dataset))
                return
        out = (work / "adapted").resolve()
        argv = ["frontier_prep", "adapt", "--dataset", dataset, "--out", str(out)]
        argv += ["--fixes", str(fixes), "--contract", str(contract), "--json"]
        for item in re.split(r"[,\n]+", str(p.get("values") or "")):
            if "=" in item:
                argv += ["--set", item.strip()]
        if not self._tools(step, "The adapter", [(argv, {})]):
            return
        self.succeeded(step, self._dataset_out(str(out)))

    def run_optimise(self, node: dict[str, Any]) -> None:
        """Move the parameters of a shape to minimise what the model predicts
        (`frontier_train optimise`); the results are held by a run of their own."""
        step, p = node["id"], node["params"]
        model = self._model_file(node)
        if not model:
            self.failed(step, NO_MODEL)
            return
        free = [f.strip() for f in str(p.get("free") or "").split(",") if f.strip()]
        if not free:
            self.failed(step, "No parameter is free to move. Name one on the node.")
            return
        request: dict[str, Any] = {}
        for key in ("fixed", "bounds", "conditions", "checks", "start", "known_error"):
            raw = p.get(key)
            if isinstance(raw, str) and raw.strip():
                try:
                    request[key] = json.loads(raw)
                except ValueError as exc:
                    self.failed(step, f"{key} is not valid JSON ({exc}). Fix it on the node.")
                    return
            elif isinstance(raw, dict | list):
                request[key] = raw
        for key in ("budget", "starts", "seed", "resolution"):
            if p.get(key):
                request[key] = int(p[key])
        for key in ("trust_z", "tol"):
            if p.get(key):
                request[key] = float(p[key])
        if p.get("minimise") is False:
            request["minimise"] = False
        out = self._work(step).resolve()
        argv = ["frontier_train", "optimise", "--model", model, "--out", str(out)]
        argv += ["--geometry", str(p.get("geometry") or "ahmed"), "--free", ",".join(free)]
        argv += ["--method", str(p.get("method") or "auto"), "--json", json.dumps(request)]
        run_id = self._hold(
            step, ["optimise"], {"free": free, "geometry": p.get("geometry") or "ahmed"}
        )
        env = self._attach(run_id)
        if self.info.get("project"):
            env["FRONTIER_PROJECT"] = str(self.info["project"])
        if not self._tools(step, "The optimisation", [(argv, env)]):
            return
        try:
            result = json.loads((out / "optimum.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.failed(
                step, f"The optimisation left no optimum in {out}. Open the job for its log."
            )
            return
        best = result.get("best") or {}
        if run_id:
            from frontier._server import ServerClient

            client = ServerClient(str(os.environ["FRONTIER_SERVER"]))
            _upload(client, run_id, [(out / "optimum.json", "optimum.json")])
            self._finish(run_id, {"optimum": best, "n_evaluations": result.get("n_evaluations")})
        for key, value in {
            "optimise/best": best.get("value"),
            "optimise/evaluations": result.get("n_evaluations"),
        }.items():
            if value is not None:
                emit("step.metric", step, data={"key": key, "value": value})
        outputs = [
            {"port": "metrics", "kind": "run", "id": run_id, "name": ""},
            {
                "port": "optimum",
                "kind": "artifact",
                "run": run_id,
                "name": str(out / "optimum.json"),
            },
            {
                "port": "dataset",
                "kind": "dataset_version",
                "name": str(result.get("geometry_dir") or out / "optimum"),
            },
        ]
        self.succeeded(step, outputs)

    def run_boundary(self, node: dict[str, Any]) -> None:
        """A Request or Response node of a served board: nothing to run here. The
        Response passes the outputs wired into it on, so a viewer after it still works."""
        if node["kind"] == "service.out":
            self.succeeded(node["id"], self._input_outputs(node) or None)
        else:
            self.succeeded(node["id"])

    def run_tools(self) -> None:
        order = Pipeline.model_validate(
            {
                "name": "steps",
                "nodes": [
                    {
                        "id": n["id"],
                        "kind": n["kind"],
                        "inputs": [i for i in n.get("inputs", []) if i in self.by_id],
                    }
                    for n in self.nodes
                ],
            }
        ).topological_order()
        runners = {
            "import": self.run_import,
            "recipe": self.run_recipe,
            "training_run": self.run_training,
            "inference": self.run_inference,
            "evaluate": self.run_evaluate,
            "report": self.run_report,
            "service.in": self.run_boundary,
            "service.out": self.run_boundary,
            "input": self.run_input,
            "adapt": self.run_adapt,
            "optimise": self.run_optimise,
        }
        for entry in order:
            node = self.by_id[entry.id]
            if node["kind"] in REGISTRY or self.action[entry.id] != "run":
                continue
            blocked = next(
                (i for i in node.get("inputs", []) if i in self.by_id and not self.usable(i)), None
            )
            if blocked is not None:
                self.skipped(entry.id, f"{blocked} did not finish")
                continue
            self.running(entry.id)
            try:
                runners[node["kind"]](node)
            except Exception as exc:  # a step must end in a state, whatever went wrong
                self.failed(entry.id, f"{type(exc).__name__}: {exc}")

    def run(self) -> bool:
        """Run every step due. True when none failed."""
        unknown = [
            n["kind"]
            for n in self.nodes
            if n["kind"] not in REGISTRY
            and n["kind"] not in STEP_KINDS
            and n["kind"] not in BOUNDARY_KINDS
        ]
        if unknown:
            raise ValueError(f"unknown step kind {unknown[0]!r}")
        self.run_data()
        self.run_tools()
        states = list(self.state.values())
        ok = "failed" not in states
        emit(
            "run.finished",
            data={
                "state": "succeeded" if ok else "failed",
                "seconds": round(time.perf_counter() - self.t0, 3),
                "summary": {
                    "ok": states.count("succeeded"),
                    "failed": states.count("failed"),
                    "cached": sum(1 for a in self.action.values() if a == "cached"),
                    "skipped": states.count("skipped")
                    + sum(1 for a in self.action.values() if a == "skip"),
                },
            },
        )
        return ok


def _version(out_dir: str) -> dict[str, Any]:
    """`{"version": id}` when the dataset directory is versioned."""
    try:
        from frontier_data.versions import VersionStore

        store = VersionStore(out_dir)
        head = store.head if store.initialized else None
    except Exception:
        head = None
    return {"version": head} if head else {}


def _has_split(dataset: str | Path, split: str) -> bool:
    """Whether the dataset has samples in `split`."""
    try:
        manifest = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool((manifest.get("splits") or {}).get(split))


def is_board_document(doc: dict[str, Any]) -> bool:
    """A document the server sent for a board run, or one with steps beyond the data kinds."""
    return "run" in doc or any(
        n.get("kind") in STEP_KINDS or n.get("kind") in BOUNDARY_KINDS for n in doc.get("nodes", [])
    )


def _upload(client: Any, run_id: str, files: list[tuple[Path, str]]) -> list[str]:
    """Attach files to a run. A failed upload is reported and skipped; the files on disk
    stay the source of truth."""
    done: list[str] = []
    for path, name in files:
        try:
            client.upload_artifact(run_id, path, name)
            done.append(name)
        except Exception as exc:  # noqa: BLE001 - best effort, keep going
            say(f"upload of {name} failed: {exc}", level="warn", file=name)
    return done


def prediction_files(directory: Path) -> list[tuple[Path, str]]:
    """What a viewer needs of a predictions directory, under the artifact names it is read
    by: `predictions/<sample>.arrow`, then `predictions/manifest.json` and `summary.json`
    (last, so a manifest on the server never names a sample that is not there yet)."""
    import tarfile

    files = [(p, f"predictions/{p.name}") for p in sorted(directory.glob("*.arrow"))]
    for store in sorted(directory.glob("*.zarr")):
        packed = store.with_suffix(".zarr.tar")
        with tarfile.open(packed, "w") as archive:
            archive.add(store, arcname=store.name)
        files.append((packed, f"predictions/{packed.name}"))
    files += [
        (directory / n, f"predictions/{n}")
        for n in ("manifest.json", "summary.json")
        if (directory / n).is_file()
    ]
    return files


def upload_predictions(
    directory: str | Path, run_id: str | None = None, server: str | None = None
) -> list[str]:
    """Attach a predictions directory to a run (`FRONTIER_RUN_ID` on `FRONTIER_SERVER` when
    not given). Returns the artifact names that were uploaded."""
    server = server or os.environ.get("FRONTIER_SERVER")
    run_id = run_id or os.environ.get("FRONTIER_RUN_ID")
    if not server or not run_id:
        return []
    from frontier._server import ServerClient

    return _upload(ServerClient(server), run_id, prediction_files(Path(directory)))


def _upload_results(out: Path) -> list[str]:
    """Attach what an evaluation or a report step wrote to the run it scored
    (``FRONTIER_RUN_ID`` on ``FRONTIER_SERVER``): predictions, evaluation files, the report.
    A failed upload is reported and skipped; the files on disk stay the source of truth."""
    server, run_id = os.environ.get("FRONTIER_SERVER"), os.environ.get("FRONTIER_RUN_ID")
    if not server or not run_id:
        return []
    from frontier._server import ServerClient

    files = [(p, p.name) for p in sorted(out.glob("eval-*.json"))]
    names = ("report.html", "report.json", "uq.json", "uq-rollout.json", "uq-plots.json")
    files += [(out / n, n) for n in names if (out / n).is_file()]
    return _upload(ServerClient(server), run_id, [*prediction_files(out / "predictions"), *files])


def run_board(doc: dict[str, Any], workers: int = 4) -> bool:
    import frontier_data.nodes  # noqa: F401  (registers the built-in kinds)

    return BoardRun(doc, workers=workers).run()


def _helper(argv: list[str]) -> int:
    """Small jobs a step runs as a child, so their output is the step's log."""
    if argv[:1] == ["recipe"] and len(argv) == 4:
        # run a recipe and leave its result as JSON for the step to read
        from frontier_prep.recipes import Recipe, run_recipe

        result = run_recipe(Recipe.load(argv[1]), argv[2])
        Path(argv[3]).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        for s in result.steps:
            say(
                f"{s.status:8} {s.op}" + (f"  {one_line(s.error)}" if s.error else ""),
                level="info" if s.status in ("ok", "skipped") else "error",
                op=s.op,
                status=s.status,
            )
        return 0
    if argv[:1] == ["upload"] and len(argv) == 2:
        done = _upload_results(Path(argv[1]))
        say(
            f"attached {len(done)} files to run {os.environ.get('FRONTIER_RUN_ID', '')}",
            files=len(done),
        )
        return 0
    print("usage: python -m frontier_data.board_run recipe <recipe.json> <dataset> <result.json>")
    return 2


if __name__ == "__main__":
    sys.exit(_helper(sys.argv[1:]))
