# SPDX-License-Identifier: Apache-2.0
"""A small Source -> Transform -> Sink node pipeline with JSON (de)serialisation.

Model
-----
* A :class:`Pipeline` is a list of :class:`Node` (``id``, ``kind``, ``params``, ``inputs``) forming
  a DAG. The JSON form is what the desktop app renders on its canvas.
* Node kinds are registered in :data:`REGISTRY` with a :class:`NodeSpec`:
  ``role`` (source|transform|sink) and ``scope``:
  - ``sample`` nodes run once per item, in a worker (process pool by default). Their function
    receives ``(ctx, item)`` and returns the updated item dict; sources receive ``(ctx)`` and
    return the list of items.
  - ``dataset`` nodes run once in the main process after every item finished, with
    ``(ctx, results)`` where results are the (JSON-able) item outputs.
  A sample-scoped sink may define ``finalize(ctx, results)`` that runs in the dataset phase.
* Progress is reported through ``on_event(PipelineEvent)``.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

PIPELINE_FORMAT = "frontier-pipeline"
PIPELINE_VERSION = 1


class Node(BaseModel):
    id: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    label: str | None = None
    position: dict[str, float] | None = None  # optional canvas hint {x, y}


class Pipeline(BaseModel):
    format: str = PIPELINE_FORMAT
    version: int = PIPELINE_VERSION
    name: str
    description: str | None = None
    nodes: list[Node] = Field(default_factory=list)

    # -- (de)serialisation --------------------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(exclude_none=True), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Pipeline:
        return cls.model_validate(json.loads(text))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: str | Path) -> Pipeline:
        return cls.from_json(Path(path).read_text())

    # -- graph helpers ------------------------------------------------------------------
    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def add(self, node_id: str, kind: str, inputs: Iterable[str] = (), **params: Any) -> Node:
        n = Node(id=node_id, kind=kind, params=params, inputs=list(inputs))
        self.nodes.append(n)
        return n

    def topological_order(self) -> list[Node]:
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate node ids")
        for n in self.nodes:
            for i in n.inputs:
                if i not in ids:
                    raise ValueError(f"node {n.id!r} references unknown input {i!r}")
        done: list[Node] = []
        seen: set[str] = set()
        temp: set[str] = set()

        def visit(n: Node) -> None:
            if n.id in seen:
                return
            if n.id in temp:
                raise ValueError(f"cycle through node {n.id!r}")
            temp.add(n.id)
            for i in n.inputs:
                visit(self.node(i))
            temp.discard(n.id)
            seen.add(n.id)
            done.append(n)

        for n in self.nodes:
            visit(n)
        return done

    def validate_kinds(self) -> None:
        for n in self.nodes:
            if n.kind not in REGISTRY:
                raise ValueError(f"unknown node kind {n.kind!r} (node {n.id!r})")
        roles = [REGISTRY[n.kind].role for n in self.nodes]
        if roles.count("source") != 1:
            raise ValueError("a pipeline needs exactly one source node")


# -- registry --------------------------------------------------------------------------------

SampleFn = Callable[["NodeContext", dict[str, Any]], dict[str, Any]]
SourceFn = Callable[["NodeContext"], list[dict[str, Any]]]
DatasetFn = Callable[["NodeContext", list[dict[str, Any]]], Any]


@dataclass
class NodeSpec:
    kind: str
    role: str  # source | transform | sink
    scope: str  # sample | dataset
    fn: Callable[..., Any]
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)  # name -> default (documentation)
    finalize: Callable[[NodeContext, list[dict[str, Any]]], Any] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "role": self.role,
            "scope": self.scope,
            "description": self.description,
            "params": self.params,
        }


REGISTRY: dict[str, NodeSpec] = {}


def register(
    kind: str,
    role: str,
    scope: str,
    *,
    description: str = "",
    params: dict[str, Any] | None = None,
    finalize: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[kind] = NodeSpec(kind, role, scope, fn, description, params or {}, finalize)
        return fn

    return deco


# -- execution -------------------------------------------------------------------------------


class PipelineEvent(BaseModel):
    node_id: str
    status: str  # start | progress | done | error
    item_id: str | None = None
    message: str = ""
    done: int = 0
    total: int = 0
    elapsed_s: float = 0.0


EventFn = Callable[[PipelineEvent], None]


@dataclass
class NodeContext:
    node: Node
    pipeline: Pipeline
    context: dict[str, Any]
    on_event: EventFn | None = None

    @property
    def params(self) -> dict[str, Any]:
        return self.node.params

    def param(self, name: str, default: Any = None) -> Any:
        if name in self.node.params:
            return self.node.params[name]
        if name in self.context:
            return self.context[name]
        return default

    def emit(self, message: str, done: int = 0, total: int = 0, item_id: str | None = None) -> None:
        if self.on_event:
            self.on_event(
                PipelineEvent(
                    node_id=self.node.id,
                    status="progress",
                    item_id=item_id,
                    message=message,
                    done=done,
                    total=total,
                )
            )


@dataclass
class RunResult:
    items: list[dict[str, Any]]
    outputs: dict[str, Any]
    errors: list[dict[str, Any]]
    elapsed_s: float
    timings: dict[str, float]

    @property
    def ok(self) -> bool:
        return not self.errors


def _jsonable(value: Any) -> Any:
    """Drop heavy / non-serialisable values so item results can cross process boundaries."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            j = _jsonable(v)
            if j is not _DROP:
                out[str(k)] = j
        return out
    if isinstance(value, list | tuple):
        vals = [_jsonable(v) for v in value]
        return [v for v in vals if v is not _DROP]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump()
    return _DROP


_DROP = object()


def _run_sample_chain(
    pipeline_json: str, context: dict[str, Any], item: dict[str, Any], node_ids: list[str]
) -> dict[str, Any]:
    """Executed in a worker: run the sample-scoped nodes for one item."""
    import frontier_data.nodes  # noqa: F401  (registers built-in kinds in the worker)

    pipeline = Pipeline.from_json(pipeline_json)
    timings: dict[str, float] = {}
    item = dict(item)
    for nid in node_ids:
        node = pipeline.node(nid)
        spec = REGISTRY[node.kind]
        t0 = time.perf_counter()
        item = spec.fn(NodeContext(node, pipeline, context), item)
        timings[nid] = time.perf_counter() - t0
    out = _jsonable(item)
    out["_timings"] = timings
    return out


class PipelineRunner:
    """Run a :class:`Pipeline` with a process/thread pool for the per-sample stage."""

    def __init__(
        self,
        workers: int = 1,
        executor: str = "process",
        on_event: EventFn | None = None,
        fail_fast: bool = False,
    ) -> None:
        self.workers = max(1, workers)
        self.executor = executor
        self.on_event = on_event
        self.fail_fast = fail_fast

    def _emit(self, **kw: Any) -> None:
        if self.on_event:
            self.on_event(PipelineEvent(**kw))

    def _make_executor(self) -> Executor | None:
        if self.workers == 1 or self.executor == "sync":
            return None
        if self.executor == "thread":
            return ThreadPoolExecutor(max_workers=self.workers)
        return ProcessPoolExecutor(max_workers=self.workers)

    def run(self, pipeline: Pipeline, context: dict[str, Any] | None = None) -> RunResult:
        import frontier_data.nodes  # noqa: F401

        context = dict(context or {})
        pipeline.validate_kinds()
        order = pipeline.topological_order()
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        outputs: dict[str, Any] = {}
        errors: list[dict[str, Any]] = []

        source = next(n for n in order if REGISTRY[n.kind].role == "source")
        sample_nodes = [
            n.id for n in order if REGISTRY[n.kind].scope == "sample" and n.id != source.id
        ]

        # 1. source
        t0 = time.perf_counter()
        self._emit(node_id=source.id, status="start")
        items: list[dict[str, Any]] = REGISTRY[source.kind].fn(
            NodeContext(source, pipeline, context, self.on_event)
        )
        timings[source.id] = time.perf_counter() - t0
        outputs[source.id] = {"n_items": len(items)}
        self._emit(
            node_id=source.id,
            status="done",
            done=len(items),
            total=len(items),
            elapsed_s=timings[source.id],
        )

        # 2. per-sample chain
        pjson = pipeline.to_json()
        results: list[dict[str, Any]] = []
        for nid in sample_nodes:
            self._emit(node_id=nid, status="start", total=len(items))
        node_time: dict[str, float] = dict.fromkeys(sample_nodes, 0.0)
        ex = self._make_executor()
        try:
            if ex is None:
                futures: list[tuple[dict[str, Any], Future[dict[str, Any]] | None]] = [
                    (it, None) for it in items
                ]
            else:
                futures = [
                    (it, ex.submit(_run_sample_chain, pjson, context, it, sample_nodes))
                    for it in items
                ]
            for k, (item, fut) in enumerate(futures):
                iid = str(item.get("id", k))
                try:
                    res = (
                        fut.result()
                        if fut is not None
                        else _run_sample_chain(pjson, context, item, sample_nodes)
                    )
                except Exception as exc:  # worker failure for one sample
                    err = {
                        "item_id": iid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                    errors.append(err)
                    log.error("item %s failed: %s", iid, exc)
                    for nid in sample_nodes:
                        self._emit(
                            node_id=nid,
                            status="error",
                            item_id=iid,
                            message=str(exc),
                            done=k + 1,
                            total=len(items),
                        )
                    if self.fail_fast:
                        raise
                    continue
                results.append(res)
                for nid, dt in res.get("_timings", {}).items():
                    node_time[nid] = node_time.get(nid, 0.0) + dt
                for nid in sample_nodes:
                    self._emit(
                        node_id=nid,
                        status="progress",
                        item_id=iid,
                        done=k + 1,
                        total=len(items),
                        elapsed_s=res.get("_timings", {}).get(nid, 0.0),
                    )
        finally:
            if ex is not None:
                ex.shutdown(wait=True)
        for nid in sample_nodes:
            timings[nid] = node_time.get(nid, 0.0)
            self._emit(
                node_id=nid,
                status="done",
                done=len(results),
                total=len(items),
                elapsed_s=timings[nid],
            )

        # 3. dataset phase (finalizers of sample sinks + dataset-scoped nodes), in topological order
        for n in order:
            spec = REGISTRY[n.kind]
            fn: Callable[..., Any] | None = None
            if spec.scope == "dataset":
                fn = spec.fn
            elif spec.finalize is not None:
                fn = spec.finalize
            if fn is None:
                continue
            t0 = time.perf_counter()
            self._emit(node_id=n.id, status="start")
            try:
                outputs[n.id] = _jsonable(
                    fn(NodeContext(n, pipeline, context, self.on_event), results)
                )
            except Exception as exc:
                errors.append(
                    {
                        "item_id": None,
                        "node_id": n.id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                log.error("node %s failed: %s", n.id, exc)
                self._emit(node_id=n.id, status="error", message=str(exc))
                if self.fail_fast:
                    raise
                continue
            dt = time.perf_counter() - t0
            timings[n.id] = timings.get(n.id, 0.0) + dt
            self._emit(node_id=n.id, status="done", elapsed_s=dt)

        return RunResult(
            items=results,
            outputs=outputs,
            errors=errors,
            elapsed_s=time.perf_counter() - t_start,
            timings=timings,
        )


def run_pipeline(pipeline: Pipeline, context: dict[str, Any] | None = None, **kw: Any) -> RunResult:
    return PipelineRunner(**kw).run(pipeline, context)
