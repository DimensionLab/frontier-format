# SPDX-License-Identifier: Apache-2.0
"""The operation contract (``docs/spec/data-prep.md`` section 2).

Every data-preparation capability is an :class:`Op`: typed parameters (a pydantic model, so a
JSON Schema exists for humans, UIs and agents), four verbs (``plan`` / ``preview`` / ``run`` /
``explain``), deterministic execution. Ops never mutate a dataset version in place; ``run``
writes a new version through :mod:`frontier_data.versions` and returns a :class:`Result` whose
provenance record names the op, its version, the parameters, the input and output versions
and the content hashes of what changed.

The module is import-light on purpose (pydantic and the standard library only) so the agent
harness, the server and a UI can load the types and the registry without numpy or zarr.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
import platform
import socket
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from frontier_data.dataset import Dataset

log = logging.getLogger(__name__)

#: entry-point group third-party packages use to publish ops
ENTRY_POINT_GROUP = "frontier.ops"
#: contract version carried by every Plan / Result / tool schema
CONTRACT_VERSION = "1.0"

Determinism = Literal["bitwise", "tolerance", "seeded"]
Severity = Literal["info", "warn", "fail"]
Mode = Literal["plan", "preview", "run"]
Category = Literal["fields", "units", "scalars", "mesh", "time", "dataset", "quality", "other"]


# -- parameters ------------------------------------------------------------------------------


class Params(BaseModel):
    """Base class of every op's parameter model. Unknown keys are an error (an agent that
    hallucinates a parameter must be told, not silently ignored)."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class Selection(BaseModel):
    """What a ``preview`` looks at. Bounded by construction: at most ``max_samples`` samples,
    at most ``max_steps`` time steps and ``max_points`` entities per array."""

    model_config = ConfigDict(extra="forbid")

    samples: list[str] | None = Field(
        default=None, description="Sample ids to preview; the first `max_samples` when omitted."
    )
    max_samples: int = Field(default=3, ge=1, le=64)
    max_steps: int = Field(default=8, ge=1, le=512, description="Time steps per sample.")
    max_points: int = Field(
        default=200_000, ge=1000, description="Entities per array read for statistics."
    )
    time_budget_s: float = Field(default=20.0, gt=0, description="Soft wall-clock bound.")
    renders: bool = Field(default=True, description="Produce thumbnails (SVG/PNG).")


# -- plan ------------------------------------------------------------------------------------


class Change(BaseModel):
    """One intended or performed modification. ``target`` is a dataset path:
    ``manifest``, ``samples/<id>``, ``samples/<id>/fields/<name>``,
    ``samples/<id>/cell_fields/<name>``, ``samples/<id>/points``, ``splits``, ..."""

    model_config = ConfigDict(extra="forbid")

    target: str
    action: Literal["add", "remove", "modify", "rename", "assert"]
    detail: dict[str, Any] = Field(default_factory=dict)
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


class Cost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_samples: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    est_seconds: float | None = None
    parallel: bool = True


class OpWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Severity = "warn"
    code: str
    message: str
    target: str | None = None


class Judgment(BaseModel):
    """A semantic micro-decision an op could not settle from the data alone (is ``p`` a
    pressure? which unit?). The op proposes options with a prior; an agent or a person picks
    one and passes it back as a parameter. Nothing is executed on a guess."""

    model_config = ConfigDict(extra="forbid")

    question: str
    target: str
    options: list[str]
    prior: list[float] | None = None
    param: str = Field(description="Parameter of this op that settles the question.")


class Plan(BaseModel):
    """What ``run`` would do: pure, no writes, cheap (reads metadata, at most small arrays)."""

    model_config = ConfigDict(extra="forbid")

    contract: str = CONTRACT_VERSION
    op: str
    op_version: str
    params: dict[str, Any]
    dataset: str
    input_version: str | None = None
    changes: list[Change] = Field(default_factory=list)
    cost: Cost = Field(default_factory=Cost)
    warnings: list[OpWarning] = Field(default_factory=list)
    judgments: list[Judgment] = Field(default_factory=list)
    determinism: Determinism = "bitwise"
    tolerance: dict[str, float] | None = None
    noop: bool = False
    summary: str = ""

    @property
    def blocking(self) -> list[OpWarning]:
        return [w for w in self.warnings if w.severity == "fail"]


# -- preview ---------------------------------------------------------------------------------


class FieldDiff(BaseModel):
    """Statistical difference of one array before and after an op."""

    model_config = ConfigDict(extra="forbid")

    target: str
    n_before: int | None = None
    n_after: int | None = None
    n_changed: int | None = None
    frac_changed: float | None = None
    max_abs_diff: float | None = None
    rel_l2_diff: float | None = None
    before: dict[str, float] | None = None  # min / max / mean / std / p01 / p99 / nan_frac
    after: dict[str, float] | None = None
    note: str | None = None


class Render(BaseModel):
    """A small picture: inline SVG text or a base64 PNG, never more than ``max_bytes``."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["svg", "png"]
    caption: str
    data: str
    target: str | None = None
    width: int = 320
    height: int = 240


class Preview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: str = CONTRACT_VERSION
    plan: Plan
    samples: list[str] = Field(default_factory=list)
    diffs: list[FieldDiff] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    renders: list[Render] = Field(default_factory=list)
    elapsed_s: float = 0.0
    truncated: bool = False
    notes: list[str] = Field(default_factory=list)


# -- result ----------------------------------------------------------------------------------


class Assertion(BaseModel):
    """A quality statement recorded in provenance (``quality.*`` ops produce them, ``quality.gate``
    reads them)."""

    model_config = ConfigDict(extra="forbid")

    check: str
    status: Literal["pass", "warn", "fail", "skip"]
    target: str = "dataset"
    detail: dict[str, Any] = Field(default_factory=dict)


class Timing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    started_at: str
    elapsed_s: float
    plan_s: float = 0.0
    apply_s: float = 0.0
    commit_s: float = 0.0
    workers: int = 1
    bytes_read: int = 0
    bytes_written: int = 0


class Machine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hostname: str
    platform: str
    python: str
    frontier_prep: str
    frontier_data: str
    #: which implementation of the hot kernels the run selected: ``native`` (the Rust ports
    #: in ``frontier_native.prep``) or ``python`` (the numpy reference). Both produce the
    #: same bytes; records written before the ports existed read as ``python``.
    prep_kernels: Literal["native", "python"] = "python"
    #: version of the Rust kernel crate, or of ``frontier-prep`` for the reference
    prep_kernels_version: str = ""

    @classmethod
    def local(cls) -> Machine:
        from frontier_prep import kernels

        k = kernels.describe()
        return cls(
            hostname=socket.gethostname(),
            platform=platform.platform(),
            python=platform.python_version(),
            frontier_prep=_dist_version("frontier-prep"),
            frontier_data=_dist_version("frontier-data"),
            prep_kernels=k["backend"],
            prep_kernels_version=k["version"],
        )


class Actor(BaseModel):
    """Who asked for the op: a person, an agent (model id) or a recipe."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["human", "agent", "recipe", "system"] = "human"
    name: str = ""
    session: str | None = None
    approved_by: str | None = None


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: str = CONTRACT_VERSION
    op: str
    op_version: str
    params: dict[str, Any]
    dataset: str
    input_version: str | None
    output_version: str | None
    changes: list[Change] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    warnings: list[OpWarning] = Field(default_factory=list)
    determinism: Determinism = "bitwise"
    tolerance: dict[str, float] | None = None
    content_hashes: dict[str, str] = Field(default_factory=dict)
    timing: Timing | None = None
    machine: Machine | None = None
    actor: Actor = Field(default_factory=Actor)
    provenance_id: str | None = None
    noop: bool = False
    summary: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(a.status == "fail" for a in self.assertions)


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: str = CONTRACT_VERSION
    op: str
    summary: str
    steps: list[str] = Field(default_factory=list)
    machine: dict[str, Any] = Field(default_factory=dict)


# -- context ---------------------------------------------------------------------------------


class Context(BaseModel):
    """Execution context of ``run``: explicit seed, worker count, actor, message. Ops read the
    seed from here (never from the clock) and the engine records all of it in provenance."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    seed: int = 0
    workers: int = Field(default=1, ge=1)
    actor: Actor = Field(default_factory=Actor)
    message: str = ""
    dry_run: bool = False
    commit: bool = True
    time_budget_s: float | None = None
    progress: Any = None  # callable(done, total, message) or None


# -- the op ----------------------------------------------------------------------------------


@runtime_checkable
class Op(Protocol):
    """The contract every operation implements. ``Params`` is the pydantic model of the
    parameters; ``params_schema`` is its JSON Schema."""

    id: str
    version: str
    category: str
    summary: str
    description: str
    determinism: Determinism
    Params: type[Params]

    def plan(self, dataset: Dataset, params: Params) -> Plan: ...

    def preview(self, dataset: Dataset, params: Params, *, sample: Selection) -> Preview: ...

    def run(self, dataset: Dataset, params: Params, *, ctx: Context) -> Result: ...

    def explain(self, result: Result) -> Explanation: ...


class OpInfo(BaseModel):
    """Serialisable description of an op (``ops list`` / ``ops show`` / the tool export)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    category: str
    summary: str
    description: str
    determinism: Determinism
    params_schema: dict[str, Any]
    when_to_use: str = ""
    produces_assertions: bool = False
    changes_geometry: bool = False
    changes_fields: bool = False
    changes_manifest: bool = False
    source: str = "builtin"


def op_info(op: Any) -> OpInfo:
    return OpInfo(
        id=op.id,
        version=op.version,
        category=getattr(op, "category", "other"),
        summary=op.summary,
        description=op.description,
        determinism=op.determinism,
        params_schema=op.Params.model_json_schema(),
        when_to_use=getattr(op, "when_to_use", ""),
        produces_assertions=getattr(op, "produces_assertions", False),
        changes_geometry=getattr(op, "changes_geometry", False),
        changes_fields=getattr(op, "changes_fields", False),
        changes_manifest=getattr(op, "changes_manifest", False),
        source=getattr(op, "source", "builtin"),
    )


# -- registry --------------------------------------------------------------------------------


class Registry:
    """Ops by id. First-party ops register at import; third-party ones come from the
    ``frontier.ops`` entry-point group (an entry point resolves to an op, an iterable of ops or
    a mapping id -> op)."""

    def __init__(self) -> None:
        self._ops: dict[str, Any] = {}
        self._loaded_entry_points = False

    def register(self, op: Any, *, source: str = "builtin") -> Any:
        if not isinstance(op, Op):
            raise TypeError(f"{op!r} does not implement the Op protocol")
        if op.id in self._ops and self._ops[op.id] is not op:
            prev = self._ops[op.id]
            if prev.version != op.version:
                log.warning("op %s re-registered: %s -> %s", op.id, prev.version, op.version)
        op.source = source  # type: ignore[attr-defined]
        self._ops[op.id] = op
        return op

    def load_entry_points(self) -> None:
        if self._loaded_entry_points:
            return
        self._loaded_entry_points = True
        try:
            eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception as exc:  # pragma: no cover - broken metadata
            log.warning("cannot read %s entry points: %s", ENTRY_POINT_GROUP, exc)
            return
        for ep in eps:
            try:
                from frontier_plugins import load_entry_point

                obj = load_entry_point(ep)
            except Exception as exc:
                log.warning("entry point %s failed to load: %s", ep.name, exc)
                continue
            for op in _iter_ops(obj):
                self.register(op, source=ep.name if ep.name != "builtin" else "builtin")

    def get(self, op_id: str) -> Any:
        self._ensure_builtin()
        if op_id not in self._ops:
            self.load_entry_points()
        if op_id not in self._ops:
            raise KeyError(f"unknown op {op_id!r}; known: {sorted(self._ops)}")
        return self._ops[op_id]

    def __contains__(self, op_id: str) -> bool:
        self._ensure_builtin()
        return op_id in self._ops

    def ids(self) -> list[str]:
        self._ensure_builtin()
        self.load_entry_points()
        return sorted(self._ops)

    def __iter__(self) -> Iterator[Any]:
        for op_id in self.ids():
            yield self._ops[op_id]

    def infos(self) -> list[OpInfo]:
        return [op_info(op) for op in self]

    def _ensure_builtin(self) -> None:
        if "prep.fields.rename" not in self._ops:
            try:
                importlib.import_module("frontier_prep.ops")
            except ImportError as exc:  # pragma: no cover
                log.warning("builtin ops not importable: %s", exc)


def _iter_ops(obj: Any) -> Iterator[Any]:
    if isinstance(obj, Op):
        yield obj
    elif isinstance(obj, dict):
        yield from obj.values()
    elif isinstance(obj, list | tuple | set):
        for x in obj:
            yield from _iter_ops(x)
    elif isinstance(obj, type):
        yield obj()


REGISTRY = Registry()


def register(op: Any) -> Any:
    """Decorator / function: ``@register`` on an op class or ``register(instance)``."""
    if isinstance(op, type):
        instance = op()
        REGISTRY.register(instance)
        return op
    return REGISTRY.register(op)


def get_op(op_id: str) -> Any:
    return REGISTRY.get(op_id)


# -- tool schema export ----------------------------------------------------------------------

TOOL_MODES: tuple[Mode, ...] = ("plan", "preview", "run")


def tool_name(op_id: str, mode: Mode | None = None) -> str:
    """``prep.fields.rename`` -> ``prep_fields_rename`` (``_run`` etc. with a mode)."""
    base = op_id.replace(".", "_").replace("-", "_")
    return base if mode is None else f"{base}__{mode}"


def tool_schema(op: Any, *, mode: Mode | None = None) -> dict[str, Any]:
    """One tool definition in the shape MCP and the Anthropic API use:
    ``{"name", "description", "input_schema"}`` plus ``x-frontier`` metadata.

    Without ``mode`` the tool takes ``mode: plan|preview|run`` (default ``plan``); with a mode
    there is one tool per verb. ``input_schema`` wraps the op's params under ``params`` next to
    ``dataset`` (path or id) and, for ``run``, ``seed`` / ``message`` / ``approval``."""
    info = op_info(op)
    props: dict[str, Any] = {
        "dataset": {
            "type": "string",
            "description": "Dataset directory (or server dataset id). Ops run on its HEAD version.",
        },
        "params": info.params_schema,
    }
    required = ["dataset"]
    if mode is None:
        props["mode"] = {
            "type": "string",
            "enum": list(TOOL_MODES),
            "default": "plan",
            "description": "plan: what would change; preview: bounded trial on a few samples "
            "with diffs/stats/renders; run: execute and create a new dataset version "
            "(needs approval unless policy grants autonomy for this op).",
        }
    if mode in (None, "preview"):
        props["selection"] = Selection.model_json_schema()
    if mode in (None, "run"):
        props["seed"] = {"type": "integer", "default": 0}
        props["message"] = {"type": "string", "description": "Provenance message."}
        props["approval"] = {
            "type": "string",
            "description": "Approval token or approver name; required for run.",
        }
    desc = info.summary
    if info.when_to_use:
        desc += f" When to use: {info.when_to_use}"
    desc += f" Determinism: {info.determinism}."
    return {
        "name": tool_name(op.id, mode),
        "description": desc,
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        },
        "x-frontier": {
            "contract": CONTRACT_VERSION,
            "op": info.id,
            "op_version": info.version,
            "category": info.category,
            "mode": mode,
            "determinism": info.determinism,
            "changes_geometry": info.changes_geometry,
            "changes_fields": info.changes_fields,
            "changes_manifest": info.changes_manifest,
            "produces_assertions": info.produces_assertions,
            "requires_approval": mode in (None, "run"),
            "result_schema": {
                "plan": "Plan",
                "preview": "Preview",
                "run": "Result",
            },
        },
    }


def export_tools(*, per_mode: bool = False, registry: Registry = REGISTRY) -> dict[str, Any]:
    """The whole tool surface: ``{"contract", "generated_at", "tools": [...], "types": {...}}``
    where ``types`` carries the JSON Schemas of Plan / Preview / Result / Explanation so a
    client can validate what comes back."""
    tools: list[dict[str, Any]] = []
    for op in registry:
        if per_mode:
            tools += [tool_schema(op, mode=m) for m in TOOL_MODES]
        else:
            tools.append(tool_schema(op))
    return {
        "contract": CONTRACT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "entry_point_group": ENTRY_POINT_GROUP,
        "tools": tools,
        "types": {
            "Plan": Plan.model_json_schema(),
            "Preview": Preview.model_json_schema(),
            "Result": Result.model_json_schema(),
            "Explanation": Explanation.model_json_schema(),
            "Selection": Selection.model_json_schema(),
            "Context": Context.model_json_schema(),
        },
    }


def export_tools_json(path: Path | None = None, **kw: Any) -> str:
    text = json.dumps(export_tools(**kw), indent=2, sort_keys=False)
    if path is not None:
        Path(path).write_text(text)
    return text


# -- helpers for op authors ------------------------------------------------------------------


def _dist_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+local"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0
