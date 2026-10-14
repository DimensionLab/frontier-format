# SPDX-License-Identifier: Apache-2.0
"""Recipes: ordered lists of ``(op, params)`` saved as JSON and re-runnable on any dataset
(``docs/spec/data-prep.md`` section 6). A cleaning session recorded through the CLI
(``run … --record recipe.json``) becomes a reproducible pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from frontier_prep.base import DatasetRef, as_ref
from frontier_prep.op import Actor, Context, Plan, Result, get_op, now_iso

RECIPE_VERSION = "1.0"


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str
    params: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    continue_on_fail: bool = False
    op_version: str | None = None


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = RECIPE_VERSION
    description: str = ""
    steps: list[Step] = Field(default_factory=list)
    created_by: Actor = Field(default_factory=Actor)
    created_at: str = Field(default_factory=now_iso)
    created_from: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        return cls.model_validate(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")

    def append(self, op_id: str, params: dict[str, Any], *, message: str = "") -> Step:
        op = get_op(op_id)
        step = Step(op=op_id, params=params, message=message, op_version=op.version)
        self.steps.append(step)
        return step


def record_step(
    path: str | Path,
    op_id: str,
    params: dict[str, Any],
    *,
    message: str = "",
    dataset: str = "",
    actor: Actor | None = None,
) -> Recipe:
    """Append a step to the recipe at ``path`` (created when missing)."""
    p = Path(path)
    recipe = (
        Recipe.load(p)
        if p.exists()
        else Recipe(name=p.stem, created_by=actor or Actor(), created_from={"dataset": dataset})
    )
    recipe.append(op_id, params, message=message)
    recipe.save(p)
    return recipe


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    op: str
    status: str  # ok | noop | skipped | failed | blocked
    result: Result | None = None
    plan: Plan | None = None
    error: str | None = None


class RecipeRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe: str
    dataset: str
    input_version: str | None
    output_version: str | None
    steps: list[StepResult]
    ok: bool

    @property
    def versions(self) -> list[str]:
        return [s.result.output_version for s in self.steps if s.result and s.result.output_version]


def plan_recipe(recipe: Recipe, dataset: Any) -> list[StepResult]:
    """Plan every step without running (``recipe diff``). Plans are against the current
    version, so later steps may see state the earlier ones would have changed."""
    ref = as_ref(dataset)
    out = []
    for i, step in enumerate(recipe.steps):
        op = get_op(step.op)
        try:
            plan = op.plan(ref, step.params)
            status = "noop" if plan.noop else ("blocked" if plan.blocking else "ok")
            out.append(StepResult(index=i, op=step.op, status=status, plan=plan))
        except Exception as exc:  # noqa: BLE001
            out.append(
                StepResult(
                    index=i, op=step.op, status="failed", error=f"{type(exc).__name__}: {exc}"
                )
            )
    return out


def run_recipe(
    recipe: Recipe,
    dataset: Any,
    *,
    ctx: Context | None = None,
    on_step: Any = None,
) -> RecipeRun:
    ref = as_ref(dataset)
    base = ctx or Context()
    actor = (
        base.actor
        if base.actor.kind != "human" or base.actor.name
        else Actor(kind="recipe", name=recipe.name)
    )
    results: list[StepResult] = []
    input_version = ref.head
    ok = True
    for i, step in enumerate(recipe.steps):
        op = get_op(step.op)
        if step.op_version and step.op_version != op.version:
            msg = f"{step.op}: recipe pins op version {step.op_version}, installed {op.version}"
            if not step.continue_on_fail:
                results.append(StepResult(index=i, op=step.op, status="failed", error=msg))
                ok = False
                break
        c = Context(
            seed=base.seed,
            workers=base.workers,
            actor=actor,
            message=step.message or f"recipe {recipe.name} step {i}: {step.op}",
            dry_run=base.dry_run,
            commit=base.commit,
            progress=base.progress,
        )
        try:
            ref = DatasetRef(ref.root)  # re-read the manifest after the previous step
            res = op.run(ref, step.params, ctx=c)
            status = "noop" if res.noop else "ok"
            if not res.ok:
                status = "failed"
                ok = ok and step.continue_on_fail
            results.append(StepResult(index=i, op=step.op, status=status, result=res))
        except Exception as exc:  # noqa: BLE001
            results.append(
                StepResult(
                    index=i, op=step.op, status="failed", error=f"{type(exc).__name__}: {exc}"
                )
            )
            ok = False
            if not step.continue_on_fail:
                break
        if on_step is not None:
            on_step(results[-1])
        if results[-1].status == "failed" and not step.continue_on_fail:
            break
    ref = DatasetRef(ref.root)
    return RecipeRun(
        recipe=recipe.name,
        dataset=ref.name,
        input_version=input_version,
        output_version=ref.head,
        steps=results,
        ok=ok,
    )
