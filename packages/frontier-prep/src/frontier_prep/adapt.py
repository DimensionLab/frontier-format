# SPDX-License-Identifier: Apache-2.0
"""Adapt data to a model's contract: the fixes a compatibility verdict names, applied as a
recipe of ordinary preparation ops, so the result is a dataset version with the provenance
every other op leaves (``docs/spec/model-package.md``, "Model contract").

The verdict comes from ``python -m frontier_train compat`` (this package does not judge
compatibility itself); ``recipe_for`` turns its problems into recipe steps and ``adapt``
copies the dataset and runs that recipe on the copy. The fixes:

* ``resample`` (``resolution_out_of_range``): a level of detail inside the model's range,
  op ``prep.mesh.decimate`` with level 1 at twice the wanted points as faces (level 0 is
  never changed);
* ``set_value`` (``missing_parameter``): a per-sample parameter the model needs and the data
  does not give, from a constant; op ``prep.params.set``;
* ``convert`` (``unit_mismatch`` of lengths): ``prep.units.convert`` with ``geometry``;
* ``rename`` (a field under another name): ``prep.fields.rename``.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from pydantic import Field

from frontier_prep.base import BaseOp, DatasetRef, SampleOutcome
from frontier_prep.io import SampleEditor
from frontier_prep.op import Change, Params, register
from frontier_prep.recipes import Recipe, RecipeRun, run_recipe

ADAPT_RECIPE = "adapt-to-model"


# -- ops ------------------------------------------------------------------------------------


class SetParamsParams(Params):
    values: dict[str, float] = Field(
        description="parameter -> value, set on every sample", min_length=1
    )


@register
class SetParameters(BaseOp):
    id = "prep.params.set"
    version = "1.0.0"
    category = "dataset"
    summary = "Set a per-sample parameter (a condition the model takes) on every sample."
    description = (
        "Writes `geometry_params[name]` of each sample's meta and lists the name in the "
        "manifest, so a model that conditions on it finds it."
    )
    when_to_use = "A model needs a parameter the data does not record, the same for every case."
    changes_manifest = True
    Params = SetParamsParams

    def plan_sample(self, ed: SampleEditor, params: SetParamsParams) -> list[Change]:
        before = {k: ed.meta.geometry_params.get(k) for k in params.values}
        if all(before[k] == v for k, v in params.values.items()):
            return []
        return [
            Change(
                target=f"samples/{ed.id}/geometry_params",
                action="modify",
                before={k: v for k, v in before.items() if v is not None},
                after=dict(params.values),
            )
        ]

    def apply_sample(
        self, ed: SampleEditor, params: SetParamsParams, *, seed: int
    ) -> SampleOutcome:
        planned = self.plan_sample(ed, params)
        if not planned:
            return SampleOutcome(sample_id=ed.id, skipped=True)
        ed.meta.geometry_params.update(params.values)
        ed.save_meta()
        return SampleOutcome(sample_id=ed.id, changes=planned)

    def apply_manifest(
        self, ref: DatasetRef, manifest: Any, params: SetParamsParams, outcomes: list[SampleOutcome]
    ) -> tuple[list[Change], dict[str, Any]]:
        for entry in manifest.samples:
            entry.geometry_params.update(params.values)
        added = [n for n in params.values if n not in manifest.geometry_params]
        manifest.geometry_params.extend(added)
        changes = (
            [Change(target="manifest", action="modify", after={"geometry_params": added})]
            if added
            else []
        )
        return changes, {"set": sorted(params.values)}


# -- the recipe of a verdict ----------------------------------------------------------------


def recipe_for(
    contract: dict[str, Any],
    verdict: dict[str, Any],
    *,
    values: dict[str, float] | None = None,
) -> Recipe:
    """The recipe that applies the fixes of ``verdict``. ``values`` gives the constants of
    ``set_value`` fixes; a missing one is left as NaN for the person to fill in (``adapt``
    refuses to run it so)."""
    recipe = Recipe(
        name=ADAPT_RECIPE,
        description=f"Adapt to the model's contract ({contract.get('task', '')}).",
        created_from={
            "contract": contract.get("trained_on", {}),
            "verdict": verdict.get("verdict", ""),
        },
    )
    missing: dict[str, float] = {}
    for p in verdict.get("problems") or []:
        fix = p.get("fix")
        if fix == "resample":
            want = p.get("want") or {}
            most = want.get("max")
            if most:
                recipe.append(
                    "prep.mesh.decimate",
                    {"levels": [1], "targets": {1: int(2 * most)}},
                    message=p.get("line", ""),
                )
        elif fix == "set_value":
            name = str(p.get("name"))
            missing[name] = float(values[name]) if values and name in values else math.nan
        elif fix == "convert" and p.get("name") == "length":
            recipe.append(
                "prep.units.convert",
                {
                    "conversions": {},
                    "from_units": {"points": p.get("have")},
                    "geometry": p.get("want"),
                },
                message=p.get("line", ""),
            )
        elif fix == "rename" and p.get("have") and p.get("name"):
            recipe.append(
                "prep.fields.rename",
                {"map": {str(p["have"]): str(p["name"])}},
                message=p.get("line", ""),
            )
    if missing:
        recipe.append("prep.params.set", {"values": missing}, message="parameters the model needs")
    return recipe


def unset_values(recipe: Recipe) -> list[str]:
    """Parameters a `prep.params.set` step still needs a value for."""
    return [
        n
        for s in recipe.steps
        if s.op == "prep.params.set"
        for n, v in (s.params.get("values") or {}).items()
        if isinstance(v, float) and math.isnan(v)
    ]


def adapt(
    dataset_dir: str | Path,
    contract: dict[str, Any],
    out_dir: str | Path,
    *,
    fixes: list[dict[str, Any]],
    values: dict[str, float] | None = None,
    recipe: Recipe | None = None,
) -> tuple[Path, RecipeRun]:
    """Copy ``dataset_dir`` to ``out_dir`` and apply ``fixes`` (the problems of a verdict)
    there, as the recipe ``recipe_for`` makes of them unless ``recipe`` is given. Returns the
    adapted directory and what the recipe did; ``adapted.json`` in it records both."""
    src, out = Path(dataset_dir), Path(out_dir)
    chosen = recipe or recipe_for(contract, {"problems": fixes}, values=values)
    unset = unset_values(chosen)
    if unset:
        raise ValueError(f"the adapter needs a value for {unset[0]}: set it on the step")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, symlinks=False)
    ref = DatasetRef(out)
    if not ref.store.initialized:
        ref.store.init(message="before adapting")
    run = run_recipe(chosen, ref)
    (out / "adapted.json").write_text(
        json.dumps(
            {
                "contract": contract,
                "recipe": chosen.model_dump(mode="json"),
                "run": run.model_dump(mode="json"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return out, run


__all__ = ["ADAPT_RECIPE", "SetParameters", "adapt", "recipe_for", "unset_values"]
