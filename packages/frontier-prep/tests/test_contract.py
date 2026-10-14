# SPDX-License-Identifier: Apache-2.0
"""The op contract: registry, schemas, tool export, plan purity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from frontier_prep.base import DatasetRef
from frontier_prep.op import (
    CONTRACT_VERSION,
    REGISTRY,
    Op,
    Plan,
    Preview,
    Result,
    export_tools,
    get_op,
    tool_name,
    tool_schema,
)
from frontier_prep.vocabulary import map_name

EXPECTED_OPS = {
    f"prep.{x}"
    for x in (
        "fields.rename fields.map fields.cast_location fields.components fields.nan_fix "
        "fields.outlier_detect fields.smooth fields.normalize units.convert scalars.derive "
        "mesh.watertight_check mesh.merge_seams mesh.repair mesh.orient_outward mesh.transform "
        "mesh.decimate mesh.resample mesh.part_ids time.subsample time.trim time.align "
        "time.drop_duplicate_frames time.window_average time.continuity_check dataset.dedup "
        "dataset.split dataset.coverage dataset.balance dataset.select dataset.merge "
        "dataset.summarize quality.check quality.gate"
    ).split()
}


def test_registry_has_every_op() -> None:
    ids = set(REGISTRY.ids())
    assert EXPECTED_OPS <= ids
    for op in REGISTRY:
        assert isinstance(op, Op)
        assert op.id.startswith("prep.") and op.version and op.summary
        schema = op.Params.model_json_schema()
        assert schema.get("additionalProperties") is False


def test_params_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        get_op("prep.fields.rename").Params(map={"a": "b"}, bogus=1)  # type: ignore[call-arg]


def test_tool_export_shape(tmp_path: Path) -> None:
    tools = export_tools()
    assert tools["contract"] == CONTRACT_VERSION
    names = {t["name"] for t in tools["tools"]}
    assert {tool_name(i) for i in EXPECTED_OPS} <= names
    t = tool_schema(get_op("prep.fields.rename"))
    assert set(t) == {"name", "description", "input_schema", "x-frontier"}
    props = t["input_schema"]["properties"]
    assert set(props) >= {"dataset", "params", "mode", "selection", "seed", "message", "approval"}
    assert props["mode"]["enum"] == ["plan", "preview", "run"]
    assert t["x-frontier"]["requires_approval"] is True
    per_mode = export_tools(per_mode=True)
    assert len(per_mode["tools"]) == 3 * len(tools["tools"])
    assert "prep_fields_rename__run" in {x["name"] for x in per_mode["tools"]}
    for key in ("Plan", "Preview", "Result", "Explanation", "Selection", "Context"):
        assert key in tools["types"]
    json.dumps(tools)  # serialisable


def test_plan_is_pure(case) -> None:
    root = case("naming")
    ref = DatasetRef(root)
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    plan = get_op("prep.fields.map").plan(ref, {})
    assert isinstance(plan, Plan)
    assert plan.changes and not plan.noop
    after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert before == after  # nothing written, not even a version store


def test_preview_leaves_no_trace(case) -> None:
    root = case("naming")
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    pv = get_op("prep.fields.map").preview(DatasetRef(root), {})
    assert isinstance(pv, Preview) and pv.samples == ["run_1"]
    after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert before == after


def test_result_and_explanation(case) -> None:
    root = case("naming")
    op = get_op("prep.fields.map")
    res = op.run(DatasetRef(root), {})
    assert isinstance(res, Result) and res.output_version and res.provenance_id
    assert res.output_version != res.input_version
    ex = op.explain(res)
    assert ex.op == op.id and ex.steps and "rename" in " ".join(ex.steps)
    # the provenance log holds the op record and the output note
    store = DatasetRef(root).store
    recs = store.provenance("op")
    assert any(r["op"] == "prep.fields.map" and r["id"] == res.provenance_id for r in recs)
    assert store.version("HEAD").op_record == res.provenance_id


def test_vocabulary_mapping() -> None:
    assert map_name("pMean", "m^2/s^2").canonical == "kinematic_pressure"
    assert map_name("pMean", "Pa").canonical == "pressure"
    amb = map_name("p", None)
    assert amb.canonical is None and set(amb.options) == {"pressure", "kinematic_pressure"}
    assert map_name("static(p)_coeffMean", "-").canonical == "pressure_coefficient"
    assert map_name("Pressure (Pa)", "Pa").canonical == "pressure"
    assert map_name("wallShearStressMean", "m^2/s^2").canonical == "kinematic_wall_shear_stress"
    bad = map_name("T", "Pa")
    assert bad.unit_compatible is False and bad.confidence < 0.5
