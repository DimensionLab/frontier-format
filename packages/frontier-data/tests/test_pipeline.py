# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

import frontier_data.nodes  # noqa: F401  (registers built-ins)
from frontier_data.dataset import Dataset
from frontier_data.pipeline import REGISTRY, Node, Pipeline, PipelineEvent, PipelineRunner, register


def test_pipeline_json_roundtrip(tmp_path: Path) -> None:
    p = Pipeline(name="demo")
    p.add("src", "synthetic_source", n=2)
    p.add("read", "vtk_reader", ["src"])
    p.add("sink", "dataset_sink", ["read"], out_dir=str(tmp_path / "out"))
    text = p.to_json()
    back = Pipeline.from_json(text)
    assert back == p
    assert [n.id for n in back.topological_order()] == ["src", "read", "sink"]
    path = p.save(tmp_path / "pipeline.json")
    assert Pipeline.load(path).nodes[2].params["out_dir"].endswith("out")
    assert json.loads(text)["format"] == "frontier-pipeline"


def test_pipeline_validation() -> None:
    p = Pipeline(
        name="bad",
        nodes=[Node(id="a", kind="lod", inputs=["b"]), Node(id="b", kind="lod", inputs=["a"])],
    )
    with pytest.raises(ValueError, match="cycle"):
        p.topological_order()
    p2 = Pipeline(name="nokind", nodes=[Node(id="a", kind="nope")])
    with pytest.raises(ValueError, match="unknown node kind"):
        p2.validate_kinds()
    p3 = Pipeline(name="nosource", nodes=[Node(id="a", kind="lod")])
    with pytest.raises(ValueError, match="source"):
        p3.validate_kinds()


@pytest.mark.parametrize("executor,workers", [("sync", 1), ("thread", 2), ("process", 2)])
def test_synthetic_end_to_end(tmp_path: Path, executor: str, workers: int) -> None:
    out = tmp_path / "ds"
    p = Pipeline(name="synthetic-e2e")
    p.add("src", "synthetic_source", n=5, subdivisions=6)
    p.add("read", "vtk_reader", ["src"])
    p.add("lod", "lod", ["read"], levels=[0, 2])
    p.add("sink", "dataset_sink", ["lod"], out_dir=str(out), name="synthetic")
    p.add(
        "split",
        "split",
        ["sink"],
        out_dir=str(out),
        strategy="parameter_holdout",
        parameter="slant-angle-degrees",
        holdout_fraction=0.2,
    )
    p.add("quality", "quality_checks", ["split"], out_dir=str(out), html=False)
    events: list[PipelineEvent] = []
    result = PipelineRunner(workers=workers, executor=executor, on_event=events.append).run(p)
    assert result.ok, result.errors
    assert len(result.items) == 5
    assert set(result.timings) >= {"src", "read", "lod", "sink", "split", "quality"}
    assert any(e.status == "progress" and e.item_id == "box_1" for e in events)
    assert any(e.node_id == "quality" and e.status == "done" for e in events)
    ds = Dataset(out)
    assert len(ds) == 5
    assert ds.manifest.pipeline == "pipeline.json" and (out / "pipeline.json").exists()
    assert ds.manifest.split_strategy is not None
    assert ds.manifest.split_strategy["name"] == "parameter_holdout"
    assert len(ds.manifest.splits["test"]) == 1
    assert set(ds.manifest.split_families) == {"random", "parameter_holdout"}
    assert (out / "quality" / "report.json").exists()
    assert result.outputs["quality"]["n_samples"] == 5
    m = ds.meta("box_1")
    assert m.lod_levels == [0, 2] and "2" in m.lod_stats
    # every sample stores levels 0 and 2 as the contract requires
    assert all(set(s.lod_levels) >= {0, 2} for s in ds.manifest.samples)


def test_item_error_is_reported(tmp_path: Path) -> None:
    @register("boom", "transform", "sample")
    def boom(ctx, item):  # type: ignore[no-untyped-def]
        if item["id"] == "box_2":
            raise RuntimeError("kaboom")
        return item

    p = Pipeline(name="err")
    p.add("src", "synthetic_source", n=3)
    p.add("boom", "boom", ["src"])
    result = PipelineRunner(workers=1).run(p)
    assert len(result.errors) == 1 and "kaboom" in result.errors[0]["error"]
    assert len(result.items) == 2
    REGISTRY.pop("boom")


def test_registry_describes_builtins() -> None:
    kinds = {
        "hf_source",
        "local_source",
        "synthetic_source",
        "vtk_reader",
        "lod",
        "quality_checks",
        "split",
        "dataset_sink",
    }
    assert kinds <= set(REGISTRY)
    assert REGISTRY["hf_source"].role == "source"
    assert REGISTRY["split"].scope == "dataset"
    assert REGISTRY["dataset_sink"].finalize is not None
    assert "params" in REGISTRY["lod"].describe()
