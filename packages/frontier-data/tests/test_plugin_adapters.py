# SPDX-License-Identifier: Apache-2.0
import tomllib
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from frontier_data import adapters
from frontier_data.query import ArrayFieldReader
from frontier_plugins import data_adapters


@pytest.mark.parametrize("engine", ["zarr", "tensorstore", "zarrista"])
@pytest.mark.parametrize("version", [2, 3])
def test_adapter_selection_to_field_rows(tmp_path, monkeypatch, engine, version):
    import zarr

    pytest.importorskip(engine)
    values = np.arange(12 * 8 * 2, dtype="float32").reshape(12, 8, 2)
    root = zarr.open_group(str(tmp_path / "data.zarr"), mode="w", zarr_format=version)
    root.create_array("pressure", data=values, chunks=(2, 4, 2))
    manifest = tomllib.loads((Path(adapters.__file__).parent / "plugin.toml").read_text())
    entry = SimpleNamespace(name=engine, value=f"frontier_data.adapters:{engine}",
                            dist=SimpleNamespace(name="frontier-data", version=manifest["version"]))
    monkeypatch.setattr(metadata, "entry_points", lambda **_: [entry])
    monkeypatch.setattr(data_adapters, "manifest_for_distribution", lambda _: manifest)
    monkeypatch.setattr(data_adapters, "require_compatible", lambda _: None)
    monkeypatch.setattr(data_adapters, "load_entry_point", lambda _: getattr(adapters, engine))
    reader = ArrayFieldReader.from_plugin(
        "frontier-data", engine, {"store": str(tmp_path / "data.zarr"), "array_path": "pressure"},
        dimensions=("sample", "point", "channel"), units="Pa", association="node",
        buffer_bytes=4096, snapshot_id="verified-source-hash",
    )
    result = reader.read_rows([9, 1, 9, -1], selectors={"point": slice(1, 7, 2)})
    np.testing.assert_array_equal(result.values, values[[9, 1, 9, 11], 1:7:2])
    assert result.units == "Pa" and result.snapshot_id == "verified-source-hash"
    assert result.dimensions == ("sample", "point", "channel")


def test_normalizer_factory_produces_training_tensor():
    torch = pytest.importorskip("torch")
    normalize = adapters.normalize(mean=[10, 20], scale=[2, 4])
    batch = torch.tensor([[[12., 28.], [8., 16.]]])
    result = normalize(batch)
    assert torch.equal(result, torch.tensor([[[1., 2.], [-1., -1.]]]))
    model = torch.nn.Linear(2, 1)
    model(result).square().mean().backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()


@pytest.fixture
def simulation(tmp_path):
    from frontier_data.dataset import DatasetWriter

    values = np.random.default_rng(41).normal(size=(3, 16, 12, 8, 3)).astype("float32")
    writer = DatasetWriter(tmp_path / "simulation", "Customer transient flow")
    writer.write_grid_sample("case", fields={"velocity": values}, times=[0, .1, .2],
                             spacing=[.01] * 3, field_units={"velocity": "m/s"})
    writer.finalize()
    return tmp_path / "simulation", values


@pytest.mark.parametrize("engine", ["zarr", "tensorstore", "zarrista"])
def test_customer_discovers_plans_and_exports_with_installed_adapter(simulation, tmp_path, engine):
    import json

    from typer.testing import CliRunner

    from frontier_data.cli import app

    dataset, values = simulation
    runner = CliRunner()
    discovered = runner.invoke(app, ["data-adapters", "--json"])
    assert discovered.exit_code == 0, discovered.output
    row, = [r for r in json.loads(discovered.output)["adapters"]
            if r["plugin"] == "frontier-data" and r["adapter"] == engine]
    assert row["available"] and not row["runtime_verified"]
    args = ["query-field", str(dataset), "case", "velocity", "--adapter", f"frontier-data/{engine}",
            "--select", "time=1", "--select", "x=2:13:3", "--budget-mib", "1"]
    planned = runner.invoke(app, args)
    assert planned.exit_code == 0, planned.output
    plan = json.loads(planned.output)
    assert plan["engine"] == f"frontier-data/{engine}"
    assert plan["units"] == "m/s" and plan["association"] == "node"
    assert plan["source"]["adapter"]["distribution"] == "frontier-data"
    assert plan["source"]["adapter"]["version"] == metadata.version("frontier-data")
    output = tmp_path / "selected.npy"
    exported = runner.invoke(app, [*args, "--out", str(output)])
    assert exported.exit_code == 0, exported.output
    assert np.load(output).tobytes() == values[1, 2:13:3].tobytes()
    result = json.loads(exported.output)
    assert result["read"]["requests"] > 0


@pytest.mark.parametrize("options", [
    ["--adapter", "missing"], ["--adapter", "missing/reader"],
    ["--adapter", "frontier-data/normalize"],
    ["--adapter", "frontier-data/zarr", "--engine", "tensorstore"],
    ["--adapter", "frontier-data/zarr", "--profile", "missing.json"],
])
def test_invalid_customer_adapter_choice_fails_without_creating_output(
    simulation, tmp_path, options,
):
    from typer.testing import CliRunner

    from frontier_data.cli import app

    output = tmp_path / "refused.npy"
    result = CliRunner().invoke(app, ["query-field", str(simulation[0]), "case", "velocity",
                                     "--out", str(output), *options])
    assert result.exit_code != 0 and "Invalid value" in result.output
    assert not output.exists()


def test_plugin_must_match_sample_metadata_before_reading(simulation, monkeypatch):
    from arrayroute.backends import NumpyBackend
    from frontier_data.dataset import Dataset
    from frontier_data.query import FieldReader

    backend = NumpyBackend(np.zeros((1, 2), dtype="float32"))
    monkeypatch.setattr(data_adapters, "activate",
                        lambda *args, **kw: (lambda *a, **k: backend, {"plugin": "third-party"}))
    monkeypatch.setattr(backend, "read", lambda *args: pytest.fail("Read before validation"))
    with pytest.raises(ValueError, match="source metadata"):
        FieldReader(Dataset(simulation[0]).sample("case"), "velocity",
                    adapter=("third-party", "reader"))
