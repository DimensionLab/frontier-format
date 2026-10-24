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
    entry = SimpleNamespace(name=engine, dist=object())
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
