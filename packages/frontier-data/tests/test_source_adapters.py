# SPDX-License-Identifier: Apache-2.0
import json
import pickle
from contextlib import closing

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from arrayroute.cache_policy import TraceRecorder
from frontier_data import adapters
from frontier_data.cli import app
from frontier_data.query import ArrayFieldReader
from frontier_plugins import data_adapters

DIMS = ("case", "time", "x", "channel")


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "flow.h5"
    values = np.random.default_rng(32).normal(size=(9, 7, 32, 3)).astype("float32")
    with h5py.File(path, "w") as f:
        f.create_dataset("velocity", data=values, chunks=(1, 1, 16, 3), compression="gzip")
        f["velocity"].attrs["units"] = "m/s"
    return path, values


def open_source(path, **kwargs):
    options = dict(dimensions=DIMS, units="m/s", association="node", buffer_bytes=65536)
    options.update(kwargs)
    return ArrayFieldReader.from_plugin(
        "frontier-data", "hdf5", {"store": str(path), "array_path": "velocity"}, **options,
    )


def test_hdf5_named_rows_prefetch_trace_and_owned_lifetime(source):
    path, values = source
    with open_source(path) as reader:
        backend = reader.reader.backend
        assert reader.adapter_info["adapter"] == "hdf5"
        traced = reader.with_access_trace(TraceRecorder(), identity="flow")
        batch = traced.read_rows([7, 1, 7, -1], selectors={"time": 3, "x": slice(2, 29, 3)})
        assert batch.values.tobytes() == values[[7, 1, 7, 8], 3, 2:29:3].tobytes()
        assert batch.units == "m/s" and batch.dimensions == ("case", "x", "channel")
        assert traced.adapter_info == reader.adapter_info
        traced.close()
        assert reader.read({"case": 0}).values.tobytes() == values[0].tobytes()
        with reader.prefetch({"case": 2}) as tiles:
            tile, result = next(tiles)
            assert result.values.tobytes() == values[2][tile.destination].tobytes()
        with pytest.raises(TypeError, match="worker"):
            pickle.dumps(backend)
    reader.close()
    for view in [reader, traced]:
        with pytest.raises(RuntimeError, match="closed"):
            view.read({"case": 0})
    with pytest.raises(RuntimeError, match="closed"):
        backend.read((0, slice(None), slice(None), slice(None)))
    assert batch.values.tobytes() == values[[7, 1, 7, 8], 3, 2:29:3].tobytes()


@pytest.mark.parametrize("kwargs", [{"dimensions": ("time",)}, {"units": ""},
                                    {"buffer_bytes": 0}])
def test_invalid_metadata_closes_factory_native_handle(source, monkeypatch, kwargs):
    original = adapters.hdf5
    opened = []

    def factory(*args, **kw):
        backend = original(*args, **kw)
        opened.append(backend)
        return backend

    monkeypatch.setattr(data_adapters, "activate", lambda *a, **k: (factory, {}))
    with pytest.raises(ValueError):
        open_source(source[0], **kwargs)
    assert len(opened) == 1
    with pytest.raises(RuntimeError, match="closed"):
        opened[0].read((0, 0, slice(None), slice(None)))


def test_exception_and_cancel_close_the_native_file(source):
    with pytest.raises(InterruptedError), open_source(source[0], buffer_bytes=2048) as reader:
        backend = reader.reader.backend
        with closing(reader.stream({}, cancelled=lambda: True)) as stream:
            next(stream)
    with pytest.raises(RuntimeError, match="closed"):
        backend.read((0, 0, slice(None), slice(None)))


def test_source_cli_plans_and_exports_without_changing_original(source, tmp_path):
    import hashlib

    path, values = source
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    out = tmp_path / "selected.npy"
    args = ["query-array", str(path), "velocity", "--adapter", "frontier-data/hdf5",
            "--units", "m/s", "--association", "node", "--select", "case=3",
            "--select", "time=1:7:2", "--select", "x=4:28:3"]
    for dim in DIMS:
        args += ["--dimension", dim]
    runner = CliRunner()
    planned = runner.invoke(app, args)
    assert planned.exit_code == 0, planned.output
    metadata = json.loads(planned.output)
    assert metadata["units"] == "m/s" and metadata["selected_dimensions"] == list(DIMS[1:])
    assert metadata["source"]["adapter"]["adapter"] == "hdf5"
    exported = runner.invoke(app, [*args, "--out", str(out)])
    assert exported.exit_code == 0, exported.output
    assert np.load(out).tobytes() == values[3, 1:7:2, 4:28:3].tobytes()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert json.loads(exported.output)["read"]["output_bytes"] == np.load(out).nbytes


def test_hdf5_nonembedded_sources_fail_without_retaining_file(source):
    path, _ = source
    with h5py.File(path, "a") as f:
        f["alias"] = h5py.SoftLink("/velocity")
    before = h5py.h5f.get_obj_count()
    with pytest.raises(ValueError, match="embedded"):
        adapters.hdf5({"store": str(path), "array_path": "alias"})
    assert h5py.h5f.get_obj_count() == before


def test_source_batches_match_direct_model_updates_and_held_out_predictions(source):
    torch = pytest.importorskip("torch")
    path, values = source
    batches = ([1, 7, 1], [2, 4, 0], [5, 2, 7])

    def train(read):
        torch.manual_seed(321)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        losses = []
        for indices in batches:
            x = torch.from_numpy(read(indices))
            target = x.sum(dim=-1, keepdim=True)
            loss = (model(x) - target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return model, losses

    direct, direct_losses = train(lambda indices: values[indices])
    with open_source(path) as reader:
        routed, routed_losses = train(lambda indices: reader.read_rows(indices).values)
        held_out = torch.from_numpy(reader.read_rows([3, 6, 8]).values)
    assert direct_losses == routed_losses
    for key, tensor in direct.state_dict().items():
        assert torch.equal(tensor, routed.state_dict()[key])
    assert torch.equal(direct(held_out), routed(held_out))


def test_spawned_readers_cover_partition_once_with_process_local_handles(source):
    import hashlib
    import subprocess
    import sys

    script = '''
import hashlib,json,os,sys
from frontier_data.query import ArrayFieldReader
rank = int(sys.argv[2])
indices = list(range(rank, 9, 2))
with ArrayFieldReader.from_plugin(
    "frontier-data", "hdf5", {"store": sys.argv[1], "array_path": "velocity"},
    dimensions=("case", "time", "x", "channel"), units="m/s", association="node",
    buffer_bytes=65536,
) as reader:
    result = reader.read_rows(indices)
    print(json.dumps({"pid": os.getpid(), "indices": indices, "units": result.units,
                      "sha256": hashlib.sha256(result.values.tobytes()).hexdigest()}))
'''
    workers = []
    try:
        for rank in range(2):
            workers.append(subprocess.Popen(
                [sys.executable, "-c", script, str(source[0]), str(rank)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        receipts = []
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=30)
            assert worker.returncode == 0, stderr
            receipts.append(json.loads(stdout))
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=10)
    assert len({r["pid"] for r in receipts}) == 2
    assert sorted(i for r in receipts for i in r["indices"]) == list(range(9))
    for receipt in receipts:
        expected = source[1][receipt["indices"]]
        assert receipt["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
        assert receipt["units"] == "m/s"
