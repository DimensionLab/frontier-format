# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from frontier_data.dataset import Dataset, DatasetWriter
from frontier_data.query import FieldReader, parse_selectors


@pytest.mark.parametrize('engine', ['zarr-python', 'tensorstore', 'zarrista'])
@pytest.mark.parametrize('strategy', ['native', 'dense'])
def test_engineer_parallel_field_readers_share_budget_and_keep_units(tmp_path, engine, strategy):
    from concurrent.futures import ThreadPoolExecutor

    from arrayroute.budget import ReadBudget
    from arrayroute.layout import GridLayoutPolicy
    from arrayroute.session import LocalReadSession

    velocity = np.random.default_rng(2107).normal(size=(2, 24, 24, 24, 3)).astype('f4')
    pressure = velocity[..., :1] * 100
    writer = DatasetWriter(tmp_path, 'shared analysis memory')
    writer.write_grid_sample('case', fields={'velocity': velocity, 'pressure': pressure},
        times=[0, .1], spacing=[.1] * 3, field_units={'velocity': 'm/s', 'pressure': 'Pa'},
        storage_layout=GridLayoutPolicy(chunk_edge=8, shard_edge=24))
    writer.finalize()
    pool = ReadBudget(512 << 10)

    def inspect(i):
        field = 'velocity' if i % 2 else 'pressure'
        session = LocalReadSession(engine, read_budget=pool, buffer_bytes=512 << 10,
                                   read_strategy=strategy)
        reader = FieldReader(Dataset(tmp_path).sample('case'), field, session=session)
        result = reader.read({'time': i % 2, 'x': slice(None, None, 2)})
        expected = (velocity if field == 'velocity' else pressure)[i % 2, ::2]
        assert result.values.tobytes() == expected.tobytes()
        assert result.units == ('m/s' if field == 'velocity' else 'Pa')
        assert result.association == 'node'

    with ThreadPoolExecutor(8) as executor:
        list(executor.map(inspect, range(8)))
    assert pool.snapshot()['used_bytes'] == 0
    assert pool.snapshot()['peak_bytes'] <= 512 << 10
    reader = FieldReader(Dataset(tmp_path).sample('case'), 'velocity',
                         session=LocalReadSession(engine, read_budget=pool,
                                                  buffer_bytes=256 << 10))
    source = reader.stream({'time': 0})
    next(source)
    assert pool.snapshot()['used_bytes'] > 0
    source.close()
    assert pool.snapshot()['used_bytes'] == 0


def test_customer_range_policy_exports_exact_sharded_field_and_records_limits(tmp_path):
    import json

    from typer.testing import CliRunner

    from arrayroute.layout import GridLayoutPolicy
    from frontier_data.cli import app

    pytest.importorskip('zarr', minversion='3.3')
    values = np.random.default_rng(2718).normal(size=(2, 32, 32, 32, 3)).astype('f4')
    writer = DatasetWriter(tmp_path / 'data', 'bounded region reads')
    writer.write_grid_sample(
        'case', fields={'velocity': values}, times=[0, .1], spacing=[.1] * 3,
        field_units={'velocity': 'm/s'},
        storage_layout=GridLayoutPolicy(chunk_edge=8, shard_edge=32),
    )
    writer.finalize()
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'max_request_bytes': 8192, 'max_inflight_bytes': 16384}))
    out = tmp_path / 'region.npy'
    args = ['query-field', str(tmp_path / 'data'), 'case', 'velocity',
            '--select', 'time=1', '--select', 'x=1:14:2', '--select', 'y=2:29:3',
            '--select', 'z=0:8', '--range-policy', str(policy), '--out', str(out)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report['units'] == 'm/s' and report['association'] == 'node'
    assert report['source']['range_policy']['max_inflight_bytes'] == 16384
    assert np.load(out).tobytes() == values[1, 1:14:2, 2:29:3, :8].tobytes()
    refused = CliRunner().invoke(app, [*args, '--engine', 'tensorstore'])
    assert refused.exit_code != 0 and 'Zarr-Python' in refused.output
    for invalid in ['[]', '{"max_request_bytes": true}', ' ' * 65537]:
        policy.write_text(invalid)
        refused = CliRunner().invoke(app, args)
        assert refused.exit_code != 0 and 'Error' in refused.output


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
@pytest.mark.parametrize("strategy", ["native", "dense"])
def test_cli_measures_customer_regions_with_units_and_verified_reads(tmp_path, engine, strategy):
    import json

    from typer.testing import CliRunner

    from frontier_data.cli import app

    writer = DatasetWriter(tmp_path / "data", "engineer field pilot")
    values = np.arange(4 * 8**3, dtype=np.float32).reshape(4, 8, 8, 8, 1)
    writer.write_grid_sample(
        "case", fields={"pressure": values}, times=[0.0, 0.1, 0.2, 0.3],
        spacing=[0.1] * 3, field_units={"pressure": "Pa"},
    )
    writer.finalize()
    args = ["benchmark-field", str(tmp_path / "data"), "case", "pressure", "--vary", "time",
            "--count", "3", "--select", "x=1:7:2", "--workers", "1", "--workers", "2",
            "--repeats", "1", "--engine", engine, "--read-strategy", strategy]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["state"] == "completed" and report["units"] == "Pa"
    assert report["selected_indices"] == [0, 1, 3]
    assert len(report["reference_sha256"]) == 3 and len(report["trials"]) == 2
    assert report["planned_output_bytes"] == 3 * 3 * 8 * 8 * 3 * 4
    assert all(trial["verified"] for trial in report["trials"])
    refused = CliRunner().invoke(app, [*args, "--select", "time=1"])
    assert refused.exit_code != 0 and "Choose --vary" in refused.output


def test_cli_concurrency_plan_reads_metadata_only(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from arrayroute import ArrayReader
    from frontier_data.cli import app

    writer = DatasetWriter(tmp_path / "data", "reader planning")
    writer.write_grid_sample(
        "case", fields={"velocity": np.zeros((1, 128, 128, 128, 3), np.float32)},
        times=[0.0], spacing=[0.1] * 3, field_units={"velocity": "m/s"},
    )
    writer.finalize()

    def no_field_reads(*args, **kwargs):
        raise AssertionError("Planning must not load field values")

    monkeypatch.setattr(ArrayReader, "read", no_field_reads)
    monkeypatch.setattr(ArrayReader, "stream", no_field_reads)
    args = ["query-field", str(tmp_path / "data"), "case", "velocity", "--select", "time=0"]
    output = CliRunner().invoke(
        app, [*args, "--plan-workers", "8", "--total-read-budget-mib", "256"]
    )
    assert output.exit_code == 0, output.output
    report = json.loads(output.output)
    assert report["units"] == "m/s"
    assert report["concurrency"]["max_workers_without_tiling"] == 5
    options = {item["workers"]: item for item in report["concurrency"]["candidates"]}
    assert options[8]["requests_per_value_lower_bound"] == 8
    assert options[4]["materialization"] == "direct"
    for incomplete in [["--plan-workers", "8"], ["--total-read-budget-mib", "256"]]:
        refused = CliRunner().invoke(app, [*args, *incomplete])
        assert refused.exit_code != 0 and "Provide both" in refused.output


def test_remote_connector_preserves_scientific_metadata_and_comparison(tmp_path):
    import functools
    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    import zarr

    from arrayroute.remote import HttpStore, RemoteReadSession
    from arrayroute.session import LocalReadSession
    from frontier_data.query import ArrayFieldReader, compare_fields

    values = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    root = zarr.open_group(tmp_path, mode="w")
    root.create_array("pressure", data=values, chunks=(2, 4))

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        session = RemoteReadSession(HttpStore(f"http://127.0.0.1:{server.server_port}"))
        remote = ArrayFieldReader(session.open("pressure"), ("time", "cell"), "Pa", "cell")
        local = ArrayFieldReader(
            LocalReadSession().open(tmp_path, "pressure"), ("time", "cell"), "Pa", "cell"
        )
        selection = {"time": slice(1, 6, 2), "cell": 3}
        result = remote.read(selection)
        assert result.units == "Pa" and result.association == "cell"
        assert result.dimensions == ("time",)
        np.testing.assert_array_equal(result.values, values[1:6:2, 3])
        assert compare_fields(local, remote, selection, absolute_tolerance=0).passed
        with pytest.raises(ValueError, match="dimensions"):
            ArrayFieldReader(remote.reader, ("time", "time"), "Pa", "cell")
        with pytest.raises(ValueError, match="declared"):
            ArrayFieldReader(remote.reader, ("time", "cell"), "", "cell")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
def test_named_query_preserves_values_units_and_association(tmp_path, engine):
    if engine != "zarr-python":
        pytest.importorskip(engine)
    values = np.arange(4 * 9 * 8 * 7 * 3, dtype=np.float32).reshape(4, 9, 8, 7, 3)
    writer = DatasetWriter(tmp_path / "dataset", "selection fixture")
    writer.write_grid_sample(
        "case",
        fields={"velocity": values},
        times=np.arange(4) * 0.1,
        spacing=[0.2, 0.3, 0.4],
        field_units={"velocity": "m/s"},
    )
    writer.finalize()
    reader = FieldReader(Dataset(tmp_path / "dataset").sample("case"), "velocity", engine=engine)
    selected = {"time": 2, "x": slice(2, 6), "y": 4, "component": 1}
    result = reader.read(selected)
    assert result.values.tobytes() == values[2, 2:6, 4, :, 1].tobytes()
    assert result.dimensions == ("x", "z")
    assert result.units == "m/s"
    assert result.association == "node"
    assert reader.plan(selected).output_bytes == result.values.nbytes
    with pytest.raises(ValueError, match="Unknown dimensions"):
        reader.read({"typo": 3})


def test_selectors():
    assert parse_selectors(["time=2", "x=10:30:2", "y=:"]) == {
        "time": 2,
        "x": slice(10, 30, 2),
        "y": slice(None, None),
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_selectors(["x=2", "x=3"])


def test_multiple_fields_share_session_but_keep_units(tmp_path):
    from arrayroute.session import LocalReadSession

    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1)
    writer = DatasetWriter(tmp_path, "shared field reads")
    writer.write_grid_sample(
        "case",
        fields={"p": values, "u": values + 10},
        spacing=[1, 1, 1],
        field_units={"p": "Pa", "u": "m/s"},
    )
    writer.finalize()
    sample = Dataset(tmp_path).sample("case")
    session = LocalReadSession("tensorstore", max_open_arrays=2)
    pressure = FieldReader(sample, "p", session=session)
    velocity = FieldReader(sample, "u", session=session)
    assert pressure.read({}).units == "Pa"
    assert velocity.read({}).units == "m/s"
    assert len(session._readers) == 2
    with pytest.raises(ValueError, match="differs"):
        FieldReader(sample, "p", engine="zarrista", session=session)


def test_engineer_cli_previews_cost_and_exports_selected_values(tmp_path):
    import json

    from typer.testing import CliRunner

    from frontier_data.cli import app

    root = tmp_path / "data"
    values = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6, 1)
    writer = DatasetWriter(root, "engineer fixture")
    writer.write_grid_sample(
        "case",
        fields={"pressure": values},
        spacing=[1, 1, 1],
        field_units={"pressure": "Pa"},
        location="cell",
    )
    writer.finalize()
    args = ["query-field", str(root), "case", "pressure", "--select", "x=2", "--select", "y=1:4"]
    preview = CliRunner().invoke(app, args)
    assert preview.exit_code == 0, preview.output
    metadata = json.loads(preview.output)
    assert metadata["units"] == "Pa"
    assert metadata["association"] == "cell"
    assert metadata["output_shape"] == [3, 6, 1]
    assert metadata["selected_dimensions"] == ["y", "z", "component"]
    assert metadata["output_bytes"] == values[2, 1:4].nbytes
    out = tmp_path / "pressure-slice.npy"
    exported = CliRunner().invoke(app, [*args, "--out", str(out)])
    assert exported.exit_code == 0, exported.output
    assert np.load(out, allow_pickle=False).tobytes() == values[2, 1:4].tobytes()
    repeated = CliRunner().invoke(app, [*args, "--out", str(out)])
    assert repeated.exit_code != 0
    assert np.load(out, allow_pickle=False).tobytes() == values[2, 1:4].tobytes()


def test_stream_query_tiles_preserve_units_and_can_cancel(tmp_path):
    from arrayroute import ReadStats

    values = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6, 1)
    writer = DatasetWriter(tmp_path, "streamed pressure")
    writer.write_grid_sample(
        "case",
        fields={"pressure": values},
        spacing=[1, 1, 1],
        field_units={"pressure": "Pa"},
        location="cell",
    )
    writer.finalize()
    reader = FieldReader(Dataset(tmp_path).sample("case"), "pressure", buffer_bytes=600)
    actual = np.empty_like(values)
    stats = ReadStats()
    for tile, result in reader.stream({}, stats=stats):
        assert result.units == "Pa" and result.association == "cell"
        assert result.dimensions == ("x", "y", "z", "component")
        actual[tile.destination] = result.values
    assert stats.requests > 1 and stats.output_bytes == values.nbytes
    np.testing.assert_array_equal(actual, values)
    with pytest.raises(InterruptedError):
        next(reader.stream({}, cancelled=lambda: True))


def test_cli_exports_larger_than_read_budget_and_enforces_disk_limit(tmp_path):
    from typer.testing import CliRunner

    from frontier_data.cli import app

    root = tmp_path / "data"
    values = np.arange(20 * 32**3, dtype=np.float32).reshape(20, 32, 32, 32, 1)
    writer = DatasetWriter(root, "transient pressure")
    writer.write_grid_sample(
        "case",
        fields={"pressure": values},
        times=np.arange(20) * 0.1,
        spacing=[1, 1, 1],
        field_units={"pressure": "Pa"},
    )
    writer.finalize()
    output = tmp_path / "pressure.npy"
    args = ["query-field", str(root), "case", "pressure", "--budget-mib", "1", "--out", str(output)]
    refused = CliRunner().invoke(app, [*args, "--max-output-mib", "1"])
    assert refused.exit_code != 0 and not output.exists()
    assert not list(tmp_path.glob(".*.partial"))
    saved = CliRunner().invoke(app, [*args, "--max-output-mib", "4"])
    assert saved.exit_code == 0, saved.output
    assert np.load(output, allow_pickle=False).tobytes() == values.tobytes()


def test_compare_fields_streams_different_chunk_layouts_and_reports_si_units(tmp_path):
    import json

    from typer.testing import CliRunner

    from arrayroute.quality import assess_error
    from frontier_data.cli import app
    from frontier_data.query import compare_fields

    values = np.linspace(-3, 7, 20 * 4 * 5 * 6, dtype=np.float32).reshape(20, 4, 5, 6, 1)
    changed = values + np.float32(0.1)
    for name, data, chunks, units in [
        ("truth", values, 1, "Pa"),
        ("candidate", changed, 4, "Pa"),
        ("wrong-units", changed, 1, "m/s"),
    ]:
        writer = DatasetWriter(tmp_path / name, name)
        writer.write_grid_sample(
            "case",
            fields={"p": data},
            times=np.arange(20) * 0.1,
            spacing=[1, 1, 1],
            field_units={"p": units},
            steps_per_chunk=chunks,
        )
        writer.finalize()
    readers = [
        FieldReader(Dataset(tmp_path / name).sample("case"), "p", buffer_bytes=4096)
        for name in ["truth", "candidate", "wrong-units"]
    ]
    selection = {"time": slice(1, 19, 2), "x": 2}
    comparison = compare_fields(*readers[:2], selection, absolute_tolerance=0.2)
    direct = assess_error(values[1:19:2, 2], changed[1:19:2, 2], absolute_tolerance=0.2)
    assert comparison.passed
    assert comparison.finite_values == direct.finite_values
    assert comparison.rmse == pytest.approx(direct.rmse)
    with pytest.raises(ValueError, match="units"):
        compare_fields(readers[0], readers[2], {}, absolute_tolerance=0.2)
    with pytest.raises(ValueError, match="no comparable"):
        compare_fields(*readers[:2], {"time": slice(0, 0)}, absolute_tolerance=0.2)
    with pytest.raises(InterruptedError):
        compare_fields(*readers[:2], {}, absolute_tolerance=0.2, cancelled=lambda: True)
    args = ["compare-field", str(tmp_path / "truth"), str(tmp_path / "candidate"), "case", "p"]
    passed = CliRunner().invoke(app, [*args, "--tolerance", "0.2"])
    assert passed.exit_code == 0, passed.output
    assert json.loads(passed.output)["units"] == "Pa"
    failed = CliRunner().invoke(app, [*args, "--tolerance", "0.01"])
    assert failed.exit_code == 1
    assert not json.loads(failed.output)["assessment"]["passed"]


def test_snapshot_field_keeps_data_and_scientific_metadata_on_same_revision(tmp_path):
    import pickle

    ic = pytest.importorskip("icechunk")
    import zarr

    from arrayroute.snapshot import FilesystemRepository, Snapshot, SnapshotReadSession
    from frontier_data.query import SnapshotFieldReader

    location = FilesystemRepository(str(tmp_path / "snapshots"))
    repo = ic.Repository.create(ic.local_filesystem_storage(location.path))
    write = repo.writable_session("main")
    group = zarr.group(write.store)
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    field = group.create_array("pressure", data=data, chunks=(1, 3, 4))
    field.attrs.update({"dims": ["time", "x", "y"], "units": "Pa", "association": "cell"})
    old_id = write.commit("native pressure")
    session = SnapshotReadSession(Snapshot(location, old_id))
    write = repo.writable_session("main")
    field = zarr.open_group(write.store, mode="a")["pressure"]
    field[:] = data / 1000
    field.attrs["units"] = "kPa"
    write.commit("convert pressure units")
    original = SnapshotFieldReader(pickle.loads(pickle.dumps(session)), "pressure")
    result = original.read({"time": 1})
    assert result.units == "Pa" and result.association == "cell"
    assert result.snapshot_id == old_id and result.dimensions == ("x", "y")
    assert result.values.tobytes() == data[1].tobytes()
    for _, tile in original.stream({"time": 1}):
        assert tile.snapshot_id == old_id and tile.units == "Pa"
    with original.prefetch({"time": 1}) as tiles:
        for _, tile in tiles:
            assert tile.snapshot_id == old_id and tile.units == "Pa"
            assert tile.values.tobytes() == data[1].tobytes()
    latest = SnapshotFieldReader(SnapshotReadSession(Snapshot.resolve(location)), "pressure")
    assert latest.read({}).units == "kPa"
    import json

    from typer.testing import CliRunner

    from frontier_data.cli import app

    output = tmp_path / "pressure.npy"
    args = [
        "query-snapshot",
        location.path,
        "pressure",
        "--snapshot-id",
        old_id,
        "--select",
        "time=1",
        "--out",
        str(output),
    ]
    exported = CliRunner().invoke(app, args)
    assert exported.exit_code == 0, exported.output
    receipt = json.loads(exported.output)
    assert receipt["snapshot_id"] == old_id and receipt["units"] == "Pa"
    assert np.load(output).tobytes() == data[1].tobytes()
    refused = CliRunner().invoke(app, args)
    assert refused.exit_code != 0
    branch = CliRunner().invoke(
        app, ["query-snapshot", location.path, "pressure", "--branch", "main"]
    )
    assert branch.exit_code == 0, branch.output
    assert json.loads(branch.output)["units"] == "kPa"


def test_snapshot_field_requires_declared_metadata(tmp_path):
    ic = pytest.importorskip("icechunk")
    import zarr

    from arrayroute.snapshot import FilesystemRepository, Snapshot, SnapshotReadSession
    from frontier_data.query import SnapshotFieldReader

    location = FilesystemRepository(str(tmp_path / "snapshots"))
    repo = ic.Repository.create(ic.local_filesystem_storage(location.path))
    write = repo.writable_session("main")
    group = zarr.group(write.store)
    group.create_array("anonymous", data=np.zeros(10, dtype=np.float32))
    sid = write.commit("metadata missing")
    with pytest.raises(ValueError, match="dims"):
        SnapshotFieldReader(SnapshotReadSession(Snapshot(location, sid)), "anonymous")


def test_prepared_revision_is_queryable_with_its_units_and_lineage(tmp_path):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from datetime import UTC, datetime, timedelta

    ic = pytest.importorskip("icechunk")
    import zarr

    from arrayroute.preparation import Preparation
    from arrayroute.snapshot import FilesystemRepository, Snapshot, SnapshotReadSession
    from frontier_data.query import SnapshotFieldReader

    location = FilesystemRepository(str(tmp_path / "prepared"))
    repo = ic.Repository.create(ic.local_filesystem_storage(location.path))
    write = repo.writable_session("main")
    values = np.arange(3 * 9 * 7, dtype=np.float32).reshape(3, 9, 7)
    attrs = {"dims": ["time", "x", "y"], "units": "m/s", "association": "cell"}
    zarr.group(write.store).create_array("u", data=values, chunks=(1, 4, 4), attributes=attrs)
    source = Snapshot(location, write.commit("velocity source"))
    job = Preparation.create(
        source,
        branch="reverse",
        job="reverse",
        source="u",
        output="minus_u",
        chunks=(1, 4, 4),
        task_shape=(1, 8, 8),
        halo=(0, 0, 0),
        dtype="f4",
        operation="negative-v1",
        parameters={},
        attributes=attrs,
    )
    with ThreadPoolExecutor(2) as pool:
        job.run(np.negative, pool)
    published = job.publish()
    pin = job.output_pin(published)
    repo.delete_branch("reverse")
    cutoff = datetime.now(UTC) + timedelta(seconds=1)
    repo.expire_snapshots(cutoff)
    repo.garbage_collect(cutoff)
    field = SnapshotFieldReader(SnapshotReadSession(published), "minus_u", retention=pin)
    result = field.read({"time": 1, "x": slice(2, 8, 2)})
    assert result.values.tobytes() == (-values[1, 2:8:2]).tobytes()
    assert result.units == "m/s" and result.association == "cell"
    assert result.dimensions == ("x", "y") and result.snapshot_id == published.snapshot_id
    assert repo.lookup_branch("main") == source.snapshot_id
    assert field.retention_tag == pin.tag
    with pytest.raises(ValueError, match="another field revision"):
        SnapshotFieldReader(SnapshotReadSession(source), "u", retention=pin)
    from typer.testing import CliRunner

    from frontier_data.cli import app

    args = [
        "query-snapshot",
        location.path,
        "minus_u",
        "--snapshot-id",
        published.snapshot_id,
        "--retention-token",
        pin.token,
        "--retention-purpose",
        pin.purpose,
    ]
    viewed = CliRunner().invoke(app, args)
    assert viewed.exit_code == 0, viewed.output
    assert json.loads(viewed.output)["retention_tag"] == pin.tag
    from arrayroute.retention import RetentionManager

    RetentionManager(location).release(pin)
    refused = CliRunner().invoke(app, args)
    assert refused.exit_code != 0


def test_description_reads_no_values_and_normalizes_selection():
    from arrayroute import ArrayReader, ArraySpec
    from frontier_data.query import ArrayFieldReader

    class MetadataOnly:
        spec = ArraySpec((100, 200, 3), (10, 20, 3), 4)
        dtype = np.dtype("float32")

        def read(self, selection):
            raise AssertionError("Description must not load simulation fields")

    reader = ArrayFieldReader(
        ArrayReader(MetadataOnly()), ("time", "cell", "component"), "m/s", "cell", "revision-123"
    )
    description = reader.describe({"time": -1, "cell": slice(2, 1000, 3), "component": 1})
    assert description["source_shape"] == (100, 200, 3)
    assert description["selection"] == {
        "time": 99,
        "cell": {"start": 2, "stop": 200, "step": 3},
        "component": 1,
    }
    assert description["selected_dimensions"] == ["cell"]
    assert description["output_shape"] == (66,)
    assert description["units"] == "m/s" and description["snapshot_id"] == "revision-123"
    assert description["output_bytes"] == 66 * 4


def test_description_separates_decode_work_from_live_memory_with_scientific_units():
    from arrayroute import ArrayReader, ArraySpec, DecodeWindow
    from frontier_data.query import ArrayFieldReader

    class MetadataOnly:
        spec = ArraySpec((100, 200, 3), (10, 20, 3), 4, DecodeWindow(1, strided_only=True))
        dtype = np.dtype("float32")

        def read(self, selection):
            raise AssertionError("Planning must not load simulation fields")

    reader = ArrayFieldReader(ArrayReader(MetadataOnly()), ("time", "cell", "component"),
                              "m/s", "cell", "revision-123")
    selection = {"time": -1, "cell": slice(2, 200, 3), "component": 1}
    description = reader.describe(selection)
    assert description["units"] == "m/s" and description["association"] == "cell"
    assert description["decoded_bytes_upper_bound"] == 24_000
    assert description["peak_decoded_bytes_upper_bound"] == 2_400
    assert description["peak_read_buffer_bytes_upper_bound"] == 2_400 + 66 * 4
    concurrency = reader.plan_concurrency(selection, total_buffer_bytes=24_000, max_workers=8)
    assert concurrency.max_workers_without_tiling == 8


def test_prefetched_scientific_regions_keep_metadata_and_cancel(tmp_path):
    from threading import Event

    from arrayroute import ReadStats

    values = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6, 1)
    writer = DatasetWriter(tmp_path, "prefetch pressure")
    writer.write_grid_sample(
        "case", fields={"p": values}, spacing=[1, 1, 1], field_units={"p": "Pa"}, location="cell"
    )
    writer.finalize()
    reader = FieldReader(Dataset(tmp_path).sample("case"), "p", buffer_bytes=1200)
    expected = values[:, 1:4, :, 0]
    actual = np.empty_like(expected)
    stats = ReadStats()
    with reader.prefetch({"y": slice(1, 4), "component": 0}, stats=stats) as blocks:
        for tile, field in blocks:
            assert field.units == "Pa" and field.association == "cell"
            assert field.dimensions == ("x", "y", "z")
            actual[tile.destination] = field.values
            del field
    assert stats.requests > 1
    np.testing.assert_array_equal(actual, expected)
    cancelled = Event()
    cancelled.set()
    with reader.prefetch({}, cancelled=cancelled.is_set) as blocks:
        with pytest.raises(InterruptedError):
            next(blocks)


@pytest.mark.parametrize("selection", ["time", "time=0.1", "x=1:2:3:4", "x=oops"])
def test_selector_syntax_explains_array_indices(selection):
    with pytest.raises(ValueError, match="dimension="):
        parse_selectors([selection])


def test_cli_explicit_association_uses_the_requested_array(tmp_path):
    import json

    from typer.testing import CliRunner

    from frontier_data.cli import app

    writer = DatasetWriter(tmp_path / "mesh", "same field name, distinct associations")
    writer.write_mesh_sample(
        "case",
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        cells=np.array([[0, 1, 2, 3]]),
        fields={"p": np.arange(4, dtype=np.float32)},
        cell_fields={"p": np.array([99], dtype=np.float32)},
        field_units={"p": "Pa"},
        lod_levels=None,
    )
    writer.finalize()
    for location, association, expected in [
        ("point", "point", np.arange(4, dtype=np.float32)),
        ("cell", "cell", np.array([99], dtype=np.float32)),
    ]:
        out = tmp_path / f"{location}.npy"
        args = [
            "query-field",
            str(tmp_path / "mesh"),
            "case",
            "p",
            "--location",
            location,
            "--out",
            str(out),
        ]
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        receipt = json.loads(result.output)
        assert receipt["association"] == association and receipt["units"] == "Pa"
        assert receipt["source"]["location"] == location and receipt["source"]["lod"] == 0
        np.testing.assert_array_equal(np.load(out, allow_pickle=False).ravel(), expected)
    ambiguous = CliRunner().invoke(app, ["query-field", str(tmp_path / "mesh"), "case", "p"])
    assert ambiguous.exit_code != 0 and "multiple locations" in ambiguous.output
    invalid = CliRunner().invoke(
        app, ["query-field", str(tmp_path / "mesh"), "case", "p", "--location", "cell_fields"]
    )
    assert invalid.exit_code != 0
    missing = CliRunner().invoke(
        app, ["query-field", str(tmp_path / "mesh"), "case", "p", "--lod", "99"]
    )
    assert missing.exit_code != 0


def test_global_signal_selection_preserves_units(tmp_path):
    writer = DatasetWriter(tmp_path, "global hydraulic measurements")
    values = np.arange(4, dtype=np.float32)
    writer.write_grid_sample(
        "case",
        fields={"p": np.zeros((4, 2, 2, 2, 1), dtype=np.float32)},
        times=np.arange(4) * 0.1,
        spacing=[1, 1, 1],
        signals={"discharge": values},
        field_units={"p": "Pa", "discharge": "m^3/s"},
    )
    writer.finalize()
    reader = FieldReader(Dataset(tmp_path).sample("case"), "discharge")
    result = reader.read({"time": slice(1, 4, 2)})
    assert result.units == "m^3/s" and result.association == "global"
    np.testing.assert_array_equal(result.values.ravel(), values[1:4:2])


@pytest.mark.parametrize('budget,step,expected,strategy', [
    (1, 2, 'native', 'dense'), (16, 2, 'dense-gather', 'dense'),
    (16, 23, 'native', 'dense'), (1, 2, 'parallel-gather', 'parallel'),
    (1, 23, 'parallel-gather', 'parallel'),
])
def test_customer_dense_policy_reports_cost_and_exports_exact_si_field(
    tmp_path, budget, step, expected, strategy,
):
    import json

    from typer.testing import CliRunner

    from arrayroute.layout import GridLayoutPolicy
    from frontier_data.cli import app

    values = np.random.default_rng(183).normal(size=(1, 64, 64, 64, 3)).astype('f4')
    writer = DatasetWriter(tmp_path / 'data', 'dense preview')
    writer.write_grid_sample('case', fields={'velocity': values}, times=[0],
        spacing=[.2] * 3, field_units={'velocity': 'm/s'},
        storage_layout=GridLayoutPolicy(chunk_edge=8, shard_edge=32))
    writer.finalize()
    output = tmp_path / 'preview.npy'
    args = ['query-field', str(tmp_path / 'data'), 'case', 'velocity',
            '--engine', 'zarrista', '--read-strategy', strategy, '--select', 'time=0',
            '--budget-mib', str(budget), '--out', str(output)]
    for axis in ['x', 'y', 'z']:
        args.extend(['--select', f'{axis}=::{step}'])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report['read_strategy'] == expected
    assert report['source']['read_strategy'] == strategy
    assert report['units'] == 'm/s'
    assert report['peak_read_buffer_bytes_upper_bound'] <= budget << 20
    assert (report['intermediate_bytes_upper_bound'] > 0) == (expected != 'native')
    assert np.load(output).tobytes() == values[0, ::step, ::step, ::step].tobytes()
    assert report['gather_workers'] == (4 if strategy == 'parallel' else 1)


def test_customer_parallel_benchmark_reports_actual_inner_workers(tmp_path):
    import json

    from typer.testing import CliRunner

    from arrayroute.layout import GridLayoutPolicy
    from frontier_data.cli import app

    values = np.random.default_rng(62).normal(size=(3, 64, 64, 64, 3)).astype('f4')
    writer = DatasetWriter(tmp_path/'data', 'parallel preview pilot')
    writer.write_grid_sample('case', fields={'velocity': values}, times=[0, .1, .2],
        spacing=[.1]*3, field_units={'velocity': 'm/s'},
        storage_layout=GridLayoutPolicy(chunk_edge=8, shard_edge=32))
    writer.finalize()
    result = CliRunner().invoke(app, ['benchmark-field', str(tmp_path/'data'), 'case',
        'velocity', '--engine', 'zarrista', '--read-strategy', 'parallel', '--gather-workers', '8',
        '--vary', 'time', '--count', '3', '--select', 'x=::4', '--select', 'y=::2',
        '--workers', '1', '--workers', '2',
        '--total-read-budget-mib', '1', '--repeats', '1'])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report['units'] == 'm/s' and report['state'] == 'completed'
    assert report['gather_workers'] == 8
    assert all(t['verified'] and t['read_strategies'] == {'parallel-gather': 3}
               for t in report['trials'])
    actual = {t['workers']: t['gather_workers_peak'] for t in report['trials']}
    assert actual[1] > actual[2] and actual[2] >= 2
