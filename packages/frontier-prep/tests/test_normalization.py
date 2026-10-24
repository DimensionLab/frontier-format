# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from arrayroute.backends import NumpyBackend
from arrayroute.budget import ReadBudget
from arrayroute.reader import ArrayReader, ReadStats
from frontier_data import DatasetWriter
from frontier_prep.base import DatasetRef
from frontier_prep.normalization import fit_field
from frontier_prep.ops.fields import NormalizeParams, NormalizeStats


def test_statistics_replay_across_budgets_and_nonfinite_rows():
    x = (1e12 + np.random.default_rng(4).integers(-5, 6, (17003, 3))).astype("f8")
    x[::101, 1] = np.nan
    x[::307, 2] = np.inf
    answers = []
    for budget in [1024, 1 << 20, 16 << 20]:
        for workspace in [2 << 20, 32 << 20]:
            reader = ArrayReader(NumpyBackend(x), buffer_bytes=budget)
            for read_ahead in [False, True]:
                answers.append(fit_field(
                    reader, [None], workspace_bytes=workspace, stats=ReadStats(),
                    read_ahead=read_ahead,
                ))
    assert all(answer == answers[0] for answer in answers)
    valid = x[np.isfinite(x).all(axis=1)]
    result = answers[0]
    assert result["sha256"] == hashlib.sha256(x.tobytes()).hexdigest()
    assert result["n"] == len(valid) and result["rows_seen"] == len(x)
    np.testing.assert_allclose(
        result["mean"], 1e12 + (valid - 1e12).mean(axis=0), rtol=0, atol=0.0001221
    )
    np.testing.assert_allclose(
        np.asarray(result["m2"]) / result["n"], (valid - 1e12).var(axis=0), rtol=2e-6, atol=2e-6
    )


def test_workspace_admission_before_source_read():
    class Unreadable(NumpyBackend):
        def read(self, selection):
            raise AssertionError("payload must not be read")

    with pytest.raises(MemoryError, match="workspace"):
        fit_field(
            ArrayReader(Unreadable(np.zeros((5, 3)))),
            [None],
            workspace_bytes=100,
            stats=ReadStats(),
        )


def dataset(root, *, invalid=None):
    writer = DatasetWriter(root, name="normalization split")
    values = {}
    for i, sid in enumerate(["train2", "train10", "heldout"]):
        x = np.random.default_rng(i).normal(size=(7, 9, 11, 3)).astype("f4") + i * 20
        if sid == "heldout":
            x[:] = 1e20
        x[2, 3, 4, 1] = np.nan
        values[sid] = x
        writer.write_grid_sample(
            sid, fields={"u": x}, times=np.arange(7), ndim=2, field_units={"u": "m/s"}
        )
    writer.manifest.splits = {"train": ["train10", "train2"], "test": ["heldout"]}
    if invalid is not None:
        writer.manifest.splits = invalid
    writer.finalize()
    return DatasetRef(root), values


def run(ref, params):
    op = NormalizeStats()
    outcomes = [
        op.apply_sample(ref.editor(sid), params, seed=0) for sid in op.plan_samples(ref, params)
    ]
    manifest = ref.manifest.model_copy(deep=True)
    op.apply_manifest(ref, manifest, params, list(reversed(outcomes)))
    return manifest.stats["normalization"]


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
@pytest.mark.parametrize("steps", [None, 3])
def test_training_only_streamed_grid_statistics(tmp_path, engine, steps):
    if engine != "zarr-python":
        pytest.importorskip(engine)
    ref, values = dataset(tmp_path)
    params = NormalizeParams(read_engine=engine, max_steps=steps)
    result = run(ref, params)
    ids = ["train2", "train10"]
    expected = np.concatenate(
        [(values[sid] if steps is None else values[sid][[0, 3, 6]]).reshape(-1, 3) for sid in ids]
    ).astype("f8")
    expected = expected[np.isfinite(expected).all(axis=1)]
    assert result["samples"] == ids
    field = result["fields"]["u"]
    assert field["n"] == len(expected) and field["units"] == "m/s"
    np.testing.assert_allclose(field["mean"], expected.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(field["std"], expected.std(axis=0), rtol=1e-14)
    assert result["source_fields"]["train2"]["u"]["selected_steps"] == (
        None if steps is None else [0, 3, 6]
    )
    # Resource policy is intentionally recorded; fitted values and identities are invariant.
    repeat = run(ref, NormalizeParams(read_buffer_mib=1, max_steps=steps, read_ahead=False))
    assert result["fields"] == repeat["fields"]
    assert result["source_fields"] == repeat["source_fields"]
    assert result["read_policy"]["read_ahead"] is True
    assert repeat["read_policy"]["read_ahead"] is False
    assert json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "splits",
    [
        {},
        {"train": [], "test": ["heldout"]},
        {"train": ["train2", "train2"]},
        {"train": ["missing"]},
        {"train": ["train2"], "test": ["train2"]},
    ],
)
def test_invalid_partition_never_falls_back_to_heldout(tmp_path, splits):
    ref, _ = dataset(tmp_path, invalid=splits)
    with pytest.raises(ValueError, match="Normalization"):
        NormalizeStats().plan_samples(ref, NormalizeParams())


def test_inconsistent_units_fail_before_manifest_write(tmp_path):
    ref, _ = dataset(tmp_path)
    op = NormalizeStats()
    params = NormalizeParams()
    outcomes = [
        op.apply_sample(ref.editor(sid), params, seed=0) for sid in op.plan_samples(ref, params)
    ]
    outcomes[1].stats["u"]["units"] = "mm/s"
    with pytest.raises(ValueError, match="units"):
        op.apply_manifest(ref, ref.manifest, params, outcomes)
    assert "normalization" not in ref.manifest.stats


def test_all_invalid_rows_rejected(tmp_path):
    ref, _ = dataset(tmp_path)
    op = NormalizeStats()
    params = NormalizeParams()
    outcome = op.apply_sample(ref.editor("train2"), params, seed=0)
    outcome.stats["u"].update(n=0, m2=[0, 0, 0], mean=[0, 0, 0])
    with pytest.raises(ValueError, match="No complete finite"):
        op.apply_manifest(ref, ref.manifest, params, [outcome])


def test_no_matching_fields_cannot_publish_empty_normalization(tmp_path):
    ref, _ = dataset(tmp_path)
    with pytest.raises(ValueError, match="No fields selected"):
        run(ref, NormalizeParams(fields=["absent"]))


@pytest.mark.parametrize("channels", [1, 3, 8, 9, 17])
@pytest.mark.parametrize("dtype", ["f4", "f8", "i4"])
def test_extrema_and_strided_selection_replay(channels, dtype):
    x = np.random.default_rng(3).integers(-10, 11, (7, 5003, channels)).astype(dtype)
    if dtype.startswith("f"):
        x[:, ::2, 0] = -0.
        x[:, 1::2, 0] = 0.
        x[:, ::17, -1] = np.nan
    selections = [(step, slice(1, None, 3), slice(None)) for step in [0, 4, 6]]
    results = []
    for read_ahead in [False, True]:
        reader = ArrayReader(NumpyBackend(x), buffer_bytes=1 << 20)
        results.append(fit_field(
            reader, selections, workspace_bytes=32 << 20, stats=ReadStats(),
            read_ahead=read_ahead,
        ))
    assert results[0] == results[1]
    expected = np.concatenate([x[s].reshape(-1, channels) for s in selections])
    assert results[0]["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    valid = expected[np.isfinite(expected).all(axis=1)]
    for key, reference in [("min", valid.min(axis=0)), ("max", valid.max(axis=0))]:
        # Compare bits as well: positive/negative zero remain distinguishable.
        assert np.asarray(results[0][key], dtype=dtype).tobytes() == reference.tobytes()


@pytest.mark.parametrize("read_ahead", [False, True])
def test_cancellation_before_any_source_read(read_ahead):
    class Unreadable(NumpyBackend):
        def read(self, selection):
            raise AssertionError("cancelled source must not be read")

    with pytest.raises(InterruptedError):
        fit_field(
            ArrayReader(Unreadable(np.zeros((5, 3)))), [None],
            workspace_bytes=32 << 20, stats=ReadStats(), read_ahead=read_ahead,
            cancelled=lambda: True,
        )


def test_read_failure_propagates_and_joins_worker():
    calls = []

    class Failing(NumpyBackend):
        def read(self, selection):
            calls.append(threading.current_thread())
            if len(calls) == 2:
                raise OSError("source unavailable")
            return super().read(selection)

    with pytest.raises(OSError, match="source unavailable"):
        fit_field(
            ArrayReader(Failing(np.zeros((200000, 3))), buffer_bytes=1 << 20),
            [None], workspace_bytes=32 << 20, stats=ReadStats(),
        )
    assert len(calls) == 2
    assert all(not thread.is_alive() for thread in calls)


def test_consumer_failure_joins_inflight_read(monkeypatch):
    import frontier_prep.normalization as module

    started, release, consumer_failed = threading.Event(), threading.Event(), threading.Event()
    calls = []

    class Blocking(NumpyBackend):
        def read(self, selection):
            calls.append(threading.current_thread())
            if len(calls) == 2:
                started.set()
                assert release.wait(5)
            return super().read(selection)

    def fail(self, values, valid=None):
        assert started.wait(5)
        consumer_failed.set()
        raise RuntimeError("consumer failed")

    monkeypatch.setattr(module.CanonicalChannelMoments, "update", fail)
    with ThreadPoolExecutor(max_workers=1) as caller:
        job = caller.submit(
            fit_field,
            ArrayReader(Blocking(np.zeros((200000, 3))), buffer_bytes=1 << 20),
            [None], workspace_bytes=32 << 20, stats=ReadStats(),
        )
        try:
            assert consumer_failed.wait(5)
            assert not job.done()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="consumer failed"):
            job.result(timeout=5)
    assert len(calls) == 2
    assert all(not thread.is_alive() for thread in calls)


def test_shared_read_allowance_does_not_deadlock_or_leak():
    x = np.arange(20003 * 3, dtype="f4").reshape(-1, 3)
    pool = ReadBudget(1 << 20)
    reader = ArrayReader(NumpyBackend(x), buffer_bytes=1 << 20, shared_budget=pool)
    with ThreadPoolExecutor(max_workers=1) as caller:
        job = caller.submit(
            fit_field, reader, [None], workspace_bytes=32 << 20, stats=ReadStats(),
        )
        result = job.result(timeout=5)
    assert result["sha256"] == hashlib.sha256(x.tobytes()).hexdigest()
    with pool.reserve(pool.max_bytes):
        pass
