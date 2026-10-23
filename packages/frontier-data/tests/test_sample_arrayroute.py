# SPDX-License-Identifier: Apache-2.0
"""Actual scientific selections, ownership and resource admission through Sample."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
import zarr

from arrayroute.session import LocalReadSession
from frontier_data import Dataset, DatasetWriter
from frontier_data.arrays import Sample


@pytest.fixture
def dataset(tmp_path):
    w = DatasetWriter(tmp_path, 'sample-route')
    values = np.arange(8 * 12 * 9 * 2, dtype=np.float32).reshape(8, 12, 9, 2)
    values[0, 0, 0, 0] = -0.0
    values[0, 0, 1, 0] = np.nan
    w.write_grid_sample('grid', fields={'velocity': values}, times=np.arange(8.) * 0.25,
                        spacing=[0.5, 0.75], static={'wet': np.ones((12, 9), bool)},
                        signals={'force': np.arange(24., dtype=np.float32).reshape(8, 3)})
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    moving = points[None] + np.arange(8, dtype=np.float32)[:, None, None]
    w.write_mesh_sample('mesh', points=points, cells=np.array([[0, 1, 2], [1, 3, 2]]),
                        fields={'pressure': np.arange(32., dtype=np.float32).reshape(8, 4)},
                        cell_fields={'stress': np.arange(16., dtype=np.float32).reshape(8, 2)},
                        points_t=moving, times=np.arange(8.) * 0.25)
    w.finalize()
    return Dataset(tmp_path)


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert actual.flags.writeable


@pytest.mark.parametrize('engine', ['zarr-python', 'tensorstore', 'zarrista'])
def test_field_signal_geometry_selection_parity_and_ownership(dataset, engine):
    session = LocalReadSession(engine, tensorstore_threads=2)
    grid = dataset.sample('grid', session=session)
    assert grid.read_session is session
    original = np.asarray(grid.array('velocity')[...])
    signals = np.asarray(grid.group['signals/force'][...])
    for t in [None, 0, -1, np.int64(2), slice(1, 7, 2), [7, 2, 2, -8], [], slice(0, 0)]:
        selection = slice(None) if t is None else t
        if isinstance(t, list) and not t:
            selection = slice(0, 0)
        exact(grid.field('velocity', t), original[selection])
        exact(grid.signal('force', t), signals[selection])
    exact(grid.times, np.asarray(grid.group['times'][...]))
    exact(grid.static('wet'), np.asarray(grid.group['static/wet'][...]))
    coordinates = [np.asarray(grid.group['coords'][a][...]) for a in grid.axes]
    expected = np.stack([v.reshape(-1) for v in np.meshgrid(*coordinates, indexing='ij')],
                        axis=1).astype(np.float32)
    exact(grid.points(), expected)
    gathered = grid.field('velocity', [3, 3])
    gathered[0] = 999
    exact(gathered[1], original[3])
    exact(grid.field('velocity', 3), original[3])
    assert grid.read_stats.requests > 0
    restored = pickle.loads(pickle.dumps(grid))
    assert restored.read_stats.requests == 0
    exact(restored.field('velocity', 3), original[3])

    mesh = dataset.sample('mesh', session=LocalReadSession(engine, tensorstore_threads=2))
    for name, location, key in [('pressure', 'point', 'fields/pressure'),
                                ('stress', 'cell', 'cell_fields/stress')]:
        exact(mesh.field(name, [7, 1, 1], location=location),
              np.asarray(mesh.group[key][...])[[7, 1, 1]])
    exact(mesh.points(t=-1), np.asarray(mesh.group['points_t'][-1]))
    exact(mesh.points(), np.asarray(mesh.group['points'][...]))
    exact(mesh.cells, np.asarray(mesh.group['cells'][...]))
    exact(mesh.cell_types, np.asarray(mesh.group['cell_types'][...]))


def test_signal_list_reads_only_distinct_requested_rows(dataset, monkeypatch):
    sample = dataset.sample('grid')
    reader = sample.read_session.open(sample.store_path, 'signals/force')
    original = reader.backend.read
    reads = []
    def track(selection):
        reads.append(selection)
        return original(selection)
    monkeypatch.setattr(reader.backend, 'read', track)
    result = sample.signal('force', [6, 2, 6])
    assert result.tolist() == [[18., 19., 20.], [6., 7., 8.], [18., 19., 20.]]
    assert reads == [(slice(2, 3, 1), slice(0, 3, 1)),
                     (slice(6, 7, 1), slice(0, 3, 1))]


def test_admission_precedes_payload_fetch_and_output_allocation(dataset, monkeypatch):
    sample = dataset.sample('grid', session=LocalReadSession(buffer_bytes=1024))
    reader = sample.read_session.open(sample.store_path, 'fields/velocity')
    def forbidden(_selection):
        pytest.fail('A rejected selection reached payload I/O')
    monkeypatch.setattr(reader.backend, 'read', forbidden)
    with pytest.raises(MemoryError, match='select fewer steps'):
        sample.field('velocity')
    with pytest.raises(MemoryError, match='select fewer steps'):
        sample.field('velocity', [1] * 100)
    assert sample.read_stats.requests == 0
    with pytest.raises(MemoryError, match='Grid points exceed'):
        sample.points()
    assert sample.read_stats.requests == 0


@pytest.mark.parametrize('t', [True, [True], [0.5], ['1'], [[0, 1]]])
def test_invalid_time_indices_are_not_silently_coerced(dataset, t):
    with pytest.raises((TypeError, ValueError)):
        dataset.sample('grid').field('velocity', t)


def test_custom_store_refresh_and_steady_time_semantics(dataset, tmp_path):
    source = dataset.sample('grid')
    store = tmp_path / 'custom.zarr'
    import shutil
    shutil.copytree(source.store_path, store)
    sample = Sample.open_store(store)
    before = sample.signal('force', 0)
    group = zarr.open_group(str(store), mode='r+')
    group['signals/force'][0] = [42, 43, 44]
    sample.refresh()
    assert sample.read_stats.requests == 0
    assert sample.signal('force', 0).tolist() == [42, 43, 44]
    assert before.tolist() == [0, 1, 2]
    assert sample.store_path == store
    writer = DatasetWriter(tmp_path / 'steady', 'steady')
    writer.write_grid_sample('g', fields={'p': np.ones((3, 4))})
    writer.finalize()
    steady = Dataset(tmp_path / 'steady').sample('g')
    exact(steady.field('p', 100), steady.field('p'))


def test_generated_geometry_obeys_shared_budget_and_releases_it(dataset):
    from arrayroute.budget import ReadBudget
    pool = ReadBudget(4096)
    sample = dataset.sample('grid', session=LocalReadSession(buffer_bytes=8192, read_budget=pool))
    coordinates = sample.points()
    assert pool.snapshot()['used_bytes'] == 0
    assert pool.snapshot()['peak_bytes'] > coordinates.nbytes
    denied = dataset.sample('grid', session=LocalReadSession(
        buffer_bytes=8192, read_budget=ReadBudget(128)))
    with pytest.raises(MemoryError, match='shared budget'):
        denied.points()
    assert denied.read_stats.requests == 0


@pytest.mark.parametrize('engine', ['zarr-python', 'tensorstore', 'zarrista'])
def test_refresh_reopens_changed_array_shape(dataset, engine):
    sample = dataset.sample('grid', session=LocalReadSession(engine, tensorstore_threads=2))
    sample.signal('force')
    group = zarr.open_group(str(sample.store_path), mode='r+')
    group['signals/force'].resize((9, 3))
    group['signals/force'][8] = [8, 9, 10]
    group.attrs['revision_note'] = 'updated'
    with pytest.raises(ValueError, match='fresh read session'):
        sample.refresh()
    sample.refresh(session=LocalReadSession(engine, tensorstore_threads=2))
    assert sample.attrs['revision_note'] == 'updated'
    assert sample.signal('force', -1).tolist() == [8, 9, 10]
