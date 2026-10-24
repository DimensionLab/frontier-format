# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.session import LocalReadSession
from frontier_data import Dataset, DatasetWriter
from frontier_data.quality_transient import inspect_sample, scan_signals, signal_reader


def test_signal_quality_uses_bounded_source_and_records_fitted_policy(tmp_path):
    values = np.random.default_rng(119).normal(size=(100_003, 4)).astype('f4')
    writer = DatasetWriter(tmp_path / 'data', name='signal-quality')
    writer.write_grid_sample('case', fields={'observed': np.ones((len(values), 1, 1), dtype='f4')},
                             ndim=2, times=np.arange(len(values), dtype='f8'),
                             signals={'sensors': values}, steps_per_chunk=4096)
    writer.finalize()
    sample = Dataset(tmp_path / 'data').sample(
        'case', session=LocalReadSession(buffer_bytes=1 << 20),
    )
    with pytest.raises(MemoryError):
        sample.signal('sensors')
    normalized, names, details = signal_reader(sample, workspace_bytes=4 << 20)
    selected = normalized.read((slice(10, 100, 3), slice(None)))
    reference = values.astype('f8')
    reference = (reference - reference.mean(axis=0)) / reference.std(axis=0)
    np.testing.assert_allclose(selected, reference[10:100:3], rtol=1e-12, atol=1e-13)
    assert names == ['sensors_0', 'sensors_1', 'sensors_2', 'sensors_3']
    assert details['workspace_bytes'] == 4 << 20
    assert len(details['parameters'][0]['source_sha256']) == 64
    steps = np.array([0, 1, 99, len(values)-1])
    scan = scan_signals(inspect_sample(sample), steps, workspace_bytes=4 << 20)
    assert scan.measured and scan.normalization == details
    np.testing.assert_allclose(scan.norm, np.linalg.norm(reference[steps], axis=1), rtol=1e-12)
