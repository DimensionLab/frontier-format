# SPDX-License-Identifier: Apache-2.0
import numpy as np

from frontier_data import DatasetWriter
from frontier_prep.op import get_op


def test_signal_continuity_handles_normalized_block_boundaries(tmp_path):
    values = np.random.default_rng(51).normal(size=(9001, 4)).astype('f4')
    root = tmp_path / 'signals'
    writer = DatasetWriter(root, name='continuity')
    writer.write_grid_sample('case', fields={'observed': np.ones((len(values), 1, 1), dtype='f4')},
                             ndim=2, times=np.arange(len(values), dtype='f8'),
                             signals={'sensors': values}, steps_per_chunk=4096)
    writer.finalize()
    op = get_op('prep.time.continuity_check')
    result = op.run(root, {'normalization_workspace_mib': 4, 'write_signal': False})
    assertions = [a for a in result.assertions if a.check == 'temporal_continuity']
    assert len(assertions) == 1
    detail = assertions[0].detail
    assert detail['signal_normalization']['workspace_bytes'] == 4 << 20
    x = values.astype('f8')
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    relative = np.linalg.norm(np.diff(x, axis=0), axis=1) / np.linalg.norm(x[:-1], axis=1)
    np.testing.assert_allclose(detail['median_step_change'], np.median(relative), rtol=1e-12)
