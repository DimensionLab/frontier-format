# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from frontier_prep.benchmarks import make_case


@pytest.fixture
def case(tmp_path: Path):
    def _make(name: str) -> Path:
        root = tmp_path / name
        make_case(name, root)
        return root

    return _make
