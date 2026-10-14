# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from frontier_data.mesh import SurfaceMesh, make_box_mesh


@pytest.fixture
def box() -> SurfaceMesh:
    return make_box_mesh(subdivisions=8)
