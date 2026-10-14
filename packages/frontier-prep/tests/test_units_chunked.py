# SPDX-License-Identifier: Apache-2.0
"""The chunked unit conversion writes byte-identical zarr arrays (metadata and every chunk
file) and the same meta.json as the whole-array ``write_field`` path it replaced, for
steady surfaces (cell field + point copy, several row chunks) and transient meshes (time
chunks holding several steps), with a factor and with factor 1."""

from __future__ import annotations

import filecmp
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from frontier_data import DatasetWriter
from frontier_data.mesh import SurfaceMesh
from frontier_prep.io import SampleEditor
from frontier_prep.ops.units import convert_field_chunked, scale_block


def _reference_convert(ed: SampleEditor, name: str, loc: str, factor: float) -> None:
    """The pre-optimisation path: read everything, scale, write wholesale."""
    data = ed.field(name, location=loc)
    if factor != 1.0:
        data = (data.astype(np.float64) * factor).astype(np.float32)
    ed.write_field(name, data, location=loc, units="Pa", meta_extra={"factor": factor})


def _dataset(root: Path) -> None:
    rng = np.random.default_rng(5)
    w = DatasetWriter(root, name="units", domain="cfd")
    # steady surface: 2.5 row chunks of cells so the rewrite spans several chunks
    n = 50_000
    pts = rng.random((n, 3)).astype(np.float32)
    faces = rng.integers(0, n, (int(2.5 * (1 << 18)) // 3 * 3 // 3, 3)).astype(np.int32)
    mesh = SurfaceMesh(points=pts, faces=faces)
    mesh.cell_fields["p"] = rng.normal(0, 1e3, (mesh.n_faces, 1)).astype(np.float32)
    mesh.cell_fields["tau"] = rng.normal(0, 1, (mesh.n_faces, 3)).astype(np.float32)
    mesh.with_point_fields_from_cells()
    w.write_surface_sample("surf", mesh, field_units={"p": "m**2/s**2", "tau": "m**2/s**2"})
    # transient mesh: 30 steps, several steps per chunk
    m = 400
    w.write_mesh_sample(
        "traj",
        points=rng.random((m, 3)).astype(np.float32),
        cells=rng.integers(0, m, (900, 3)).astype(np.int32),
        fields={"p": rng.normal(0, 1, (30, m, 1)).astype(np.float32)},
        times=0.01 * np.arange(30),
        field_units={"p": "m**2/s**2"},
    )
    w.finalize()


def _tree(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file()}


def _assert_identical(a: Path, b: Path) -> None:
    ta, tb = _tree(a), _tree(b)
    assert ta.keys() == tb.keys(), set(ta) ^ set(tb)
    for rel in ta:
        assert filecmp.cmp(ta[rel], tb[rel], shallow=False), rel


@pytest.mark.parametrize("factor", [1.0, 1.225, -2.0])
def test_chunked_matches_whole_array(tmp_path: Path, factor: float) -> None:
    src = tmp_path / "src"
    _dataset(src)
    a, b = tmp_path / "chunked", tmp_path / "reference"
    shutil.copytree(src, a)
    shutil.copytree(src, b)
    for sid, locs in (("surf", ("cell", "point")), ("traj", ("point",))):
        ea = SampleEditor(a / "samples" / sid, sid)
        eb = SampleEditor(b / "samples" / sid, sid)
        for loc in locs:
            n_read, n_written = convert_field_chunked(
                ea, "p", loc, factor, units="Pa", meta_extra={"factor": factor}
            )
            _reference_convert(eb, "p", loc, factor)
            assert n_read > 0 and n_written == n_read
            arr = ea.array("p", loc)
            assert arr.dtype == np.float32
            assert np.array_equal(ea.field("p", location=loc), eb.field("p", location=loc))
        assert json.loads((a / "samples" / sid / "meta.json").read_text()) == json.loads(
            (b / "samples" / sid / "meta.json").read_text()
        )
        assert not list((a / "samples" / sid).glob(".*converting"))
    _assert_identical(a / "samples", b / "samples")


def test_scale_block_kernel() -> None:
    x = np.array([1.0, -2.5, 3e-7, 1e30], dtype=np.float32)
    assert np.array_equal(scale_block(x, 1.0), x)
    y = scale_block(x, 1.225)
    assert y.dtype == np.float32
    assert y.tolist() == (x.astype(np.float64) * 1.225).astype(np.float32).tolist()
    i = np.array([1, 2], dtype=np.int32)
    assert scale_block(i, 1.0).dtype == np.int32
    assert scale_block(i, 2.0).tolist() == [2.0, 4.0]
