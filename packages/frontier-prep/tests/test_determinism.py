# SPDX-License-Identifier: Apache-2.0
"""Determinism proof: the same recipe on the same input gives the same content-addressed
output version, with one worker and with three, and twice in a row."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from frontier_data import DatasetWriter
from frontier_data.versions import VersionStore
from frontier_prep.base import DatasetRef
from frontier_prep.op import Context
from frontier_prep.recipes import Recipe, Step, run_recipe


def _dataset(root: Path) -> None:
    w = DatasetWriter(root, name="det", domain="cfd")
    rng = np.random.default_rng(7)
    for i in range(6):
        n = 120
        pts = rng.random((n, 2)).astype(np.float32)
        cells = rng.integers(0, n, (200, 3)).astype(np.int32)
        u = rng.normal(0, 1, (30, n, 2)).astype(np.float32)
        p = rng.normal(0, 1, (30, n, 1)).astype(np.float32)
        p[3] = p[2]  # a duplicated frame
        if i == 2:
            p[10, 5] = np.nan
        w.write_mesh_sample(
            f"traj_{i}", points=pts, cells=cells,
            fields={"U": u, "p": p}, times=0.01 * np.arange(30), parameters={"Re": 50.0 * i},
            field_units={"U": "m/s", "p": "Pa"},
        )  # fmt: skip
    w.finalize()


RECIPE = Recipe(
    name="det",
    steps=[
        Step(op="prep.fields.map", params={"decisions": {"p": "pressure"}}),
        Step(op="prep.fields.nan_fix", params={"policy": "mask"}),
        Step(op="prep.time.drop_duplicate_frames", params={}),
        Step(op="prep.time.subsample", params={"stride": 2}),
        Step(op="prep.fields.outlier_detect", params={"spatial": False}),
        Step(op="prep.dataset.split", params={"strategy": "random", "seed": 3}),
        Step(op="prep.fields.normalize", params={}),
        Step(op="prep.dataset.coverage", params={"n_suggest": 2}),
    ],
)


def _run(root: Path, workers: int) -> list[str]:
    out = run_recipe(RECIPE, DatasetRef(root), ctx=Context(workers=workers, seed=11))
    assert out.ok, [s.error for s in out.steps]
    return out.versions


def test_recipe_is_bitwise_reproducible(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _dataset(src)
    a, b, c = (tmp_path / k for k in "abc")
    for d in (a, b, c):
        shutil.copytree(src, d)
    va = _run(a, workers=1)
    vb = _run(b, workers=3)
    vc = _run(c, workers=1)
    assert va == vb == vc, (va, vb, vc)
    assert len(va) == len(RECIPE.steps)
    # the version chain is intact and the baseline matches across copies
    assert VersionStore(a).version("baseline").id == VersionStore(b).version("baseline").id


def test_seed_changes_seeded_ops_only(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _dataset(src)
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(src, a)
    shutil.copytree(src, b)
    recipe = Recipe(name="s", steps=[Step(op="prep.mesh.resample", params={"n_points": 16})])
    ra = run_recipe(recipe, DatasetRef(a), ctx=Context(seed=1))
    rb = run_recipe(recipe, DatasetRef(b), ctx=Context(seed=2))
    assert ra.versions != rb.versions  # seeded op: a different seed is a different version
