# SPDX-License-Identifier: Apache-2.0
"""Every synthetic benchmark case: the designated ops recover the closure truth."""

from __future__ import annotations

from pathlib import Path

import pytest

from frontier_prep.benchmarks import GENERATORS, load_cases, score_case


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_case(case, name: str) -> None:
    root: Path = case(name)
    out = score_case(name, root)
    assert out["ok"], out


def test_cases_json_lists_every_generator() -> None:
    cases = load_cases()
    assert {c["name"] for c in cases["synthetic"]} == set(GENERATORS)
