# SPDX-License-Identifier: Apache-2.0
"""Time the reference (numpy) and native (Rust) kernels on one surface sample and check that
they return the same bytes.

    python packages/frontier-prep/benchmarks/bench_kernels.py <sample dir> [--repeat 5]

Prints a Markdown table: seconds per call (minimum of ``--repeat`` runs), the speed-up and
whether the two outputs are bitwise equal. ``RAYON_NUM_THREADS`` sets the native thread
count (default: all cores).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from frontier_data.arrays import Sample
from frontier_prep import kernels
from frontier_prep.ops import mesh as mk
from frontier_prep.ops import units as un


def _best(fn: Callable[[], Any], repeat: int) -> tuple[float, Any]:
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def _arrays(x: Any) -> list[np.ndarray]:
    if isinstance(x, mk.TopologyTables):
        return [x.order, x.inv, x.start, x.counts, x.uniq]
    if isinstance(x, np.ndarray):
        return [x]
    return [a for a in x if isinstance(a, np.ndarray)]


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict):
        return json.dumps(a) == json.dumps(b)
    xs, ys = _arrays(a), _arrays(b)
    scalars = [v for v in (a if isinstance(a, tuple) else ()) if not isinstance(v, np.ndarray)]
    others = [v for v in (b if isinstance(b, tuple) else ()) if not isinstance(v, np.ndarray)]
    return scalars == others and all(
        x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        for x, y in zip(xs, ys, strict=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Time reference and native prep kernels.")
    ap.add_argument("sample", type=Path, help="sample directory (…/samples/<id>)")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    s = Sample.open(args.sample, args.sample.name)
    p, f = np.asarray(s.points()), np.asarray(s.cells)
    with kernels.forced("native"):
        merged = mk.merge_vertices(p, f)
        info, threads = kernels.describe(), kernels.threads()
    rng = np.random.default_rng(0)
    flipped = merged[1].copy()
    sel = rng.random(len(flipped)) < 0.3
    flipped[sel] = flipped[sel][:, [0, 2, 1]]
    field = np.asarray(s.field(s.field_names("cell")[0], location="cell"))

    cases: list[tuple[str, Callable[[], Any]]] = [
        ("topology audit (`topology`)", lambda: mk.topology(p, f)),
        ("topology audit, merged mesh", lambda: mk.topology(merged[0], merged[1])),
        ("edge tables (`TopologyTables.build`)", lambda: mk.TopologyTables.build(f)),
        ("components (tables + labels)", lambda: mk.TopologyTables.build(f).components()),
        ("vertex merge (`merge_vertices`, tol 0)", lambda: mk.merge_vertices(p, f)),
        ("vertex merge (tol 1e-4)", lambda: mk.merge_vertices(p, f, 1e-4)),
        ("orientation (30 % flipped)", lambda: mk.orient_consistently(flipped)),
        ("orientation (consistent)", lambda: mk.orient_consistently(merged[1])),
        ("duplicate faces", lambda: mk.duplicate_faces(f)),
        ("face areas", lambda: mk.face_areas(p, f)),
        (
            f"unit scaling (`scale_block`, {field.nbytes / 1e6:.1f} MB)",
            lambda: un.scale_block(field, 1.225),
        ),
    ]
    print(
        f"{args.sample}: {len(p):,} points ({p.dtype}), {len(f):,} faces ({f.dtype}); "
        f"{platform.machine()}, {os.cpu_count()} cores, load {os.getloadavg()[0]:.1f}; "
        f"native kernels {info['version']} on {threads} threads; min of {args.repeat}\n"
    )
    print("| kernel | Python (s) | Rust (s) | speed-up | bitwise equal |")
    print("|---|---|---|---|---|")
    for name, fn in cases:
        with kernels.forced("python"):
            t_py, a = _best(fn, args.repeat)
        with kernels.forced("native"):
            t_rs, b = _best(fn, args.repeat)
        print(
            f"| {name} | {t_py:.3f} | {t_rs:.3f} | {t_py / t_rs:.1f}x | "
            f"{'yes' if _equal(a, b) else 'NO'} |"
        )


if __name__ == "__main__":
    main()
