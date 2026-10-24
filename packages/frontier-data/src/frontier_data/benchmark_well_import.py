# SPDX-License-Identifier: Apache-2.0
"""Measure early selection in The Well import against an archived converter source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np


def load_module(baseline: Path | None):
    if baseline is None:
        from frontier_data.ingest import the_well

        return the_well
    spec = importlib.util.spec_from_file_location("well_baseline", baseline)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the archived baseline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadOptions(TypedDict, total=False):
    hdf5_strategy: Literal["auto", "native"]


def worker(args):
    module = load_module(args.baseline if args.worker == "baseline" else None)
    options: ReadOptions = (
        {}
        if args.worker == "baseline"
        else {"hdf5_strategy": "native" if args.worker == "arrayroute-native" else "auto"}
    )
    start = time.perf_counter()
    if args.operation == "convert":
        ds = module.convert_the_well(
            {"train": [args.file]},
            args.output,
            name="synthetic_3d",
            fields=["velocity"],
            max_steps=8,
            time_stride=8,
            quality=False,
            **options,
        )
        sample = ds.sample(ds.sample_ids[0])
        values = sample.field("velocity")
        times = sample.times
    else:
        reader = module.read_well_file(
            args.file, fields=["velocity"], max_steps=8, time_stride=8, **options
        )
        result = next(reader)
        reader.close()
        values, times = result["fields"]["velocity"], result["times"]
    digest = hashlib.sha256(memoryview(np.ascontiguousarray(values)))
    digest.update(memoryview(np.ascontiguousarray(times)))
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            dict(
                mode=args.worker,
                operation=args.operation,
                seconds=elapsed,
                peak_rss_bytes=peak if sys.platform == "darwin" else peak * 1024,
                sha256=digest.hexdigest(),
                shape=values.shape,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Archived pre-migration the_well.py"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["baseline", "arrayroute", "arrayroute-native"])
    parser.add_argument("--operation", choices=["read", "convert"])
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    import h5py

    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "synthetic.hdf5"
    with h5py.File(source, "w") as f:
        f.attrs.update(
            dataset_name="synthetic_3d", n_trajectories=1, n_spatial_dims=3, grid_type="cartesian"
        )
        d = f.create_group("dimensions")
        d.attrs["spatial_dims"] = ["x", "y", "z"]
        for name in ("x", "y", "z"):
            d[name] = np.linspace(0, 1, 64)
        d["time"] = np.arange(128) * 0.01
        a = f.create_dataset(
            "t1_fields/velocity",
            shape=(1, 128, 64, 64, 64, 3),
            chunks=(1, 1, 64, 64, 64, 3),
            dtype="f4",
            compression="gzip",
        )
        a.attrs.update(sample_varying=True, time_varying=True, dim_varying=[True] * 3)
        for i in range(128):
            a[0, i] = np.random.default_rng(i + 20261024).standard_normal(
                (64, 64, 64, 3), dtype=np.float32
            )
    trials = [
        (mode, operation)
        for _ in range(3)
        for operation in ("read", "convert")
        for mode in ("baseline", "arrayroute", "arrayroute-native")
    ]
    random.Random(20261024).shuffle(trials)
    started = time.time()
    results = []
    for index, (mode, operation) in enumerate(trials):
        out = args.output / f"converted-{index}"
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "frontier_data.benchmark_well_import",
                "--baseline",
                str(args.baseline),
                "--output",
                str(out),
                "--worker",
                mode,
                "--operation",
                operation,
                "--file",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        (args.output / f"trial-{index:02}.json").write_text(run.stdout)
        (args.output / f"trial-{index:02}.stderr").write_text(run.stderr)
        results.append(json.loads(run.stdout))
        if out.exists():
            shutil.rmtree(out)
    assert len({r["sha256"] for r in results}) == 1
    report = dict(
        started_unix=started,
        finished_unix=time.time(),
        results=results,
        physical_bytes=source.stat().st_size,
        logical_bytes=128 * 64**3 * 3 * 4,
        selected_bytes=8 * 64**3 * 3 * 4,
        python=sys.version,
        numpy=np.__version__,
        h5py=h5py.__version__,
        platform=sys.platform,
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        scope="Synthetic local 3D vectors, not simulated CFD. Eight of 128 steps, "
        "stride eight. Deterministic shuffled fresh-process trials; OS caches "
        "uncontrolled. Both methods return the same selected values. Baseline "
        "reads skipped steps, so this is avoided work, not equal-I/O throughput.",
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    source.unlink()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
