# SPDX-License-Identifier: Apache-2.0
"""Measure metadata inspection versus an explicit storage walk on an unchanged dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.is_relative_to(dataset):
        parser.error("output must be outside the measured dataset")
    manifest = dataset / "manifest.json"
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    work = [(repeat, measured) for repeat in range(args.repeats) for measured in (False, True)]
    random.Random(17).shuffle(work)
    method = {
        "dataset": str(dataset),
        "manifest_sha256": before,
        "order": work,
        "platform": platform.platform(),
        "python": sys.version,
        "scope": "Fresh CLI process, bounded sample page; latency includes interpreter/imports",
        "cache_state": "OS uncontrolled; concurrent work must be recorded by the caller",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    (output / "method.json").write_text(json.dumps(method, indent=2))
    results, reference = [], None
    for repeat, measured in work:
        command = [sys.executable, "-m", "frontier_data", "inspect", str(dataset), "--json"]
        if measured:
            command.append("--measure-storage")
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        elapsed = time.perf_counter() - started
        record = json.loads(process.stdout)
        measured_bytes = record.pop("storage_bytes")
        record.pop("storage_measured")
        if reference is None:
            reference = record
        elif reference != record:
            raise ValueError("Catalog metadata changed across trials")
        result = {
            "repeat": repeat,
            "measure_storage": measured,
            "elapsed_s": elapsed,
            "storage_bytes": measured_bytes,
        }
        results.append(result)
        with (output / "results.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != before:
        raise ValueError("Source manifest changed")
    summary = {
        "verified_same_catalog": True,
        "total_samples": reference["total_samples"] if reference else None,
        "displayed_samples": len(reference["samples"]) if reference else None,
        "trials": len(results),
        "median_seconds": {
            label: statistics.median(
                r["elapsed_s"] for r in results if r["measure_storage"] == measured
            )
            for label, measured in [("metadata", False), ("storage_walk", True)]
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
