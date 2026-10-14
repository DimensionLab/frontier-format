# Frontier data-prep benchmark

Messy cases with closure-based ground truth, so ops, judges and agents are scored on the
same defects. `src/frontier_prep/benchmark_cases.json` lists the cases (it ships in the wheel; `frontier_prep.benchmarks.load_cases()` reads it);
`frontier_prep.benchmarks` generates the synthetic ones and scores them.

```bash
uv run python -c "from frontier_prep.benchmarks import run_benchmark; import json, tempfile; \
  print(json.dumps(run_benchmark(tempfile.mkdtemp()), indent=1))"
uv run pytest packages/frontier-prep/tests/test_benchmarks.py
```

Synthetic cases run in CI (seconds, no data). Real cases need the verified hub subsets under
`datasets/hub/` and are exercised by the cleanup pass of the Phase 4 data-prep report.
