# frontier-prep

Data-preparation operations (ops) for Frontier datasets: field mapping, unit conversion, mesh
repair, time-series clean-up, splits and quality gates. Every op can be planned and previewed
before it runs, and a run records a new dataset version with provenance instead of editing the
dataset in place.

## Install

Install the wheel from the release bundle:

```bash
pip install --no-index --find-links wheels frontier-prep
```

## Example

```bash
python -m frontier_prep ops list
python -m frontier_prep ops show prep.time.subsample

python -m frontier_prep plan    prep.time.subsample -d my-dataset --set stride=2
python -m frontier_prep preview prep.time.subsample -d my-dataset --set stride=2
python -m frontier_prep run     prep.time.subsample -d my-dataset --set stride=2 --record clean.json

python -m frontier_prep versions log my-dataset
python -m frontier_prep recipe run clean.json -d my-other-dataset
```

```python
from frontier_prep import Context, Selection, get_op

op = get_op("prep.time.subsample")
plan = op.plan("my-dataset", {"stride": 2})  # changes, warnings, open questions
preview = op.preview("my-dataset", {"stride": 2}, sample=Selection(max_samples=2))
result = op.run("my-dataset", {"stride": 2}, ctx=Context(seed=0))
print(result.output_version, op.explain(result).steps)
```

## Commands

Every command prints JSON with `--json`. Exit codes: 0 ok, 1 error, 2 blocked (a plan has a
`fail` warning, the gate fails or a recipe step fails).

| Command                                    | What it does                                                               |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| `ops list`, `ops show <op>`                | List ops; show the description and parameters of one                       |
| `plan <op> -d <dataset>`                   | List the changes, warnings and open questions without running              |
| `preview <op> -d <dataset>`                | Apply the op to a few samples in a temporary copy and show the differences |
| `run <op> -d <dataset>`                    | Run the op and record a new dataset version                                |
| `explain -d <dataset>`                     | Show the op, parameters and assertions that produced a version             |
| `recipe run`, `recipe diff`, `recipe show` | Replay, plan or print a recipe recorded with `run --record`                |
| `versions <command> <dataset>`             | `log`, `diff`, `revert`, `lineage`, `tag`, `approve`, `gc`, `export-prov`  |
| `ops export-tools`                         | Export the ops as tool definitions for an agent                            |

## Ops

| Category       | Ops                                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| fields         | `rename`, `map`, `cast_location`, `components`, `nan_fix`, `outlier_detect`, `smooth`, `normalize`                       |
| units, scalars | `units.convert`, `scalars.derive`                                                                                        |
| mesh           | `watertight_check`, `merge_seams`, `repair`, `orient_outward`, `transform`, `decimate`, `resample`, `part_ids`, `taubin` |
| time           | `subsample`, `trim`, `align`, `drop_duplicate_frames`, `window_average`, `continuity_check`                              |
| dataset        | `dedup`, `split`, `coverage`, `balance`, `select`, `merge`, `summarize`                                                  |
| quality        | `check`, `gate` (sets the `train-ready` tag when the required checks pass)                                               |

Op ids are `prep.<category>.<name>`, for example `prep.fields.map`. To add an op, subclass
`frontier_prep.base.BaseOp`, register it with `@register` and publish it under the entry-point
group `frontier.ops`.

## Preview comparisons through ArrayRoute

Preparation contract 2.0 uses ArrayRoute for statistical comparisons, including geometry,
fields, masks, signals and time arrays. Change counts and extrema cover **every value in
the selected time window**, rather than extrapolating from a few points. Percentiles alone
use a deterministic sample; their scope and sample count are included in the result.
`Selection.percentile_samples` replaces the old, misleading `max_points` setting.

```python
preview = op.preview("my-dataset", params, sample=Selection(
    max_samples=2, max_steps=8, percentile_samples=65536,
    read_engine="tensorstore", read_buffer_bytes=32 << 20,
))
```

The CLI exposes `--max-steps`, `--percentile-samples`, `--read-engine` and
`--read-buffer-mib`. Install `frontier-data[readers]` for the optional engines.
Results retain source shapes, exact selections, separate numerical/bitwise change counts,
units and read statistics. A unit conversion reports before/after values in their own
units and suppresses physical deltas between different units. Missing units remain explicit.

Comparison buffers are bounded independently per source; codec scratch and engine caches
are additional. Transform execution, rendering, scratch-directory metadata and the result
list have separate costs. The time limit is cooperative between comparison reads and
reductions and cannot interrupt a blocked native read or a transformation. An interrupted
or failed comparison marks the preview incomplete and includes a reason; partial counts
are never returned as complete. Source metadata/data must remain fixed during comparison.

## Optional CUDA surface smoothing

`prep.mesh.taubin` smooths static triangle surfaces while retaining connectivity and field
values. It does not re-solve or interpolate simulation fields on the changed geometry.
Preview the result before adopting it in a scientific workflow. Moving meshes are rejected.

```bash
python -m frontier_prep preview prep.mesh.taubin -d my-dataset --set iterations=200
python -m frontier_prep run prep.mesh.taubin -d my-dataset --set backend=cpu --record smooth.json
```

CPU is the default. On Linux with an NVIDIA GPU and compatible CUDA driver, install the
optional dependency from the release wheel bundle and its dependency index:

```bash
pip install --find-links wheels 'frontier-prep[gpu]'
python -m frontier_prep run prep.mesh.taubin -d my-dataset --set backend=cuda
```

The optional extra pins `cupy-cuda12x==14.2.0` on Linux and requires a compatible CUDA 12
runtime/toolkit. It does not install CUDA toolkit packages into a training environment,
where they can conflict with PyTorch's CUDA dependencies. In a separate geometry worker
environment without a system toolkit, `pip install 'cupy-cuda12x[ctk]==14.2.0'` can supply
the toolkit dependencies. Keep that environment separate from the model training stack.
`cuda` fails if unavailable;
`auto` falls back to CPU only during CUDA availability checks and records the reason.
Allocation, compilation and computation errors propagate without a CPU retry. On one GPU,
use one preparation worker to avoid competing CUDA contexts.

The array API accepts an explicit boolean `(N,3)` component mask. The dataset operation
accepts the same mask as a static point field through `fixed_mask_field`:

```python
from frontier_prep.ops.smoothing import taubin_smooth

smoothed, execution = taubin_smooth(
    vertices,
    triangles,
    fixed=held_components,
    iterations=200,
    lam=0.75,
    mu=-0.76,
    backend="cuda",
)
```

Both backends use float64 coordinates and a fixed row-normalized, uniform vertex adjacency.
Each iteration applies lambda then mu, then restores the held coordinate components. The
mask is never guessed from bounds. No-face meshes are unchanged; isolated vertices in a
mesh with faces have zero neighbor average and must be explicitly fixed if they should be
held. The validated parameter ranges are iterations 0–10000, 0 < lambda <= 1, -1 <= mu < 0;
these ranges do not guarantee geometric quality or volume preservation for arbitrary meshes.

Execution results include actual backend, versions, CUDA device/runtime, topology hash,
cache status and synchronized wall timings including transfers. Dataset metadata stores
the stable execution details; timings and cache hits stay outside versioned content.
A per-process host CSR cache is bounded to four entries and 64 MiB, keyed by topology
contents. GPU arrays are not cached. GPU/CPU results have a numerical tolerance contract,
not a bitwise-equality promise; validate downstream curvature and topology-sensitive metrics
separately. This operation does not change marching cubes or curvature definitions.

The package advertises the independently versioned `ops >=1,<2` interface through its
plugin manifest. Its package release number does not need to match Frontier's release.

## Notices

- Licence: Apache-2.0.
- Determinism: each op declares `bitwise`, `tolerance` or `seeded`; a run gives the same
  version id for any worker count, and tolerances are recorded in provenance.
- Stored LOD levels of a surface are rebuilt after an op changes level 0, within a relative
  area tolerance of 1e-3.
- A versioned dataset stores files as read-only hard links, so change it through ops or
  `frontier_data` writers, not by editing files in place.
- `FRONTIER_PREP_KERNELS=auto|python|native` selects the kernel implementation; both give the
  same bytes.

## Documentation

- [Data-preparation specification](../../docs/spec/data-prep.md)

## Quality-reader configuration

`prep.quality.check` 1.2.0 accepts a shared ArrayRoute `reader` configuration and a per-worker
`read_buffer_mib` allowance. They are recipe parameters, so worker execution and replay use
the recorded policy. First-frame hashing uses the same session as the sample's quality scan.

```json
{
  "reader": {
    "engine": "tensorstore",
    "tensorstore_threads": 8,
    "tensorstore_cache_bytes": 134217728,
    "immutable": false
  },
  "read_buffer_mib": 128,
  "frame_hashes": true,
  "write_report": true
}
```

Install the optional reader engine before choosing it. A nonzero TensorStore cache is a soft
limit separate from read buffers; multiply session allowances by worker count. Numerical
scratch, FFTs and retained results need additional memory. `immutable: true` is an explicit
promise that input array values and metadata will not change during the session, not a
snapshot implementation. The default keeps freshness checks enabled. Per-sample policies
are recorded in the quality report, including results transported from worker processes.

`prep.time.continuity_check` 1.1.0 standardizes signal histories through ArrayRoute and checks
relative changes in blocks, including changes across block boundaries. Its
`normalization_workspace_mib` parameter defaults to 32 MiB per source. Fitted coefficients,
finite-only policy, channel order and fitting-source hashes are recorded in the continuity
assertion. `prep.quality.check` 1.5.0 uses the same fitted reader with its existing
`statistics_workspace_mib` parameter. These diagnostic operations leave source units and
values intact. Reader buffers, native caches and retained time/diagnostic arrays are separate
from normalization scratch; the latter arrays still grow with the number of time steps.

## Fit training-field normalization

`prep.fields.normalize` 2.0.0 fits complete finite channel rows from a declared split.
A missing, empty, duplicated, unknown or overlapping split fails instead of falling back
to all samples. All training timesteps are included by default; set `max_steps` explicitly
for deterministic evenly spaced sampling. Validation and inference reuse the recorded
coefficients without fitting their own statistics.

```json
{
  "split": "train",
  "method": "zscore",
  "max_steps": null,
  "read_engine": "tensorstore",
  "read_buffer_mib": 128,
  "statistics_workspace_mib": 32,
  "tensorstore_cache_mib": 64
}
```

The default engine is Zarr-Python, with no native cache allowance. A nonzero
`tensorstore_cache_mib` requires TensorStore. Optional engines must be installed.
ArrayRoute supplies ordered reads and fixed 4096-row statistical leaves. Per-sample
moments are merged in natural sample order, independently of worker completion order.
The mean and centred second moment replace cancellation-prone raw sum-of-squares
subtraction. Rows with any nonfinite component are excluded as a whole; fields with no
usable rows fail. Channel counts and declared units must agree across fitted samples.

The manifest records the algorithm, units, selected samples/steps, original shapes/dtypes,
selected-value SHA-256 fingerprints and read policy. Fingerprints cover selected native
field bytes, not every source file or time coordinate; preparation's input version supplies
that wider identity. Exact replay requires fixed sources, sample grouping/order, selections,
dtypes and numerical runtime. Different read buffers/workspaces preserve fitted values.
Results can differ from the previous numerical algorithm; the op version marks that change.

Numeric workspace admission requires at least `channels * (4096 * 128 + 256)` bytes.
Ordered read intervals contain at most 65,536 rows and shrink with available workspace and
read allowance. Native caches, source buffers, sample metadata and worker multiplicity are
additional to numeric scratch. This is not a global process-memory limit. Full fitting
scans every selected value; sampling changes the fit and is recorded explicitly.
