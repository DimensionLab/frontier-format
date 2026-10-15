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
