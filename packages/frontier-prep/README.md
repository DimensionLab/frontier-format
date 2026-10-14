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

| Command | What it does |
|---|---|
| `ops list`, `ops show <op>` | List ops; show the description and parameters of one |
| `plan <op> -d <dataset>` | List the changes, warnings and open questions without running |
| `preview <op> -d <dataset>` | Apply the op to a few samples in a temporary copy and show the differences |
| `run <op> -d <dataset>` | Run the op and record a new dataset version |
| `explain -d <dataset>` | Show the op, parameters and assertions that produced a version |
| `recipe run`, `recipe diff`, `recipe show` | Replay, plan or print a recipe recorded with `run --record` |
| `versions <command> <dataset>` | `log`, `diff`, `revert`, `lineage`, `tag`, `approve`, `gc`, `export-prov` |
| `ops export-tools` | Export the ops as tool definitions for an agent |

## Ops

| Category | Ops |
|---|---|
| fields | `rename`, `map`, `cast_location`, `components`, `nan_fix`, `outlier_detect`, `smooth`, `normalize` |
| units, scalars | `units.convert`, `scalars.derive` |
| mesh | `watertight_check`, `merge_seams`, `repair`, `orient_outward`, `transform`, `decimate`, `resample`, `part_ids` |
| time | `subsample`, `trim`, `align`, `drop_duplicate_frames`, `window_average`, `continuity_check` |
| dataset | `dedup`, `split`, `coverage`, `balance`, `select`, `merge`, `summarize` |
| quality | `check`, `gate` (sets the `train-ready` tag when the required checks pass) |

Op ids are `prep.<category>.<name>`, for example `prep.fields.map`. To add an op, subclass
`frontier_prep.base.BaseOp`, register it with `@register` and publish it under the entry-point
group `frontier.ops`.

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
