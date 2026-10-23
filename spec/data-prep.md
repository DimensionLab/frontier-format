<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontier data preparation: operations, versions, provenance

Normative description of the data-preparation contract that `frontier_prep` (operations,
recipes, CLI) and `frontier_data.versions` (immutable versions, provenance, diff, revert)
implement. The server, the app, the Rust CLI and the agent harness build on these types; the
JSON shapes below are the interface, the Python classes are the reference implementation.
Contract version: **1.0**. Additive changes keep the major; a renamed or removed field bumps it.

## 1. Vocabulary

- **Dataset**: a `frontier-dataset` directory (`docs/spec/dataset-format.md`).
- **Version**: an immutable snapshot of a dataset identified by the hash of its content tree.
  Versions share unchanged files (content-addressed objects); a rename copies nothing.
- **Op** (operation): a named, versioned, deterministic transform or check with typed
  parameters. Ops never mutate a version; `run` produces the next version.
- **Plan / Preview / Result / Explanation**: the four answers an op gives.
- **Recipe**: an ordered list of `(op, params)` saved as JSON, re-runnable on any dataset.
- **Provenance**: the append-only `provenance.jsonl` of a dataset: op records, assertions,
  approvals, version records. The DAG of versions is derived from it.
- **Assertion**: a quality statement (`check`, `status`, `target`, `detail`) recorded by a
  `quality.*` op; `quality.gate` decides from assertions whether a version is train-ready.

## 2. Operation contract

Reference: `frontier_prep.op`. Python shape:

```python
class Op(Protocol):
    id: str            # "prep.fields.rename"; namespaced, dotted, stable
    version: str       # semver of the op implementation; results carry it
    category: str      # fields | units | scalars | mesh | time | dataset | quality | other
    summary: str; description: str; when_to_use: str
    determinism: "bitwise" | "tolerance" | "seeded"
    Params: type[pydantic.BaseModel]   # extra="forbid"; JSON Schema via model_json_schema()

    def plan(dataset, params) -> Plan            # pure, cheap: metadata and small reads only
    def preview(dataset, params, *, sample: Selection) -> Preview   # bounded trial run
    def run(dataset, params, *, ctx: Context) -> Result             # deterministic, new version
    def explain(result) -> Explanation           # human + machine account of what was done
```

Rules:

1. **Purity.** `plan` never writes. `preview` writes only to a scratch directory it deletes.
   `run` writes a new version through `frontier_data.versions` and never edits files in place
   (object files are read-only hardlinks; writers rewrite arrays wholesale).
2. **Determinism.** Same input version + same params + same op version => same output version
   hash, for any worker count. Randomness takes `ctx.seed` (default 0) and is recorded.
   Reductions over samples use the sorted sample id order; reductions inside an array use
   fixed-order float64 accumulation. An op that cannot promise bit-identity declares
   `determinism: "tolerance"` and records `tolerance: {metric: value}` in Plan and Result
   (section 9 lists them).
3. **Look before leaping.** Every op has `plan` and `preview`; a UI or an agent calls them
   first. `run` requires an approval (actor + approver) unless policy grants autonomy.
4. **Params are closed.** Unknown parameter keys are an error (`extra="forbid"`).
5. **Judgments are parameters.** When an op needs a semantic decision (is `p` a pressure? which
   unit?), `plan` returns `judgments[]` with options and a prior; the decision comes back as a
   parameter. Execution never depends on a model call.
6. **Idempotence.** Running an op whose plan is `noop` creates no version.

### 2.1 Types (JSON)

`Plan`

```json
{"contract":"1.0","op":"prep.fields.rename","op_version":"1.0.0","params":{...},
 "dataset":"ahmedml-40","input_version":"v_8f3c…",
 "changes":[{"target":"samples/run_1/cell_fields/pMean","action":"rename",
             "detail":{},"before":{"name":"pMean"},"after":{"name":"pressure"}}],
 "cost":{"n_samples":40,"bytes_read":0,"bytes_written":0,"est_seconds":1.2,"parallel":true},
 "warnings":[{"severity":"warn","code":"unit-unknown","message":"…","target":"…"}],
 "judgments":[{"question":"Is `p` a static pressure?","target":"fields/p",
               "options":["pressure","kinematic_pressure"],"prior":[0.7,0.3],"param":"map.p"}],
 "determinism":"bitwise","tolerance":null,"noop":false,"summary":"Rename 2 fields in 40 samples"}
```

`Preview` = `{plan, samples[], diffs[FieldDiff], stats{}, renders[Render], elapsed_s,
truncated, notes[]}` where `FieldDiff` = `{target, n_before, n_after, n_changed, frac_changed,
max_abs_diff, rel_l2_diff, before{min,max,mean,std,p01,p99,nan_frac}, after{…}, note}` and
`Render` = `{kind: "svg"|"png", caption, data (SVG text or base64 PNG), target, width, height}`.
A preview is bounded by `Selection` = `{samples[], max_samples=3, max_steps=8,
percentile_samples=65536, read_engine="zarr-python", read_buffer_bytes=134217728,
time_budget_s=20, renders=true}`. Contract 2.0 replaces `max_points`: change counts and
extrema scan all selected values; only percentiles are sampled. Diffs retain numerical and
bitwise counts, source shapes, exact selections, units and the algorithm identifier.
`truncated` marks an interrupted or failed preview, with its reason in `notes`. The comparison
deadline is cooperative; operation execution and rendering have separate resource costs.

`Result` = `{contract, op, op_version, params, dataset, input_version, output_version,
changes[], assertions[Assertion], warnings[], determinism, tolerance, content_hashes{target:
sha256}, timing{started_at, elapsed_s, plan_s, apply_s, commit_s, workers, bytes_read,
bytes_written}, machine{hostname, platform, python, frontier_prep, frontier_data, prep_kernels:
native|python, prep_kernels_version}, actor{kind:
human|agent|recipe|system, name, session, approved_by}, provenance_id, noop, summary, stats{}}`.

`Explanation` = `{contract, op, summary, steps[], machine{}}`.

`Context` (input of `run`) = `{seed=0, workers=1, actor, message, dry_run=false, commit=true,
time_budget_s}`.

### 2.2 Registry and plugins

`frontier_prep.op.REGISTRY` holds ops by id. First-party ops live in `frontier_prep.ops`.
Third-party packages publish ops under the entry-point group **`frontier.ops`**; an entry
point resolves to an op instance, an op class, an iterable of ops or a mapping id -> op.

```toml
[project.entry-points."frontier.ops"]
myvendor = "myvendor_prep.ops:OPS"
```

### 2.3 Tool schema (agents)

`python -m frontier_prep ops export-tools --json [--per-mode]` prints

```json
{"contract":"1.0","generated_at":"…","entry_point_group":"frontier.ops",
 "tools":[{"name":"prep_fields_rename","description":"…",
           "input_schema":{"type":"object","properties":{
               "dataset":{"type":"string"},
               "params":{…op JSON Schema…},
               "mode":{"enum":["plan","preview","run"],"default":"plan"},
               "selection":{…Selection schema…},
               "seed":{"type":"integer"},"message":{"type":"string"},
               "approval":{"type":"string"}},
             "required":["dataset"],"additionalProperties":false},
           "x-frontier":{"op":"prep.fields.rename","op_version":"1.0.0","category":"fields",
                         "determinism":"bitwise","requires_approval":true,
                         "changes_fields":true,"changes_geometry":false,
                         "changes_manifest":true,"produces_assertions":false,
                         "result_schema":{"plan":"Plan","preview":"Preview","run":"Result"}}}],
 "types":{"Plan":{…},"Preview":{…},"Result":{…},"Explanation":{…},"Selection":{…},"Context":{…}}}
```

Tool names are the op id with dots replaced by underscores (`--per-mode` emits
`<name>__plan`, `<name>__preview`, `<name>__run`). The MCP server (`frontier_agent`) exposes
exactly these; a call with `mode: "run"` without `approval` must be refused unless policy
grants autonomy for the op's category.

## 3. Versions

Reference: `frontier_data.versions`. Layout inside a dataset directory:

```
<dataset>/
  manifest.json, samples/…, quality/…      # the working tree (always a readable dataset)
  .frontier/
    objects/<aa>/<sha256>                   # content-addressed, read-only files
    versions/<version_id>.json              # {"id","parent","created_at","message","tree":
                                            #  {"<path>": {"hash","size"}}, "op": <prov id>,
                                            #  "stats": {"n_files","n_bytes","n_new_bytes"}}
    HEAD                                    # current version id
    refs/<name>                             # tags: train-ready, baseline, …
    index.json                              # working-tree cache {path: {hash,size,mtime,ino}}
    provenance.jsonl                        # append-only records (section 4)
```

- **Object**: a file (a zarr chunk, a `zarr.json`, a `meta.json`, …) stored once under its
  sha256. Working-tree files are hardlinks to objects (copies when the filesystem refuses
  hardlinks), so a checkout costs directory entries, not bytes, and an op that renames a
  field writes one new `zarr.json` and no chunk.
- **Version id**: `"v_" + sha256(sorted "path\0hash\n" lines)[:16]` — content-addressed, so two
  machines producing the same bytes produce the same id (the determinism test relies on it).
- **Baseline migration**: `versions init <dataset>` hashes an existing dataset into objects and
  records version 0 with an `op` record `prep.baseline` carrying the manifest's `pipeline` /
  `source`.
- **Working tree** is the mutable HEAD checkout; `status` lists modified/added/removed paths;
  `commit` records them; a dirty tree blocks `run` unless `--allow-dirty`.
- **GC**: `versions gc` deletes objects referenced by no version and not in the index.

Operations (Python / CLI `python -m frontier_data versions …` / future `frontier data …`):

| call                          | returns                                                                |
| ----------------------------- | ---------------------------------------------------------------------- |
| `init(root)`                  | baseline version                                                       |
| `commit(root, message, op=…)` | new version id (noop when nothing changed)                             |
| `log(root)`                   | versions newest first with message, op id, actor, stats                |
| `diff(root, a, b)`            | section 5                                                              |
| `revert(root, version)`       | new version whose tree equals `version` (history kept)                 |
| `checkout(root, version)`     | working tree := version (HEAD moves; no new version)                   |
| `lineage(root, target)`       | versions and ops that touched `samples/<id>`, a field, or the manifest |
| `gc(root)`                    | removed objects, bytes freed                                           |
| `tag(root, name, version)`    | ref                                                                    |

## 4. Provenance

`provenance.jsonl`: one JSON object per line, append-only, each with `id` (sha256 of the
record without `id`, first 16 hex), `kind`, `at` (ISO 8601 UTC), `actor`.

| kind        | fields                                                        |
| ----------- | ------------------------------------------------------------- | -------------------------------------------------------- |
| `version`   | `version`, `parent`, `message`, `op_record`                   |
| `op`        | the `Result` (section 2.1), plus `kernels`: `{backend: native | python, version, spec_revision, requested}` (section 11) |
| `assertion` | `version`, `check`, `status`, `target`, `detail`, `op_record` |
| `approval`  | `version` or `op_record`, `approved_by`, `decision: approve   | reject`, `note`                                          |
| `note`      | free text attached to a version                               |

The DAG: `version.parent` edges plus `op.input_version -> op.output_version`. Records are
never rewritten; a server signs them when connected and merges offline logs on sync by id.

## 5. Diff

`diff(a, b)` returns

```json
{"a":"v_…","b":"v_…","files":{"added":n,"removed":n,"modified":n},
 "manifest":{"changed_keys":["fields","splits"], "splits":{"train":{"added":[…],"removed":[…]}},
             "fields":{"renamed":[["pMean","pressure"]],"added":[],"removed":[]}},
 "samples":{"added":[…],"removed":[…],"modified":{"run_1":{"arrays":{"cell_fields/pressure":
     {"n_before":…,"n_after":…,"n_changed":…,"max_abs_diff":…,"rel_l2_diff":…,
      "before":{"min":…,"max":…,"mean":…,"std":…},"after":{…}}},
     "meta":{"changed_keys":["fields"]},"forces":{"cd":[0.31,0.30]}}}},
 "scalars":{"parameters":{"Re":{"changed":3}}}}
```

Array statistics use bounded ArrayRoute reads of both versions (hardlink checkout to a
temporary directory, or file copies if hardlinks are unavailable). Counts and extrema scan
the full arrays; only percentiles use deterministic sampled positions. Preparation previews
may restrict time steps and retain their exact selection rather than claiming full-trajectory
coverage. Different source units suppress physical delta metrics instead of subtracting
incompatible quantities. Float reductions use fixed-order blocks with a pinned NumPy/runtime.

## 6. Recipes

A recipe is JSON: `{"name", "version":"1.0", "description", "steps":[{"op","params",
"message"?, "continue_on_fail"?: false}], "created_by", "created_from": {"dataset", "versions":
[in, out]}}`. `recipe run <recipe> --dataset <ds>` executes the steps in order, each creating a
version; a step whose plan is `noop` is skipped and recorded as such. `recipe diff <recipe>
--dataset` plans every step without running. Recording: every `run` through the CLI or the
SDK can append to a recipe (`--record <file>`), which is how a cleaning session becomes a
pipeline.

## 7. CLI (`python -m frontier_prep`)

```
ops list [--json] | ops show <op> [--json] | ops export-tools --json [--per-mode] [-o file]
plan|preview|run <op> --dataset <ds> [--params '{json}' | --set k=v …] [--json]
run … --seed N --workers N --message … --actor name --approve name [--allow-dirty] [--record recipe.json]
recipe run <file> --dataset <ds> [--workers N] | recipe diff <file> --dataset <ds>
versions init|status|log|diff|revert|checkout|lineage|gc|tag <ds> …
```

Every command prints JSON with `--json`; exit codes: 0 ok, 1 error, 2 blocked (plan has a
`fail` warning or the gate fails).

## 8. Server / app contract (to implement; not part of this package)

Endpoints over `docs/spec/api.md`:

- `GET /datasets/{id}/versions` -> `log`; `GET …/versions/{v}` -> version record;
  `GET …/versions/{a}/diff/{b}` -> section 5; `POST …/versions/{v}/revert`;
  `POST …/versions/{v}/tags/{name}`.
- `GET …/provenance?kind=&since=` (stream of records), `POST …/provenance/approvals`
  (`{op_record|version, decision, note}`; the server signs).
- `GET /ops` (tool export), `POST /datasets/{id}/ops/{op}:plan|preview|run` with
  the tool input schema; `run` is a job; the result is the `Result`.
- `GET …/previews/{preview_id}/renders/{n}` serves a render.

App workbench page: dataset header with the version chain and tags; left: op catalogue by
category with `when_to_use`; centre: plan (changes table, warnings, judgments as choice chips),
preview (diffs as before/after stats, renders), approve-and-run; right: provenance timeline
(ops, assertions, approvals) with revert. Recipes are the saved sessions.

## 9. Determinism register

| op                                                             | class                                                        | reason                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prep.mesh.decimate`                                           | tolerance                                                    | quadric collapse order depends on the library's heap tie-breaking; recorded `rel_area` 1e-3 and `face_count_rel` 0.05                                                                                                                                                                                                                                                                                                                                 |
| `prep.mesh.resample`                                           | seeded                                                       | random / farthest-point selection from `ctx.seed` (per-sample seed = sha256(seed, sample id))                                                                                                                                                                                                                                                                                                                                                         |
| `prep.dataset.split`, `prep.dataset.coverage`                  | seeded                                                       | permutation / LHS candidates from `params.seed`                                                                                                                                                                                                                                                                                                                                                                                                       |
| `prep.quality.check`                                           | tolerance                                                    | assertions are bitwise; `quality/report.json` carries timings and a timestamp                                                                                                                                                                                                                                                                                                                                                                         |
| any op that changes level 0 of a sample with stored LOD levels | level 0: the op's class; rebuilt `lod/<n>` groups: tolerance | the engine rebuilds the stored levels from the new level 0 with the `prep.mesh.decimate` cascade (`rel_area` 1e-3, `face_count_rel` 0.05; identical run to run on one machine and library version); the Result and the provenance record carry `tolerance: {lod_rel_area, lod_face_count_rel}` and `lods_rebuilt`; samples without stored levels keep the op's class unchanged, and a rename is applied to the levels as a directory rename (bitwise) |
| everything else                                                | bitwise                                                      | sorted sample order, fixed-order float64 reductions, renames by directory rename                                                                                                                                                                                                                                                                                                                                                                      |

### 9.1 The gate fails closed

`prep.quality.gate` decides from the assertions that are _effective_ at the version it
evaluates: the latest record per (check, target) that was made on content identical to that
version's. An op that changes a sample (including an LOD rebuild) invalidates the assertions
made on it. A check the gate requires that has no effective record for a sample that is in
its scope counts as **not evaluated**, and the gate fails naming the check and the samples;
it never passes on the absence of evidence. In particular, when any version in the history
recorded a closure check (`force_closure`, `area_conservation`, `temporal_continuity`, ...)
for a sample and no effective record of that check exists at the evaluated version, the gate
fails with `not_evaluated` until the check is run again. `stats.gate` records `missing`,
`not_evaluated` (check -> samples), `failing_checks` and `closure_present`.

The checks the gate requires follow the dataset (`required`, default): `area_sum` for
surfaces, `force_consistency` and `force_closure` when samples carry forces,
`temporal_continuity` for transient data, `mesh_cells` for steady meshes, and
`predictable_fields`, `nan_inf` and `split_leakage` for every dataset. `stats.gate.required`
records the list used.

- `predictable_fields` fails for a sample without a real-valued field and for a transient
  sample none of whose real-valued fields changes over time (a trajectory at rest, or one
  whose only stored field is a label).
- `mesh_cells` (closure check of a steady mesh): every cell refers to existing points, no cell
  repeats a point and, for planar meshes, no cell has zero area; it fails above 1 % such cells.
- `freestream_consistency` (closure check, steady meshes with a velocity field and
  `conditions.u_inf`): the median speed of the 2 % of points farthest from the centre against
  `u_inf`; warn above 5 %, fail above 25 %.
- `field_outliers` can fail a sample only in datasets of 30 samples or more; below that the
  robust z-score describes the cluster of similar cases and the check warns.

`versions revert`: when the reverted HEAD's content equals that of the version carrying
`train-ready`, the tag follows to HEAD and a note names the source version; otherwise HEAD is
untagged and the CLI says which version holds the tag and how to re-run the gate.

## 10. First-party ops (v1.0.0)

| id                                | category                     | changes                                                 | assertions                               |
| --------------------------------- | ---------------------------- | ------------------------------------------------------- | ---------------------------------------- |
| `prep.fields.rename`              | fields                       | field names (no array rewrite)                          |                                          |
| `prep.fields.map`                 | fields                       | canonical names + units; judgments for ambiguous names  |                                          |
| `prep.fields.cast_location`       | fields                       | derived cell/point copies tagged `derived_from`         |                                          |
| `prep.fields.components`          | fields                       | reorder / sign / split / merge / magnitude              |                                          |
| `prep.fields.nan_fix`             | fields                       | fill / mask / drop steps                                | `nan_inf`                                |
| `prep.fields.outlier_detect`      | fields                       | `<field>_outlier` masks                                 | `field_outliers`, `unit_plausibility`    |
| `prep.fields.smooth`              | fields                       | Laplacian, justification recorded                       |                                          |
| `prep.fields.normalize`           | manifest                     | `stats.normalization` from the train split              |                                          |
| `prep.units.convert`              | fields, geometry             | pint factor, reference quantities, declared unit kept   |                                          |
| `prep.scalars.derive`             | forces.json, Cp/Cf fields    | surface integrals                                       | `force_closure`                          |
| `prep.mesh.watertight_check`      | meta                         | topology audit                                          | `watertight`                             |
| `prep.mesh.merge_seams`           | geometry                     | coincident vertices                                     | `area_conservation`                      |
| `prep.mesh.repair`                | geometry                     | degenerate/duplicate faces, orientation, small holes    | `area_conservation`                      |
| `prep.mesh.orient_outward`        | geometry                     | consistent outward normals                              |                                          |
| `prep.mesh.transform`             | geometry, vector fields      | scale / translate / axes with handedness check          |                                          |
| `prep.mesh.decimate`              | LOD groups                   | quadric cascade after seam merge                        | `lod_area_conservation`                  |
| `prep.mesh.resample`              | `static/resample_index`      | seeded point budget                                     |                                          |
| `prep.mesh.part_ids`              | `cell_fields/part_id`        | connectivity (+ sharp edges), inferred                  |                                          |
| `prep.time.subsample`             | time axis                    | stride / budget / target dt                             |                                          |
| `prep.time.trim`                  | time axis                    | MSER burn-in or explicit                                |                                          |
| `prep.time.align`                 | time axis                    | common t0 / length                                      |                                          |
| `prep.time.drop_duplicate_frames` | time axis                    | frame hashes                                            | `duplicate_frames`                       |
| `prep.time.window_average`        | `<field>_mean/_rms`          | documented window                                       |                                          |
| `prep.time.continuity_check`      | `signals/continuity_<field>` | jumps, frozen frames                                    | `temporal_continuity`, `times_monotonic` |
| `prep.dataset.dedup`              | manifest (+ sample removal)  | exact + invariants + Chamfer; "rerun?" stays a judgment | `duplicate_geometry`                     |
| `prep.dataset.split`              | manifest splits/families     | random / parameter hold-out / group                     | `split_leakage`                          |
| `prep.dataset.coverage`           | `stats.coverage`             | bins, NN gaps, maximin suggestions                      | `parameter_coverage`                     |
| `prep.dataset.balance`            | `stats.weights`              | regime weights                                          |                                          |
| `prep.dataset.select`             | manifest (+ sample removal)  | ids / predicates / recorded quality                     |                                          |
| `prep.dataset.merge`              | manifest, sample dirs        | schema reconciliation                                   |                                          |
| `prep.dataset.summarize`          | `stats.summary`, `CARD.md`   |                                                         |                                          |
| `prep.quality.check`              | `quality/report.*`           | every frontier-data check + frame hashes                | all checks, `initial_frame_duplicates`   |
| `prep.quality.gate`               | `stats.gate`, tag            | closure checks required                                 | `train_ready`                            |

## 11. Kernels: reference and native implementations

The hot array kernels exist twice: the numpy **reference** in `frontier_prep.ops` (functions
named `*_py`) and the **native** Rust ports in the crate `frontier-prep-kernels`, reached through
`frontier_native.prep`. Both implement the rules below and return the same bytes, so the choice
never changes a version id and no op's determinism class depends on it. Kernel spec revision:
**1** (`frontier_prep.kernels.SPEC_REVISION`, `frontier_native.prep.SPEC_REVISION`); a native
module with another revision is not used.

Selection (`frontier_prep.kernels`): environment variable `FRONTIER_PREP_KERNELS` =
`auto` (default: native when importable, else the reference) | `python` | `native` (an error
when the native module is missing). Every `op` provenance record carries `kernels`, and
`Result.machine` carries `prep_kernels` / `prep_kernels_version` (the Rust crate version, or the
`frontier-prep` version for the reference). A native kernel hands a single call back to the
reference for input it does not take: arrays that are not float32/float64 `(N, 3)` points or
integer `(M, 3)` faces, negative or out-of-range indices, and a tolerance grid whose keys do
not fit int64 (non-finite coordinates with `tol > 0`), where numpy's float-to-integer
conversion is platform-defined. The reference then decides the result or the error.

General rules: an output is a pure function of the input arrays; no hash map decides an
output; every sort has a total order (ties broken by the original index); threads are used
only for element-wise maps and for sorts with a total order, so results are identical for any
thread count (`RAYON_NUM_THREADS`; the engine gives each of `W` worker processes
`cores // W` kernel threads). Float arithmetic is plain IEEE-754 binary32/binary64 with
separate multiplications and additions (no fused multiply-add). `sum(x)` below means numpy's
pairwise summation of a contiguous array, started from `0.0`: a plain left-to-right loop
below 8 elements; up to 128 elements, eight running partial sums `r[j] += x[i + j]` combined
as `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))` and the remainder added left to right; above 128,
`sum(x[:h]) + sum(x[h:])` with `h = n // 2` rounded down to a multiple of 8.

**Edge tables** (`faces` int64 `(M, 3)`, indices `>= 0`). Slot `k` in `[0, 3M)` is face
`k % M`, edge slot `k // M` (0: `(f0, f1)`, 1: `(f1, f2)`, 2: `(f2, f0)`). Edges are the
distinct undirected pairs `(min, max)` in lexicographic order. `order`: the slots sorted by
(edge, slot index); `inv[k]`: edge id of slot `k`; `start[e]`, `counts[e]`: the run of edge
`e` in `order`; `uniq[e] = (min, max)`. All int64.

**Components.** Faces sharing a slot id are connected (any number of faces per id). Labels
are numbered by first appearance in face order (int32).

**Topology audit** (`points` `(N, 3)` float32 or float64). `open_edges = #(counts == 1)`,
`nonmanifold_edges = #(counts > 2)`; `inconsistent_edges`: per edge, `n_fwd` = slots whose
face traverses it from `min` to `max` (a slot with `u == v` counts as forward), `n_bwd =
counts - n_fwd`, result `#(n_fwd > 1) + #(n_bwd > 1)`; `euler = used - E + M` with `used` the
number of distinct vertex indices in `faces`; `genus_estimate = (2 * components - euler) / 2`
when no edge is open, else null; `watertight = open_edges == 0 and nonmanifold_edges == 0`;
`oriented = inconsistent_edges == 0`. Floats:

- per-face area `a_i = 0.5 * sqrt((c0*c0 + c1*c1) + c2*c2)` with `c = (p1 - p0) x (p2 - p0)`
  in float64 on the points **narrowed to float32** first, and `x` the cross product
  `(a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0)`; `area = sum(a)`;
  `degenerate_faces = #(two equal indices or a_i <= 0)`;
- `boundary_length = sum(|points[u] - points[v]|)` over open edges in edge order, **in the
  dtype of `points`** (float32 arithmetic and a float32 pairwise sum for float32 points), the
  squares added left to right, widened to float64 at the end; `0.0` without open edges;
- `bounds = [xmin, xmax, ymin, ymax, zmin, zmax]` over all points as float32: NaN if the
  column has a NaN, `-0.0` orders below `+0.0`;
- `signed_volume = sum(d) / 6` with `d_i = (p0[0]*x[0] + p0[1]*x[1]) + p0[2]*x[2]`,
  `x = p1 x p2`, in float64 on the points as given (not narrowed).

A mesh without faces yields `{n_points, n_faces: 0}` only.

**Coincident-vertex merge** (`tol`). Key of a vertex: its coordinate row when `tol <= 0`
(compared numerically: `-0.0 == 0.0`; a NaN sorts after every number and never equals
anything, so a vertex with a NaN coordinate is always its own group), else
`round_half_even(points / tol)` per axis as int64, **the division in the dtype of `points`**
(`tol` converted to that dtype). Groups are the distinct keys in lexicographic order; a
group's representative is its lowest vertex index, whose coordinates are kept verbatim.
Outputs: `points[representatives]` as float32, `point_map` (int64, group of every vertex),
the faces remapped, those with two equal corners dropped, order preserved, as int32.

**Orientation.** Two faces are neighbours iff they share an edge with exactly two slots. The
relation value of a pair is the sum over such shared edges of 2 (both traverse the edge in
the same direction) or 1; the pair disagrees iff the sum is 2. Faces are visited breadth-first
from the lowest unvisited face index, neighbours in ascending face index; the first face of
each walk keeps its winding and every other face gets `flip = flip[predecessor] XOR
disagree`. Flipped faces become `[f0, f2, f1]`; output int32 and the number of flips.
**Duplicate faces**: a face is a duplicate iff an earlier face has the same vertex set.

**Unit scaling.** `y = float32(float64(x) * factor)` for every element when `factor != 1`
(round-to-nearest-even on the narrowing); integer arrays and `factor == 1` stay with the
reference (`factor == 1` returns float inputs narrowed to float32 and other dtypes unchanged).
