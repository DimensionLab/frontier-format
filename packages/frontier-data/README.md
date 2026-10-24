# frontier-data

Reads, writes, validates and versions datasets in the Frontier dataset format: surface, mesh,
grid and point-cloud samples, steady or transient, one zarr v3 store per sample. Includes
quality checks, train/val/test splits and converters for public datasets.

## Install

Install the wheel from the release bundle:

```bash
pip install --no-index --find-links wheels frontier-data
```

For the latest public source integration, install the sibling libraries together
inside a Python 3.11–3.13 environment. Source releases may precede PyPI releases:

```bash
git clone https://github.com/DimensionLab/arrayroute.git
git clone https://github.com/DimensionLab/frontier-format.git
python -m pip install './arrayroute/packages/arrayroute[tensorstore,zarrista,disk-cache]' \
  ./frontier-format/packages/frontier-plugins \
  './frontier-format/packages/frontier-data[readers]'
```

Pin the repository commits when reproducing an experiment. These packages do not
require the private Frontier application or an account.

The Well converter reads HDF5 and needs the extra `frontier-data[ingest]`.
`frontier-data[readers]` installs the TensorStore and Zarrista reader engines;
`frontier-data[server]` installs the SDK and HTTP client for server-backed board runs
and remote execution. Local dataset inspection and Zarr-Python reads need neither extra.

## Example

```bash
python -m frontier_data synth my-dataset --kind grid   # write a small synthetic dataset
python -m frontier_data validate my-dataset            # check it against the format
python -m frontier_data quality my-dataset             # write quality/report.json and report.html
python -m frontier_data inspect my-dataset
```

```python
import numpy as np
from frontier_data import Dataset, DatasetWriter, validate_dataset

w = DatasetWriter("my-dataset", name="my-dataset", domain="cfd", license="CC-BY-4.0")
times = 0.01 * np.arange(50)
w.write_grid_sample(
    "case_0",
    fields={"vorticity": np.zeros((50, 64, 64))},  # time first: (T, X, Y)
    times=times,
    spacing=[0.1, 0.1],
    parameters={"Re": 200.0},
)
w.finalize()  # writes manifest.json

s = Dataset("my-dataset").sample("case_0")  # lazy: nothing is read yet
u = s.field("vorticity", t=slice(10, 14))  # reads four steps
assert validate_dataset("my-dataset").ok
```

## Grid writes and preparation

Grid fields use ArrayRoute's bounded spatial and temporal layout. Small complete frames
share chunks of up to 512 KiB and 4096 time steps by default; large volumes retain one
step per chunk with spatial chunks and shards. Pass `steps_per_chunk=1` when independent
frame objects are required, or supply `storage_layout=GridLayoutPolicy(...)` to tune
the limits. Each array records the policy version and effective byte limits alongside
its actual Zarr layout. Existing datasets are not silently rewritten.

Grid field rewrites and unit conversion in Frontier Prep use the same writer. Conversion
reads through ArrayRoute and writes one bounded destination shard at a time. It preserves
the numerical conversion order, physical times and declared units. Tests compare complete
stored arrays, including metadata and compressed bytes, against the whole-array conversion.
Temporary writes finish before replacing an existing field, so a source-read failure
preserves it; this is not a crash-atomic multi-file transaction. The writer buffer does
not include source-reader caches, numerical conversion scratch or caller-retained arrays.

## Read sessions and memory limits

Sample fields, signals, times and geometry use ArrayRoute. `read_surface(...)` and
`Dataset.surface(...)` also accept `session=...`; their allowance applies per array, while
the returned mesh retains all selected arrays. The default local session uses
Zarr-Python with a 128 MiB read-buffer allowance. Choose an explicit allowance or optional
reader engine when needed:

```python
from arrayroute.session import LocalReadSession

session = LocalReadSession(engine="zarr-python", buffer_bytes=512 << 20)
sample = Dataset("my-dataset").sample("case_0", session=session)
frames = sample.field("vorticity", t=[13, 10, 13])  # requested order and duplicates preserved
```

The allowance covers planned read buffers and gathered/generated outputs, not total process
memory, native codec scratch or arrays retained by callers. Oversized selections fail before
payload reads; select fewer steps, use `FieldReader` streaming, or choose a larger allowance.
Sessions are worker-local and require fixed metadata. Use multiprocessing spawn for native
engines. After changing a sample, call `refresh()` for its internally owned session, or
`refresh(session=fresh_session)` when you supplied one. Immutable TensorStore groups require
a fresh group too. These sample sessions support local and mounted filesystems.

## Commands

Run `python -m frontier_data <command> --help` for the flags.

| Command                        | What it does                                                                                                      |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `ingest ahmedml`               | Download AhmedML runs and convert them to a surface dataset                                                       |
| `ingest the-well <name>`       | Download part of a The Well dataset and convert it to transient grids                                             |
| `ingest meshgraphnets <name>`  | Download part of a MeshGraphNets dataset and convert it to transient meshes                                       |
| `inspect <dataset>`            | Read catalog metadata and a bounded sample page; storage walks are opt-in                                                         |
| `validate <dataset>`           | Check a dataset against the format; exits non-zero on errors                                                      |
| `quality <dataset>`            | Run quality checks and write `quality/report.json` and `report.html`                                              |
| `synth <out>`                  | Write a small synthetic transient dataset                                                                         |
| `mesh-lods <dataset>`          | Add coarser LOD levels to mesh samples                                                                            |
| `pipeline <pipeline.json>`     | Run an ingest pipeline; `nodes` lists the node kinds                                                              |
| `versions <command> <dataset>` | `init`, `log`, `status`, `commit`, `diff`, `revert`, `checkout`, `lineage`, `tag`, `approve`, `gc`, `export-prov` |

## API

| Module                      | Contents                                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `frontier_data`             | `Dataset`, `DatasetWriter`, `Sample`, `validate_dataset`, `write_surface_sample`, `write_mesh_sample`, `write_grid_sample`, `write_points_sample` |
| `frontier_data.quality`     | `run_quality`: per-sample and dataset checks with status `pass`, `warn`, `fail` or `skip`                                                         |
| `frontier_data.splits`      | `random_split`, `parameter_holdout_split`, `group_split`, `find_leakage`                                                                          |
| `frontier_data.versions`    | `VersionStore`: immutable versions, provenance log, diff, revert, W3C PROV-JSON export                                                            |
| `frontier_data.ingest`      | `ingest_the_well`, `convert_the_well`, `ingest_meshgraphnets`, `convert_meshgraphnets`                                                            |
| `frontier_data.pipeline`    | `Pipeline`, `PipelineRunner`, `register` for JSON-serialisable ingest pipelines                                                                   |
| `frontier_data.predictions` | `write_transient_prediction`, `read_transient_prediction`, `pack_prediction_store`                                                                |
| `frontier_data.prediction_view` | `PredictionView`: selected prediction/reference/error reads and budgeted tile streams through ArrayRoute |
| `frontier_data.grid_export` | Selected-frame ParaView export, with provenance and unit checks |

## Explore and export long prediction rollouts

Use the chunked prediction store for interactive reads. `read_transient_prediction` still
materializes the NPZ rollout; it is not the API for browsing large results.

```python
from arrayroute.session import LocalReadSession
from frontier_data.prediction_view import PredictionView

view = PredictionView("./predictions", "case-1",
                      session=LocalReadSession(buffer_bytes=128 << 20))
print(view.field_names, view.n_steps, view.time_units)
frame = view.read("velocity", "pred", selection=view.frame_selection("velocity", 12))
print(view.time(12), view.units("velocity"))
for tile, values in view.stream("velocity", "true"):
    # Process this tile before requesting another; do not retain all tiles in RAM.
    print(tile.destination, values.shape)
```

LocalReadSession selects Zarr-Python, TensorStore or Zarrista. Values and ordering are
unchanged. The source must remain immutable throughout the session. Read allowances cover
outputs and planned decoded tiles; caches, codec scratch, Python metadata and values retained
by the caller are additional. `view.stats` exposes measured read counts and planned bytes.
`stream` checks cancellation between tiles; `read` checks before and after a selection.
Neither can interrupt a blocking storage call.

```sh
frontier-prediction-export --predictions ./predictions --dataset ./dataset \
  --sample case-1 --fields velocity pressure --start 0 --stop 100 --stride 2 \
  --buffer-mib 256 --out ./paraview
```

Open the resulting `flow.pvd` in ParaView. Frame indices retain their original absolute
simulation times in seconds; coordinates and per-field units are included. This command
requires a 3D uniform grid, matching dataset-manifest hash, coordinates and physical units.
It reads selected fields one frame at a time. Per-frame admission includes NumPy result and
workspace arrays; VTK's native allocations are additional, so the allowance is not an RSS cap.
The completed directory appears atomically and never replaces an existing destination.
Handled failures and cancellation remove staging; power-loss recovery is not provided.

If a prediction declares validity masks, those masks must be chunked `fields/<name>_valid`
arrays matching the field shape and containing only zero/one. NPZ-only masks produce an
actionable error; they are never ignored. Invalid cells export as NaN alongside explicit
validity fields. `write_transient_prediction(..., validity=masks)` and the training prediction
path now record both the chunked masks and evaluator sidecar. Older NPZ-only masked results
must be regenerated or explicitly converted before using this exporter.
The comparison-dataset and service-response helpers retain their current eager behavior;
this change does not make those independent consumers streaming.

### Score chunked predictions

```python
curves = view.error_curves("velocity", buffer_bytes=128 << 20)
print(curves["rmse"], view.units("velocity"))
```

This uses ArrayRoute's canonical paired error curves directly from the chunked store;
it does not read the NPZ or materialize the whole rollout. Declared validity masks are
required. It returns per-step RMSE in field units, dimensionless relative errors and
correlation, and mean-square values in squared field units. Fixed 64-frame tiles bound
payload memory independently of rollout length; output/state still grows with timestep
count. The second pass reads references and masks for retrospective climatology. The
numerical allowance includes workspace and windows, but excludes backend caches and codec scratch.
Source values, shape and numerical stack must stay fixed for byte-identical replay;
record `arrayroute.error_curves.ALGORITHM` with results. This method does not perform
spectral, conservation or uncertainty diagnostics and does not replace a full benchmark.

## Inspect before reading payloads

```sh
python -m frontier_data inspect ./dataset --limit 20
python -m frontier_data inspect ./dataset --offset 20 --limit 20 --json
python -m frontier_data inspect ./dataset --sample case-1 --json
python -m frontier_data inspect ./dataset --measure-storage
```

Ordinary inspection reads the manifest and, only when requested, one sample metadata
file. It does not walk payload directories or open a quality report. `--measure-storage`
explicitly inventories all files and can be slow on large or remotely mounted stores.
The default page contains 20 samples (maximum 1,000); JSON output includes `next_offset`
and distinguishes an unmeasured storage size from zero bytes. Field units and association
come from the catalog. The JSON manifest itself is still loaded in full: pagination here
bounds output, not catalog memory or remote catalog indexing.

## Bounded field queries

To plan concurrent reads without loading field values, provide a reader ceiling and a
total read-buffer allowance. The same options work with `query-snapshot`:

```bash
python -m frontier_data query-field ./dataset case-1 velocity --select time=10 \
  --plan-workers 8 --total-read-budget-mib 256
```

The JSON report includes the field's units, selected dimensions and candidate reader
counts. Each candidate reports whether the region fits directly, requires tiled reads,
needs streaming, or has a chunk too large for its buffer. A bounded tile count may be
reported as a lower bound. This plans one selected region per concurrent reader; queued
batches, cached data and model memory are additional. Benchmark viable candidates on
your own storage before selecting a training configuration. These planning options do
not change the separate `--budget-mib` used when exporting a field.

Measure the viable settings on an unchanged dataset:

```bash
python -m frontier_data benchmark-field ./dataset case-1 velocity --vary time \
  --count 8 --select x=0:64 --select y=0:64 --select z=0:64 \
  --workers 1 --workers 2 --workers 4 --engine tensorstore \
  --total-read-budget-mib 256 --max-output-mib 1024 --time-budget-s 60 > read-pilot.json
```

This samples evenly spaced **indices** along the chosen dimension, keeping the other
selected regions fixed. Every read must match a sequential reference. The report retains
field units, selection identities, hashes, individual shuffled trials and the fastest
observed worker count. It does not automatically alter training settings. The decoded
output cap includes the reference and all repetitions; it is checked before payload reads.
The time budget stops new reads but must wait for an in-flight native read. For a hard
deadline, run the command in a process controlled by your scheduler.

Timing includes opening worker readers, reading, hashing and thread teardown. Repeated
regions may fit in OS caches, so this is not cold-storage bandwidth or training throughput.
Native caches, decoder scratch and small hash buffers are outside the read-buffer budget.
Install `frontier-data[readers]` for TensorStore or Zarrista; the same command works with
the default Zarr-Python engine. Keep the dataset fixed during a pilot; changed metadata or
values cause an error rather than a recommendation. A timed-out pilot returns partial
evidence with no recommended setting.

Inspect a region before reading it, then optionally export it:

```sh
python -m frontier_data query-field ./dataset case-1 pressure --select time=10 --select x=20:80
python -m frontier_data query-field ./dataset case-1 pressure --select time=10:100 \
  --budget-mib 128 --max-output-mib 8192 --out pressure-window.npy > pressure-window.json
```

The JSON report records source shape, dtype, chunk shape, normalized index selections,
selected dimensions, units, node/cell association, source/LOD identity and read costs.
Selectors use array indices, not physical coordinates or seconds. Negative indices and
omitted slice bounds are normalized in the report. `--location point|cell|grid|global`
selects a declared scientific association explicitly; `--lod` chooses an available stored LOD (0 is
finest). A missing field/location/LOD is an error, not silently substituted data. A name declared
at multiple locations requires an explicit choice. Fields without catalog metadata must
be described before querying; `-` is the explicit unknown-unit value. Global signals use
their declared units and the `signals/` array, independent of mesh fields.
NPY retains numerical bytes, dtype and shape; keep the JSON report for physical metadata.
Exports stream through the chosen buffer budget and publish atomically without replacing
existing files. `--max-output-mib` limits output size independently (default 1024 MiB).
Local/mounted filesystems must support hard links. Backend caches, decoder scratch and OS
file caches are additional memory. `--engine` accepts `zarr-python`, `tensorstore` or `zarrista`;
install the corresponding optional reader before choosing it.

Python users can call `frontier_data.query.FieldReader.describe`, `.plan`, `.read` or `.stream` with
named dimensions. Every streamed `FieldResult` retains units and association, while its
`Tile` provides the source selection and destination region. The stream accepts a
cancellation callback; callers retaining yielded arrays must budget that memory themselves.
`describe()` returns metadata and costs without reading values. For read/compute overlap,
use the context-managed `prefetch()`; it splits the explicit read budget into two slots
and preserves the same units, axes and snapshot identity:

```python
from frontier_data import Dataset
from frontier_data.query import FieldReader

pressure = FieldReader(Dataset("dataset").sample("case-1"), "pressure", buffer_bytes=64 << 20)
with pressure.prefetch({"time": slice(10, 100)}) as blocks:
    for tile, result in blocks:
        process(tile.destination, result.values, result.units)
        del result
```

One background worker owns reads. Cancellation callbacks must be thread-safe, and context
exit waits for in-flight native I/O to finish. Additional arrays retained by consumers,
backend caches and decoder scratch are outside the explicit read budget.

Connectors can expose remotely stored arrays through the same scientific query API:

```python
from arrayroute.remote import RemoteReadSession, S3Store
from frontier_data.query import ArrayFieldReader

storage = RemoteReadSession(S3Store("engineering-data", prefix="snapshots/revision-42"))
pressure = ArrayFieldReader(
    storage.open("case-17/pressure"),
    dimensions=("time", "z", "y", "x"), units="Pa", association="cell",
)
section = pressure.read({"time": 10, "x": 40})
```

The dimensions, units and association must come from the connector's scientific catalog;
they are not inferred from numerical values. `ArrayFieldReader` works with any ArrayRoute
backend and preserves these declarations through reads, streaming and field comparisons.
Remote endpoints must be authorized by the caller and use immutable objects. This is a
Python integration point; it does not register a remote dataset in the native app or add
remote URL support to the local-dataset CLI commands.

Compare a reconstructed dataset with its explicitly paired reference:

```sh
python -m frontier_data compare-field ./original ./reconstructed case-1 pressure \
  --tolerance 0.1 --budget-mib 128
```

The absolute tolerance is in the stored field units (for example 0.1 Pa). The command
streams both layouts and reports maximum absolute error, RMSE, relative L2, nonfinite
mismatches and pass/fail; a failed tolerance exits with status 1. Units, axes, association,
shape and dtype must agree. Empty or all-nonfinite regions cannot pass as scientific
evidence. This is an elementwise comparison of fields in the same geometry/index space;
it does not establish mesh correspondence or remap different meshes. Two reader budgets
and comparison scratch are additional memory. Use `--select` for a region and
`--candidate-sample` when paired sample identifiers differ.

## Notices

- Licence: Apache-2.0. Converted datasets keep the licence of their source: AhmedML is
  CC-BY-SA-4.0, The Well is CC-BY-4.0.
- Downloads need network access and are refused when `network.offline` is set; `--local`
  converts files that are already in `--raw-dir`.
- The AhmedML converter fetches surface files only and refuses the volume files (5.6 GB per run).
- A versioned dataset stores files as read-only hard links, so change it through writers or
  ops, not by editing files in place.
- Dataset references: AhmedML, Ashton et al., arXiv:2407.20801 (2024); The Well, Ohana et al.,
  NeurIPS 2024, arXiv:2412.00568; MeshGraphNets, Pfaff et al., ICLR 2021, arXiv:2010.03409.

## Documentation

- [Dataset format specification](../../docs/spec/dataset-format.md)

### Read a committed scientific field

With Python 3.12+ and the `snapshots` extra, an Icechunk field can be selected by an immutable
revision. Declare the field's `dims`, `units` and `association` in its array attributes.

```sh
frontier-data query-snapshot simulation.icechunk velocity --branch main --select time=10
frontier-data query-snapshot simulation.icechunk velocity --snapshot-id SAVED_ID --select time=10 --out velocity.npy
```

The result records the resolved snapshot ID, physical units, selected dimensions and read
costs. Save that JSON alongside any exported NPY; NPY itself does not store scientific
units or provenance. `--branch` resolves once per invocation; use the returned ID to repeat
exactly the same revision later. Exports never replace existing files. The Python
`SnapshotFieldReader` also attaches the revision ID to each returned field/tile. Repository
retention remains the owner's responsibility. This command does not yet register a remote
snapshot as a canvas dataset or replace the training catalog.

Inspection latency can be measured on an existing unchanged dataset with
`python -m frontier_data.benchmark_inspection --dataset ./dataset --output ./inspection-results`.
The output directory must be outside the dataset. The benchmark compares fresh CLI processes
with/without explicit storage measurement, verifies identical catalog output, retains manifest
and source hashes, and includes process startup in its timings. Record concurrent work and
storage/cache conditions when reporting results. It does not measure simulation read throughput.

### Comparing bulk shard reads

For sharded Zarr fields with many small inner chunks, compare bulk shard reads with
partial reads on the field you intend to train or visualize:

```sh
frontier-data benchmark-field ./dataset case-1 velocity --engine zarrista \
  --read-strategy bulk --gather-workers 8 --vary time --count 8 \
  --select x=::8 --select y=::8 --select z=::8 --workers 1 \
  --total-read-budget-mib 8
```

Repeat with `--read-strategy parallel`, or `--engine tensorstore --read-strategy native`
without the inner-worker option. Compare the same selections and allowances. Bulk
reads are optional; they can help small shards but hurt large ones. The report shows
the actual strategies and admitted inner workers. Skipped chunks and tight budgets
select partial decoding. `query-field` accepts the same policy and preserves units
in its JSON receipt alongside exact NPY output. This does not rewrite the dataset.

### Indexed metadata catalogs

For large manifests, create a portable metadata snapshot once and browse it without
reopening all sample records or accessing the numerical data:

```sh
python -m frontier_data catalog-build ./dataset ./dataset-catalog.sqlite
python -m frontier_data catalog-inspect ./dataset-catalog.sqlite --limit 20
python -m frontier_data catalog-inspect ./dataset-catalog.sqlite --start 20 --limit 20
python -m frontier_data catalog-inspect ./dataset-catalog.sqlite --sample case-1
python -m frontier_data catalog-inspect ./dataset-catalog.sqlite --verify
```

The JSON output includes dataset identity, license/provenance metadata, sample metadata,
field names, SI units as declared by the source, field locations, split membership and
revision identity. Same-named point and cell fields remain distinct. Sample paths remain
relative; creating a catalog does not copy or relocate simulation arrays. It also does not
infer missing units or convert unit systems. Browsing works with the numerical storage
disconnected, so the catalog can be shared separately from the much larger dataset.

Build from a dataset that is not being modified. Conversion reads the existing manifest
and split membership once into memory, then streams sample metadata; per-file hashes
identify what was captured. It does not provide a transaction across a mutable source
dataset. The destination must be new. `--verify` is an explicit full integrity scan;
ordinary browsing checks only returned records. Authenticate catalog receipts separately
when transferring files between trust boundaries.

This additional catalog format uses ArrayRoute's standard SQLite snapshot. The existing
Frontier v1 dataset remains authoritative; training, native canvas and server listings do
not automatically switch to indexed access. Distributed catalog publication, remote range
reads and those integrations require separate work and are not implied by these commands.

### Tune a field, save the settings, reuse them

Use `tune-field` to compare supported installed readers on representative regions of
your own dataset. Dimension names and SI units come from the field's metadata.

```bash
frontier-data tune-field ./dataset case-1 velocity --vary time --count 8 \
  --select 'x=::8' --select 'y=::8' --select 'z=::8' \
  --workers 1 --workers 2 --repeats 3 --total-read-budget-mib 64 \
  --max-output-mib 1024 --time-budget-s 60 --out velocity-read-profile.json

frontier-data query-field ./dataset case-1 velocity --profile velocity-read-profile.json \
  --select time=17 --select 'x=::8' --select 'y=::8' --select 'z=::8' \
  --out velocity-17.npy
```

The pilot compares Zarr-Python, four Zarrista strategies (native, dense, parallel,
bulk), and TensorStore. `--engine` restricts the set and may be repeated. Missing
optional readers and explicitly unsupported layouts are reported as skipped. All
measured values must exactly match the portable reader's sequential reference.
The pilot never changes source values or layout.

A profile is bounded JSON containing plain settings, measurements and a metadata
binding. It contains no executable factories. Reuse requires a completed pilot,
the same field, local dataset path, metadata and region pattern, and matching
recorded runtime versions, CPU count and thread environment. Another integer index
on the varied dimension is allowed. A changed selection or metadata requires retuning.
These checks do not identify the physical machine or storage device; retune after
moving storage or changing hardware, even when these coarse signatures still match.

`query-field` applies the measured per-worker reader settings to one requested region;
it does not start the pilot's outer worker pool. The receipt records the measured
worker count and applied allowance. A batch-optimal policy is not necessarily the
fastest single-region export. Pilot timings include setup, materialization and hashing,
not NPY writing, transforms or model training. Compare those complete workflows
separately when they determine the outcome.

Reports cannot overwrite existing files. A time-limited pilot may save diagnostic
JSON but provides no reusable recommendation. The deadline is cooperative and joins
in-flight native reads. The output cap counts selected bytes across all trials, not
storage traffic. The buffer allowance excludes native caches and codec scratch and
is not an RSS limit. Profiles are advisory local performance records, not authenticated
data snapshots or an integrity guarantee for future values.

### Repeated remote field reads with a local cache

Install the `readers` extra for optional reader engines and disk caching. Choose
`admission="retain"` for a fixed dataset scanned repeatedly when it exceeds cache
capacity; use `"lru"` when the active region changes. Measure uncached reads as well.

```python
from arrayroute.disk_cache import ArrayIdentity, TileCache
from arrayroute.remote import HttpStore, RemoteReadSession
from frontier_data.query import ArrayFieldReader

cache = TileCache("/local-ssd/my-user/study-revision-tiles",
                  capacity_bytes=8 << 30, max_entry_bytes=32 << 20,
                  admission="retain")
session = RemoteReadSession(
    HttpStore("https://my-storage.example/study/immutable-revision"),
    tile_cache=cache,
    cache_identity=ArrayIdentity("study", "immutable-revision", "velocity-SI-v1"),
)
try:
    field = ArrayFieldReader(session.open("velocity"),
                            ("time", "x", "y", "z", "component"), "m/s", "grid")
    values = field.read({"time": 17, "x": slice(0, 64),
                         "y": slice(0, 64), "z": slice(0, 64)})
    print(values.units, values.dimensions)
finally:
    session.clear()
    cache.close()
```

The connector must supply the real dimensions, units, immutable revision and array
path from its scientific metadata. The example does not infer them. Cache storage
must be authorized node-local disk, not NFS or a shared cloud filesystem. No saved
account credentials enter cache identities. Retained tiles survive process restarts;
use a fresh cache directory after a revision or working-set change. Source datasets
are never modified by cache admission.

To compare cache choices from your actual inspection or training access pattern,
add an optional access trace while the field's source and session are open:

```python
from arrayroute.cache_policy import TraceRecorder, compare_cache_policies

capture = TraceRecorder(max_events=10_000)
observed = field.with_access_trace(
    capture, identity="study@immutable-revision:velocity:physical-m/s-v1",
)
# Run a short representative serial workload using observed.read(...).
# Named dimensions, SI units, snapshot identity and shared buffer limits are preserved.
trace = capture.snapshot()
trace.save("study-access.json")
report = compare_cache_policies(
    trace, capacities_bytes=(120 << 20, 240 << 20), max_entry_bytes=32 << 20,
    training_reads=len(trace.accesses) // 2, entry_overhead_bytes=40,
)
```

Choose a pilot with repeated reads and a later validation segment. The report selects
no cache, LRU or retain on the prefix and checks it against LRU on the held-out suffix.
It does not apply settings. Sizes are decoded byte estimates with a usable payload
capacity, not compressed network traffic or a disk quota; reserve database/filesystem
space separately and confirm latency with actual reads. Truncated or failed traces
are rejected; overlapping calls receive no recommendation. See ArrayRoute's cache
policy documentation for capture limits and the standalone comparison command.

### Comparing large dataset revisions

Version differences now scan arrays through ArrayRoute rather than loading both revisions
into memory. Change counts and error metrics cover every element; percentile estimates have
an explicit sample count and scope in JSON output. Select the reader and allowance with:

```bash
python -m frontier_data versions diff my-dataset --a HEAD~1 --b HEAD \
  --read-engine zarr-python --read-buffer-mib 64 --percentile-samples 65536 --json
```

`VersionStore.diff()` exposes `read_engine`, `read_buffer_bytes` and `percentile_samples`.
The old `max_elements` argument has been replaced by `percentile_samples`, clarifying that
it never limits exact change counts. Version materialization still visits the referenced
files, so this is a bounded numerical scan, not a constant-time metadata comparison.

## Streaming grid ingestion

`DatasetWriter.write_grid_sample` accepts NumPy arrays or deferred
`arrayroute.transfer.BlockArray` sources for fields and static arrays. Fields retain the
float32 storage contract. Static types keep the existing floating/bool/integer policy;
`static_dtypes` can explicitly select different stored types. Names, units, coordinates,
physical times and boundary conditions use the same sample schema.

Grid writes transfer one storage shard at a time. `write_buffer_bytes` bounds the source
block plus conversion buffer and constrains the output layout when necessary. Reader
scratch, codec buffers, coordinates, global signals and OS caches are additional; it is not
a whole-process memory ceiling. Call inside a dataset publication transaction so cancelled
or failed output never appears as a completed dataset.

The Well direct and Hub converters now consume deferred fields, including static masks and
broadcast tensor fields. They no longer require a whole trajectory to fit in memory.
`--materialization-mib` on Well ingestion now sets the per-block transfer allowance;
`--read-buffer-mib` separately limits ArrayRoute reads. The low-level `read_well_file` API
still materializes by default for callers requesting arrays; `deferred=True` returns borrowed
sources that must be consumed before its iterator closes. Tensor components must fit together
within a block. Optional quality/preparation stages have their own resource behavior.

## Reader policies for quality checks

Quality checks accept an explicit ArrayRoute read session. The same session reaches steady
surface fields/geometry, transient scans, static arrays, signals and spectral crops.
Per-sample `read_policy` records the engine, read allowance, native threads, cache allowance
and immutable-input assertion. A report merged from precomputed workers retains each
worker's policy; it does not relabel those reads with the merger's settings.

```sh
python -m frontier_data quality my-dataset --engine tensorstore \
  --read-buffer-mib 128 --cache-mib 128 --native-threads 8
```

Install the `frontier-data[readers]` extra for alternative engines. Zarr-Python remains the
explicit default; TensorStore and Zarrista are opt-in. Invalid engine/cache combinations
fail instead of silently switching readers. In Python, pass `session=...` to `run_quality`
or `compute_sample_stats`; create it with `ReadSettings(...).session(buffer_bytes)`.

The cache is a TensorStore soft limit in addition to read buffers. Numerical reductions,
FFT workspaces, report/probe storage, retained geometry and native codec memory are also
additional. Spectral crops must fit the selected read allowance; larger crops require
an explicitly larger allowance. This is not a process memory limit.
Each preparation worker has its own session and allowances.

Freshness checks stay enabled by default. `--immutable-inputs` asserts that array values and
metadata will remain fixed for the session and allows TensorStore to omit data freshness
checks. It does not create a snapshot or lock a writer. Use that option only with an owned
immutable revision, and create a fresh session after changing inputs. Choosing a cache size
or engine does not change the canonical numerical reduction partition or physical units.

Transient field scans overlap exact frame hashing with their numerical checks through
ArrayRoute's `DigestPipeline`. Each scan adds one hashing worker and a copied payload bounded
by its I/O batch (up to 16 MiB for the numeric types supported by the scanner). Hash algorithms,
byte order, probe selection and the versioned numerical reduction order remain unchanged.
Preparation workers each own their hashing pipeline; account for these workers and buffers
when setting outer parallelism. A scan returns only after hashing succeeds, and closes its
worker if reading or numerical processing fails.

Non-transient fields on grid/mesh/point-cloud samples, and reported signal-field statistics,
now use exact ArrayRoute finite statistics with 32 MiB of numerical workspace by default. They can
exceed the read allowance without full-array materialization. Static non-finite and geometry
point checks use bounded count scans. Each field summary records its algorithm, float64
conversion, exact percentile scope, pass count and selected-byte SHA-256 in the
`nan_inf_per_step` check's `field_statistics` detail. Changed bytes between passes cause a
failure. Mean/std reduction semantics are versioned by `prep.quality.check` 1.3.0; percentiles
retain full-selection linear semantics. Original data and physical units are not rewritten.

These changes do not bound every quality stage: steady-surface mesh statistics,
mesh connectivity and FFT workspace still
have separate materialization requirements. Reader/native cache allowances remain additional
to numerical workspace. Exact bounded percentiles may cost additional reads; choose and
measure the reader/cache policy for the actual immutable dataset revision.

Set `statistics_workspace_bytes` on `run_quality` / `compute_sample_stats`,
`--statistics-workspace-mib` on `frontier-data quality`, or `statistics_workspace_mib`
on `prep.quality.check` to change that allowance (minimum 4 MiB). It applies per worker,
in addition to read buffers and native caches. Increasing it can reduce the number of
passes without changing the scientific results in the same runtime. Reports retain the
workspace and pass count actually used by each worker, even when merged elsewhere.

Static and steady velocity CFL now reads magnitudes lazily through ArrayRoute and calculates
the full-selection speed p99 within that statistics allowance. An additional 8 MiB transform
workspace and 8 MiB derived-reader allowance apply. The report records magnitude/statistics
algorithms, pass count and a hash of the magnitude bytes (not the original vector bytes).
Float64 sources use channel-order `hypot` to avoid squaring overflow/underflow; smaller
numeric source types use float64 square/sum/square-root, which is safe for their finite range.
The chosen reduction is recorded. Float64 results can differ in last bits. This change
is versioned by `prep.quality.check` 1.4.0. Time-dependent velocity CFL continues to use
probes for p99 and full scanned-frame maxima; the two percentile scopes remain distinct.

Signal-history quality now fits ArrayRoute channel statistics in bounded blocks and scans a
lazy standardized view. `prep.quality.check` 1.5.0 records the versioned finite-only policy,
channel order, fitted coefficients, source-byte fingerprints and workspace in
`signal_normalization`. The statistics workspace applies per source; the derived reader
adds a 16 MiB allowance. Time axes and retained diagnostics still grow with history length.
Original physical units remain unchanged; normalization is dimensionless and used only for
quality comparisons. This per-sample diagnostic fit is not a training transform: training
pipelines must fit only training data and reuse the recorded parameters for held-out data.

Single-entity temporal scans batch selected rows through ArrayRoute's strided row reader.
Regularly spaced samples share reads without adding unrequested rows to the logical
selection. Duplicates and requested order are preserved. Batches use at most half the
reader allowance for output; planned reads use the remainder. This changes read grouping,
not canonical reduction order, per-frame hashes or diagnostic values. Backend caches and
retained report arrays remain separate from those allowances.

## Reproducible prediction archives

`pack_prediction_store(directory, sample_id)` now uses ArrayRoute's deterministic archive
writer. The resulting `<sample_id>.zarr.tar` preserves every encoded store byte and remains
readable by standard tar readers. File timestamps, permissions and ownership do not affect
archive bytes. The store must remain immutable while packing; symlinks and special files
are rejected. Invalid sample path components are rejected before accessing the store.

Packing atomically replaces the previous archive only on success. A read failure,
resource-limit failure or cancellation preserves that previous archive. Optional arguments
are `max_archive_bytes` (default 1 TiB), `buffer_bytes` (default 1 MiB, covering copy/output
buffers), and a `cancelled` callback. Python metadata and OS caches are additional memory.
This does not make the earlier NPZ/JSON/Zarr prediction-generation steps transactional,
nor does it make NPZ prediction reads lazy. Those remain separate integration work.
