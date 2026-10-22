# frontier-data

Reads, writes, validates and versions datasets in the Frontier dataset format: surface, mesh,
grid and point-cloud samples, steady or transient, one zarr v3 store per sample. Includes
quality checks, train/val/test splits and converters for public datasets.

## Install

Install the wheel from the release bundle:

```bash
pip install --no-index --find-links wheels frontier-data
```

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
