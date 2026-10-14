<!-- SPDX-License-Identifier: Apache-2.0 -->
# Frontier dataset format

Normative description of `frontier-dataset` v1 and the v1.1 additions. The reference
implementation is `frontier_data` (Python) with a second reader in Rust.

## Version 1: steady surfaces (`frontier-dataset v1`)

A dataset is a directory (local, S3, or PFS) with:

```
<dataset>/
  manifest.json              # {"format":"frontier-dataset","version":1,"name":..,"source":{..},"samples":[{"id":"run_1","path":"samples/run_1", ...}],"splits":{"train":[..],"val":[..],"test":[..]},"fields":[...],"conditions":[...]}
  samples/<id>/
    meta.json                # v_CFD vector: solver, solver_version, turbulence_model, fidelity, mesh stats, bcs, geometry params, conditions, provenance (source URL, sha256 of raw files)
    surface.zarr/            # zarr v3 group: points (N,3) f32, faces (M,3) i32 (triangulated), cell_area (M,) f32, fields/<name> (N or M, C) f32, lod/<level>/... (decimated copies)
    forces.json              # {"cd":..,"cl":..,"cs":..}
    geometry.stl (optional)
  quality/
    report.json              # per-sample checks + dataset summary (see Quality)
    report.html
```

Sample ids are stable strings. Fields are named as in the solver (`p`, `wallShearStress`),
with units in meta. Points/faces are float32/int32. LOD levels: 0 = full, 1 = ~100k faces,
2 = ~20k faces, 3 = ~5k faces (quadric decimation or vertex clustering). Every sample stores
at least levels 0 and 2 so a viewer can show 100 cases at once.

## Version 1.1: kinds, time, signals `frontier-dataset` v1.1 (backward compatible with v1)

`manifest.json` gains:

```json
{"version": 1, "minor": 1,
 "sample_kind": "surface | mesh | grid | point_cloud",
 "temporal": "steady | transient",
 "time": {"n_steps": 100, "dt": 0.01, "units": "s", "uniform": true},
 "domain": "cfd | structural | thermal | multiphysics | materials | geophysical | other",
 "license": "CC-BY-SA-4.0", "citation": "...", "hub_id": "ahmedml"}
```

Per sample, exactly one of:

- `surface.zarr` (v1): `points (N,3)`, `faces (M,3)`, `cell_area (M,)`, `fields/<name>`, `cell_fields/<name>`, `lod/<level>/...`.
- `mesh.zarr`: unstructured 2D/3D volume mesh: `points (N,D)`, `cells (K,P)` padded with -1,
  `cell_types (K,)` (VTK cell type ids), `fields/<name>`, `cell_fields/<name>`.
- `grid.zarr`: regular grid: `fields/<name>` shaped `(T?, X, Y[, Z], C)`, `coords/x|y|z`,
  attrs `spacing`, `origin`, `boundary_conditions`.
- `points.zarr`: point cloud: `points (N,D)`, `fields/<name>`.

Time: transient samples carry `times (T,)` f64 at the group root and every time-dependent
array has time as its **first** axis, chunked one step (or a few) per chunk so a viewer or
loader can fetch a single step. Every field array carries the attribute
`dims = ["time","point","component"]` (or `["time","x","y","component"]`, `["cell","component"]`,
...). Readers must use `dims`; when absent the v1 default `["point"|"cell","component"]`
applies. Moving meshes add `points_t (T,N,D)`. Static inputs (geometry, material maps, BC
masks) live under `static/<name>` without a time axis.

Additions (v1.1, backward compatible; reference: `packages/frontier-data/README.md`):
global signals (force coefficients over time, probes, ODE outputs) live in the sample's store
as `signals/<name>` shaped `(T, C)` with `dims = ["time","component"]`; cell-centred grids set
the root attribute `location: "cell"` (`coords` are then cell centres, default `"node"`); every
store carries the root attribute `sample_kind`; small per-step arrays may group up to 16 steps
per chunk; rollout predictions are a sample store of the same kind with fields
`<field>_pred|_true|_err[|_std]`, uploaded as the run artifact `predictions/<sid>.zarr.tar`.
Step-by-step writing: `frontier_data.SampleAppender` / `DatasetWriter.append_sample`.

`meta.json` keeps the `v_CFD` vector and adds `time` (t0, dt, n_steps, averaging window),
`parameters` (the per-sample scalar conditioning: Re, Ma, AoA, material constants, ...).

Reference readers/writers: Python `frontier_data.dataset`,
Rust `frontier-server/src/zarr.rs`.

Server additions: `GET .../samples/{sid}/surface?lod=&t=<index>` and
`GET .../samples/{sid}/grid?field=&t=<index>&stride=<n>` (Arrow IPC: one row, `values`
List<f32> + schema metadata `shape`, `dims`, `time`), `GET .../samples/{sid}/times`.

## Quality & UQ vocabulary

Quality checks (each yields `{"check":..,"status":"pass|warn|fail","detail":..}`): nan/inf
fields, field range outliers (per-field robust z-score vs dataset), degenerate faces, area
sum vs STL area, force-coefficient consistency (integrated pressure drag vs csv), duplicate
geometry (hash of geometry params), parameter coverage (Latin-hypercube gaps), split leakage.
Dataset-level: parameter-space coverage, field statistics, recommended normalization.

Uncertainty quantification: deep ensemble (≥3 seeds) or MC-dropout giving per-point std;
conformal prediction on the calibration split giving 90% intervals for Cd; OOD score by
Mahalanobis distance of geometry parameters vs train set.

## Metric keys

`train/loss`, `train/lr`, `train/samples_per_s`, `train/loader_wait_frac`, `train/gpu_mem_gb`,
`val/loss`, `val/field/p/l2_rel`, `val/field/p/linf`, `val/cd/mae`, `val/cd/rel_err`,
`val/cd/r2`, `test/...` same, `uq/cd/coverage_90`, `uq/cd/interval_width`, `ood/score_mean`.
Steps are optimizer steps; epoch metrics also log `epoch`.
