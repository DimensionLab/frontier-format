<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model package format (`frontier-model` v1)

A trained physics model as one self-describing unit: the network in one or more formats, every
constant needed to turn physical inputs into network inputs and back, where the model came
from, how well it did, and where it may be used. A package is what a registry stores, what an
inference server loads, and what is copied onto an embedded device.

Status: normative for version 1. Readers must reject a `version` they do not know.

## Layout

A package is a directory, or the same directory as an uncompressed tar with `model.json` as its
first member (a reader can stop after the manifest). Paths inside the package are relative and
use `/`.

```
model.json               manifest (this document)
prepost.json             the pre/post-processing constants flattened for non-Python integrations
model_card.json          the trainer's model card of the checkpoint (provenance, verbatim)
normalization.json       training-split statistics (verbatim)
parity/reference.npz     small real inputs and the reference outputs of the training framework
weights/<variant>/...    one directory per variant (see "Variants")
```

Producers may add files; every file except `model.json` must be listed in `files` with its
SHA-256. Weight files do not compress, which is why the tar is not gzipped.

## `model.json`

```jsonc
{
  "format": "frontier-model", "version": 1,
  "name": "ahmed-transolver-lite",          // [a-z0-9._-]+, the registry's model name
  "model_version": "1.0.0",                 // free-form; semantic versions recommended
  "created_at": "2026-10-11T00:00:00Z",
  "description": "...",
  "task": "surface",                        // surface | transient
  "sample_kind": "surface",                 // surface | grid | mesh | points
  "architecture": {
    "name": "transolver_lite", "params": {...}, "n_params": 1610054,
    "members": 1,                           // > 1: deep ensemble in one graph
    "export_rewrites": ["physics-attention: einsum/sdpa -> matmul+softmax"]
  },
  "inputs":  [TensorSpec, ...],             // network inputs, in call order
  "outputs": [TensorSpec, ...],
  "preprocessing":  {...},                  // see below
  "postprocessing": {...},
  "provenance":     {...},
  "evaluation":     {...},
  "license":        {...},
  "intended_use":   {...},                  // includes the operating envelope
  "variants":       [Variant, ...],
  "export_failures": [{"format", "stage", "error"}],
  "files":     {"<path>": {"sha256": "...", "bytes": 123}},
  "signature": null | {"algorithm": "hmac-sha256", "key_id": "...", "value": "<hex>"}
}
```

### TensorSpec

```json
{
  "name": "points",
  "dtype": "float32",
  "shape": ["batch", "n_points", 3],
  "units": "normalised",
  "description": "...",
  "dynamic_axes": { "0": "batch", "1": "n_points" }
}
```

A string in `shape` is a dynamic axis; `dynamic_axes` repeats them by index. The network works
in normalised units; the constants in `preprocessing` and `postprocessing` map to physical
units.

Surface models (`task: surface`):

| name                               | shape                  | meaning                                                                                         |
| ---------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------- |
| `points` (in)                      | `[batch, n_points, 3]` | prediction locations, `(xyz - center) / scale`                                                  |
| `cond` (in)                        | `[batch, C]`           | conditioning vector, z-scored (`[batch, 1]` of zeros when the model has none)                   |
| `p` (out)                          | `[batch, n_points, 1]` | the surface field, normalised                                                                   |
| `cd` (out)                         | `[batch, 1]`           | drag coefficient, normalised                                                                    |
| `p_std`, `cd_std` (out, ensembles) | as above               | standard deviation over the members, in normalised units (multiply by `std`, do not add `mean`) |

Transient models (`task: transient`): `x` `[batch, *S, history * n_state]` (the history window,
states stacked on the last axis, oldest first), optional `cond` `[batch, P]`, `static`
`[batch, *S, Sc]` (z-scored static fields followed by the one-hot node type), `points`
`[batch, n_points, D]`; output `next_state` `[batch, *S, n_channels]`. `*S` is the grid shape
(fixed in the file) or `n_points`.

### `preprocessing`

```jsonc
{
  "coordinate_frame": {"units": "m", "description": "..."},
  "points": {"center": [cx, cy, cz], "scale": s, "location": "cell"},   // cell = face centroids
  "point_sampling": {
    "policy": "uniform-random-chunks",       // or "all-nodes" for time steppers
    "chunk_size": 131072,                    // points per forward pass
    "train_num_points": 32768, "seed": 0,
    "last_chunk": "topped up with uniformly drawn points (results discarded)"
  },
  "conditions": {"names": [...], "mean": [...], "std": [...], "units": "mm",
                 "definition": "one value per name, in this order; z-scored with mean/std"},
  // transient only
  "state": {"fields", "channels", "slices", "aux_fields", "n_state", "n_channels",
            "mean", "std", "history", "lead", "stride", "dt", "predict"},
  "grid": {"shape": [128, 384]} | null,
  "static": {"fields", "mean", "std", "node_type_classes"}
}
```

**Point sampling is normative.** Models that pool a global context over the points of one
forward pass (Transolver slice tokens, pooled drag features) must not be fed contiguous pieces
of a mesh: mesh order is spatially clustered and a contiguous chunk is one patch of the body.
A reader with more points than `chunk_size` shuffles the point indices once, feeds consecutive
chunks of the shuffled order, tops the last chunk up to `chunk_size` with randomly drawn points
whose outputs are discarded, and averages scalar outputs over chunks. The seed only changes
which points share a chunk.

### `postprocessing`

```jsonc
{
  "fields":  {"pMean": {"output": "p", "mean": m, "std": s, "units": "m^2/s^2",
                        "location": "cell", "std_output": "p_std"}},
  "scalars": {"cd": {"output": "cd", "mean": m, "std": s, "units": "-",
                     "aggregate": "mean over chunks", "std_output": "cd_std",
                     "conformal": {"quantile": q, "normalized": true, "confidence": 0.75}}},
  "forces":  {"method": "pressure-integration", "field": "pMean",
              "formula": "F = sum_i p_i * n_i * A_i (outward normals, per prediction point)",
              "units": "N/(kg/m^3)", "reference": {"u_inf": 1.0, "rho": 1.0,
              "reference_area": 0.112, "drag_direction": [1,0,0], "lift_direction": [0,0,1]}}
}
```

Physical value = `output * std + mean`; a standard deviation output is multiplied by `std` only.
`conformal`, when present, gives the half-width of an interval for the scalar: `quantile`
(times the ensemble `std` when `normalized`). `forces` is the pressure part only.

### `provenance`

`run_id`, `run_name`, `dataset` (`name`, `manifest_sha256`, and when the builder saw the
dataset: `local_manifest_sha256`, `manifest_matches`, `lod`, `license`, `citation`,
`splits`), `git_sha`, `config` (the resolved training configuration), `training_metrics`,
`members` (per checkpoint: path, SHA-256, run id, epoch, step, seed), `base_model` (what a
fine-tune started from, or null).

### `evaluation`

Collected from evaluation outputs (`frontier-eval/1` documents):
`status` (`evaluated` | `not-evaluated`), `benchmarks` (name, split, primary metric and value,
tier, passed, thresholds), `metrics`, `verdict` (`credible`: true | false | null, `tier`: the
worst benchmark tier), optional `uq` (metrics and conformal calibration), `report`, and
`edge_recommendation` (the smallest variant whose benchmark metrics stay within the stated
tolerance of FP32).

### `intended_use` and the operating envelope

```jsonc
{
  "description": "...", "limits": ["..."],
  "envelope": {
    "policy": "flag",                        // flag | refuse
    "range_margin": 0.0,                     // fraction of each range allowed outside it
    "parameters": {"<name>": {"min", "max", "mean", "std"}},   // training split
    "n_train": 31,
    "mahalanobis": {"names": [...], "mean": [...], "cov_inv": [[...]],
                    "threshold": 3.72, "threshold_quantile": 0.99}
  }
}
```

An input is **in the envelope** when every named parameter lies within its training range
(widened by `range_margin`) and the Mahalanobis distance
`sqrt((x - mean)^T cov_inv (x - mean))` is at most `threshold` (the 99th percentile of the
training distances; the covariance is shrunk towards its trace by 1e-3). Readers return the
flag and the distance with every prediction; with `policy: refuse` (or a per-request override)
they reject out-of-envelope inputs.

### Variants

```jsonc
{
  "id": "onnx-fp16", "format": "onnx",       // onnx | torchscript | exported-program | pytorch
                                             // | tensorrt | coreml | executorch
  "precision": "fp16",                       // fp32 | tf32 | fp16 | int8-dynamic | int8-static | int8 | fp32-pruned
  "default": false,
  "files": [{"path": "weights/onnx-fp16/model.onnx", "sha256": "...", "bytes": 3773126}],
  "opset": 18, "operators": ["Add", ...], "derived_from": "onnx-fp32",
  "max_points": 131072,                      // engines with a bounded profile
  "fixed_points": null,                      // static-shape engines: pad every chunk to this
  "quantization": {...}, "compression": {...},
  "parity":   {"reference": "pytorch-fp32", "max_abs", "max_rel", "l2_rel",
               "tolerance": {"max_abs", "l2_rel"}, "passed", "n_cases", "points_per_case", "per_output"},
  "accuracy": {"split", "n_samples", "chunk_size", "metrics", "delta_vs_fp32",
               "field_l2_rel_vs_fp32_mean", "field_l2_rel_vs_fp32_max", "tolerance", "within_tolerance"},
  "benchmarks": [{"host": {"label", "os", "arch", "cpu", "threads", ...}, "device", "providers",
                  "load_ms", "latency": [{"n_points", "batch", "latency_ms": {"p50","p95","p99","mean","min","n"},
                  "throughput_points_s"}], "peak_rss_mb"}]
}
```

`parity` compares the variant with the training framework's FP32 output on real samples in
normalised units: `max_abs` is the largest absolute difference, `max_rel` that divided by the
largest reference magnitude, `l2_rel` the relative L2 norm of the difference (worst output and
case). A variant is stored only when parity passes its precision's tolerance (FP32: max_abs
1e-3 and l2_rel 1e-4; TF32: 2e-2 / 2e-3; FP16: 0.25 / 2e-2); reduced-integer variants are
judged on `accuracy`, which uses the benchmark's own metrics on a held-out split. The first ONNX
FP32 variant is the default. `parity/reference.npz` holds two reduced cases
(`case<i>/in/<input>`, `case<i>/out/<output>`) so a target device can repeat the check.

A TensorRT plan is specific to the GPU model and TensorRT version that built it; its variant
records both.

### Integrity and signature

`files` lists the SHA-256 and size of every file but `model.json`. A reader verifies the files
of the variant it loads; a full verification also reports missing and unlisted files.

`signature` (optional) is computed over the canonical manifest: `model.json` without the
`signature` key, serialised with sorted keys, `(",", ":")` separators and UTF-8. Because the
manifest carries every file's hash, the signature covers the whole package. Version 1 defines
`hmac-sha256` (shared key, identified by `key_id`); asymmetric algorithms (`ed25519`) are
reserved for a later version.

## Model contract (`model-contract.v1`)

What a model takes and what it produces, for every model Frontier knows: a training run
writes it into its model card (`extra.contract`) from the dataset version and the config,
a package carries it as `contract` in `model.json`, a hub model has one derived from its
catalog entry. A model made before contracts existed gets one derived from what it records
(its card, its `model.json`), with `derived: true`. Read it with
`python -m frontier_train contract <model>` (`frontier_train.contract.contract_of`).

```jsonc
{
  "schema": "model-contract.v1",
  "contract": 1,
  "task": "steady-surface-field", // steady-surface-field | steady-mesh-field |
  // transient-grid | transient-mesh | scalar-regression
  "inputs": {
    "geometry": {
      "kind": "surface", // surface | mesh | grid | points
      "resolution": {
        "kind": "points", // points | spacing
        "trained": [10002, 10002], // fewest and most points seen
        "range": { "min": 2500, "max": 80016 },
        "augmented": false, // resolution varied on purpose
        "trained_lod": 1,
      }, // for the record only
      "requires": ["points", "normals"],
      "units": { "length": "m" },
    }, // {} when the dataset does not say
    "fields": [], // per-point/cell input fields, canonical names
    "parameters": [{ "name": "slant-angle-degrees", "required": true }],
    "conditioning": [],
  },
  "outputs": {
    "fields": [{ "name": "pressure", "location": "cell", "dims": ["scalar"] }],
    "scalars": ["drag_coefficient"],
    "uncertainty": { "kind": "ensemble", "members": 5 }, // absent for a single model
  },
  "normalisation": { "ref": "v_7eb1…" }, // the dataset version the statistics came from
  "trained_on": {
    "dataset": "ahmedml",
    "hub_id": "ahmedml",
    "version": "v_7eb1…",
  },
  "derived": true, // only on a contract that was not written at training
}
```

Field names are the canonical names of the preparation vocabulary. `resolution` is a
tolerance, not a requirement: a level of detail is our own decimation, never something a
model needs. Training records the point counts it read (`trained`) and the range the
model's family tolerates (`range`): point-set models trained on random subsets (PointMLP,
Transolver) read from a quarter of the fewest to eight times the most points they saw; other
families, the counts they saw. A sample without the level the model was trained at is read
as it is when it is within the range, else resampled toward it with the LOD kernels the
datasets are written with; the predictions record which (`provenance.config.resampled`,
per sample `{lod, trained_lod?, action: full | resampled}`). Varying resolution per epoch
on purpose (`augmented: true`) is not built yet.

**Compatibility.** `frontier_train.compat.check(contract, target)` (`python -m
frontier_train compat <model> <dataset> [--ids]`) returns `{ok, adaptable, verdict,
problems: [{kind, …, fix, line}]}`. Kinds: `resolution_out_of_range` (fix `resample`),
`missing_parameter` (`set_value`), `unit_mismatch` (`convert`), `geometry_kind` (no fix).
`verdict` is the one line a node shows: "Fits.", "Needs adapting: resample to the model's resolution.",
"Does not fit: …". Inference refuses data with a problem other than its resolution, with
that verdict and what to do.

**Adapting.** `python -m frontier_prep adapt --dataset <dir> --out <dir> --fixes
<verdict.json> [--contract <contract.json>] [--set name=value]`
(`frontier_prep.adapt.adapt`) copies the dataset and applies the verdict's fixes as a
recipe of ordinary preparation ops: `resolution_out_of_range` → `prep.mesh.decimate`
(level 1 at twice the wanted points as faces), `missing_parameter` → `prep.params.set`
(the value from `--set`), `unit_mismatch` → `prep.units.convert`, a renamed field →
`prep.fields.rename`. The copy's head version carries the ops' provenance and
`adapted.json` records the contract, the recipe and the run. On a board this is the
`adapt` step (`docs/spec/pipeline-run-events.md`).

## `prepost.json`

The subset a C, C++ or Rust integration needs, without the provenance:

```json
{"format": "frontier-model-prepost", "version": 1, "model": "name:version", "task": "surface",
 "inputs": [...], "outputs": [...], "preprocessing": {...}, "postprocessing": {...},
 "envelope": {...}}
```

## Compatibility

New optional keys may appear in any object within version 1; readers ignore keys they do not
know. Removing or changing the meaning of a key, or a new required key, increments `version`.
