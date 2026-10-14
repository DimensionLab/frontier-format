# frontier-format

Frontier data model and dataset format: the frontier-dataset specification, the plugin contract, and reference readers/writers in Rust and Python.

Physics ML has model code and solver output, and very little in between. `frontier-dataset`
is that in-between: a chunked, cloud-friendly on-disk layout (Zarr v3 plus a JSON manifest)
for surfaces, volume meshes, regular grids and point clouds, steady or transient, with the
metadata a model needs for conditioning and an auditor needs for lineage (solver, physics
model, boundary conditions, mesh statistics, fidelity tier).

## What is in this repository

- `crates/frontier-core/`: Rust crate `frontier-core`
- `crates/frontier-data/`: Rust crate `frontier-data`
- `crates/frontier-prep-kernels/`: Rust crate `frontier-prep-kernels`
- `packages/frontier-data/`: Python package `frontier-data`
- `packages/frontier-prep/`: Python package `frontier-prep`
- `spec/`: the normative specification (when present), licensed CC-BY-4.0

`frontier-core` is the dependency-light base every other Frontier component builds on: ids,
run and metric types, and the plugin contract (`Plugin`, `PluginManifest`,
`DATA_MODEL_VERSION`). `frontier-data` (Rust) is the Arrow-based data model; `frontier-data`
(Python) is the reference reader/writer, validator and dataset tooling.

```python
from pathlib import Path
from frontier_data.dataset import read_surface

mesh = read_surface(Path("datasets/my-cases/samples/run_1/surface.zarr"), lod=0)
```

## Stability

The format is versioned (`manifest.json`: `version`, `minor`). Minor versions are additive
and readers must ignore what they do not know. A new major version comes with a converter
and one release of overlap in which both are read. Independent implementations are welcome;
the validator in the Python reference package is the conformance check.

## Plugins

Every extension point of Frontier (connectors, metric sinks, visualizers, exporters) is a
plugin that declares the data-model version it was built against:

```rust
PluginManifest { id: "connector-openfoam".into(), kind: PluginKind::Connector,
                 data_model_version: frontier_core::DATA_MODEL_VERSION, .. }
```

Python plugins use entry points (`frontier.connectors`, ...); see
[frontier-connectors](https://github.com/DimensionLab/frontier-connectors) for a worked example.

## Ecosystem

| Repository | What it is |
|---|---|
| [arrayroute](https://github.com/DimensionLab/arrayroute) | Bounded scientific array selection and measured storage query planning. |
| **frontier-format** (this repository) | Frontier data model and dataset format: the frontier-dataset specification, the plugin contract, and reference readers/writers in Rust and Python. |
| [frontier-connectors](https://github.com/DimensionLab/frontier-connectors) | Connector plugins that turn CFD/CAE solver output (OpenFOAM, VTK, Fluent, STAR-CCM+, CGNS, LS-DYNA, ...) into Frontier datasets. Write your own in an afternoon. |
| [frontier-sdk](https://github.com/DimensionLab/frontier-sdk) | Frontier Python SDK: offline-first experiment tracking for physics ML, the native Rust core, and the NVIDIA PhysicsNeMo bridge. |
| [frontier-physicsnemo](https://github.com/DimensionLab/frontier-physicsnemo) | NVIDIA PhysicsNeMo plugin for Frontier: logging, model registry, checkpoint compatibility and dataset adapters. Installable independently of the platform. |
| [frontier-hub](https://github.com/DimensionLab/frontier-hub) | Curated hub of public physics-AI datasets and benchmarks: catalog, fetchers and converters to the Frontier dataset format. |
| [frontier-editor](https://github.com/DimensionLab/frontier-editor) | Editor core and GPU text view for physics-ML configs, recipes and logs: rope document, undo tree, hunks with per-hunk approval, tree-sitter syntax, JSON-Schema diagnostics, provenance events, and a wgpu renderer that draws into the host's render pass. |
| [frontier-viz](https://github.com/DimensionLab/frontier-viz) | GPU-accelerated (wgpu) rendering core for meshes, fields and neural-network topology: native, browser (WebGPU) and headless. |
| [foldstate](https://github.com/DimensionLab/foldstate) | Statecharts as data with an event log: a pure, deterministic kernel (state, event) -> (state, effects), replay, fork, restore and explanations. no_std + alloc, runs natively and in WebAssembly. Not specific to Frontier. |
| [frontier-zed](https://github.com/DimensionLab/frontier-zed) | Frontier for Zed: JSON schemas, snippets and the Frontier MCP context server for physics-ML training configs, RunSpecs, data-prep recipes, pipelines, hub entries and plugin manifests. A Zed extension (extension.toml plus a small wasm32-wasip2 crate), installable as a dev extension today and submitted to the Zed extension registry later. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes to the format start as an issue. Licensed
under [Apache-2.0](LICENSE).
