# Frontier plugin API

`frontier-plugins` supplies compatibility checks for Python extensions. Its public API
versions are independent of Frontier's application version. Its only runtime dependency
is `packaging`; metadata discovery does not execute third-party code.

A community plugin can release `17.3.0` while Frontier releases `0.9.2`. Its manifest
version must match its own wheel metadata; PEP 440 equivalent spellings such as `1.0`
and `1.0.0` are accepted. Neither must equal Frontier's version.

Register extensions with standard Python entry points and ship a `plugin.toml` in the
package named by the `frontier.plugins` entry point. For example:

```toml
schema = 1
id = "acme-flow"
kind = "connector"
name = "Acme Flow"
version = "17.3.0"
licence = "Apache-2.0"
data_model_version = "1.1"

[api]
connectors = ">=1,<2"

[connectors]
acme-flow = "acme_flow.reader:FlowReader"
```

The distribution declares `frontier-plugins>=1,<2` plus tested version ranges for
framework packages it imports in its `pyproject.toml` dependencies. Those dependency
ranges and API requirements must both pass. Test new framework releases before
expanding dependency ranges; private imports do not become stable public interfaces.

| API          | Entry point group          |
| ------------ | -------------------------- |
| `connectors` | `frontier.connectors`      |
| `importers`  | `frontier.importers`       |
| `exporters`  | `frontier.exporters`       |
| `ops`        | `frontier.ops`             |
| `benchmarks` | `frontier_eval.benchmarks` |
| `geometry`   | `frontier_data.geometry`   |
| `converters` | `frontier_hub.converters`  |
| `hosts`      | `frontier_deploy.hosts`    |
| `models`     | `frontier_train.models`    |
| `profilers`  | `frontier.profilers`       |

These interfaces currently expose API `1.0.0`. Declare every interface the plugin
requires or provides under `[api]`, using PEP 440 version ranges. Unknown required
interfaces and unsatisfied ranges prevent loading. Existing plugins without API
metadata remain usable through the legacy v1 interface, with a compatibility warning.

The `profilers` v1 interface exposes a `TraceSession(Path)` class with `info`,
`events`, `event`, `summary`, `compare` and `close` methods. It is an analysis
interface, not a native canvas node or capture-control contract. Optional packages
must be loaded through the compatibility helper and retained outside hot loops.

## Model registration modules

A model package registers its factories when its entry-point module is imported. The
loader checks the manifest, dependencies and disable state first. It does not invoke an
entry-point function. For example, a package released as `17.3.0` can declare:

```toml
# pyproject.toml
[project.entry-points."frontier.plugins"]
acme-models = "acme_models"

[project.entry-points."frontier_train.models"]
acme-models = "acme_models.registration"
```

Ship this manifest as `acme_models/plugin.toml`:

```toml
schema = 1
id = "acme-models"
kind = "model"
name = "Acme research models"
version = "17.3.0"
licence = "Apache-2.0"

[api]
models = ">=1,<2"
```

Use `frontier_train.models.register(name)` for surface factories or
`frontier_train.transient.models.register(name, kinds)` for time-stepping factories.
The models API covers those registration functions and the constructor/forward tensor
contracts documented in their modules. Other trainer internals remain private.
Declare tested `frontier-train` dependency ranges as well as `frontier-plugins>=1,<2`.
The plugin version must match its own distribution, not the trainer or application.

Discovery is cached for the process. Activation failures log the plugin name, module and
reason; registrations made by a failed import are removed from both model registries.
Legacy registration modules without a manifest still load with a diagnostic. Restart the
worker after installing, enabling or disabling plugins. Direct registration inside a
research script remains available and does not pass through package discovery.

This registers models for the trainer. Arbitrary models still need adapters for canvas
configuration or serving; registering a factory does not generate either adapter.

## Compatibility policy

- Patch API releases fix behavior without changing supported signatures or schemas.
- Minor API releases add optional functionality while preserving existing behavior.
  Plugins using a new feature raise their minimum API minor.
- Breaking changes require a new API major; a product release alone never changes it.
- Deprecated public symbols remain throughout their current API major. Document the
  replacement and removal major and emit `DeprecationWarning` when they are used.
  Plugin CI should treat deprecation warnings as errors.
- Dataset format versions are checked separately from interface compatibility.
- Rust plugins use the core plugin API major and data model contract. Statically
  compiled Rust traits do not promise a stable dynamically loaded Rust ABI.

## Activation and testing

Extension loaders call `frontier_plugins.load_entry_point(ep)` instead of `ep.load()`.
The helper checks metadata, API ranges, dependency requirements, and disable state
before executing code. Catch failures per plugin so a broken extension does not
prevent another from loading. Retain loaded objects in the registry; do not repeat
compatibility checks in per-sample or GPU execution loops.

Test real wheels in clean environments against the oldest and newest framework
versions you declare, supported Python versions and relevant operating systems.
Run `python -m frontier_connectors plugins doctor` and test actual operations with
redistributable fixtures, including missing optional dependencies and malformed input.
A successful import alone is not a functional contract test.

The compatibility layer is not a sandbox. In-process extensions have host privileges.
Use separate worker environments for conflicting dependencies and OS/container
isolation where untrusted execution requires it.

Standards: [PyPA entry points](https://packaging.python.org/en/latest/specifications/entry-points/),
[dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/),
[version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/).
