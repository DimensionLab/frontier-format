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
| `views`      | `frontier.views`           |
| `data_adapters` | `frontier.data_adapters` |

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

## Artifact view renderers

Optional `frontier.views` entry points implement the independently versioned `views`
v1 API. Declare `views = ">=1,<2"` in the manifest's `[api]` table. An entry point
resolves to a callable accepting immutable artifact bytes and returning a self-contained
HTML string. Use only bundled assets and escape all input labels. The generic host
supplies an opaque-origin iframe; renderers cannot depend on network access, parent
DOM access or account credentials. Installed plugins remain trusted Python code.

`python -m frontier_plugins.views list` reads descriptors and enable/compatibility
state without importing renderer code. `render --plugin ID --view NAME --source FILE`
is the operator-side bridge. A network host must authorize access to the source before
calling it; never pass a client-supplied filesystem path. The Frontier server resolves
registered run artifacts after project authorization. Requests are bounded and outputs
include the source SHA-256 so human and agent clients can identify the same evidence.
This interface provides read-only artifact visualization; execution nodes and commercial
entitlement checks are separate capabilities.

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

## Data adapter factories

Requires `frontier-plugins>=1.3,<2`. Register a trusted, installed factory with
the `frontier.data_adapters` group and an explicit `api.data_adapters` range:

```toml
# pyproject.toml
[project.entry-points."frontier.plugins"]
acme-data = "acme_data"

[project.entry-points."frontier.data_adapters"]
reader = "acme_data.reader:open_array"
```

The package's `plugin.toml` includes its ordinary identity, version and author
metadata plus:

```toml
kind = "data-adapter"

[api]
data_adapters = ">=1,<2"

[data_adapters.reader]
summary = "Bounded local array reads"
stages = ["read"]
devices = ["cpu"]
platforms = ["Linux", "Darwin", "Windows"]
determinism = "exact"
requires = ["acme-array-engine>=1,<2"]
```

`catalog()` checks metadata, API compatibility, disable state, platform and
installed requirements without importing the factory. `available` means those
checks passed; `runtime_verified` remains false until the caller tests an actual
operation on the intended executor. Each executor discovers its own environment.

```python
from frontier_plugins.data_adapters import catalog, load

print(catalog())
open_array = load("acme-data", "reader", stage="read", device="cpu")
backend = open_array(source, **engine_options)
```

A `read` factory used by `ArrayFieldReader.from_plugin` accepts a source mapping
and engine options and returns an ArrayRoute backend. The wrapper validates
scientific dimensions and applies bounded selections. Native library caches are
separate allocations and must have explicit engine budgets. A transform factory
has its operation-specific documented parameters; Frontier's `normalize` factory
accepts `mean`, `scale` and `device` and returns a callable for float32 tensors.
Cache the selected factory/operation outside per-sample loops.

With `frontier-plugins>=1.4`, consumers retain provenance with `activate(...)`
instead of `load(...)`. It
returns `(factory, receipt)`, recording the plugin and adapter ids, distribution
version, entry-point target, supported API ranges and capabilities. The version
belongs to the plugin's own release sequence. Activation rejects a package
version or entry-point change between discovery and import; callers must still
pin their complete environment for reproducibility.

`exact` describes lossless deterministic reads of a fixed source. `fixed-runtime`
requires the same recorded execution environment for exact arithmetic replay;
`statistical` must supply its own error/distribution checks. A declaration alone
is not evidence: providers must test their claims. Snapshot names alone do not
freeze mutable data. Source identity, units, split and statistics provenance must
be retained by the workflow.

Stages `write`, `load` and `visualize` are reserved capability names. They do not
create a writer, trainer or renderer automatically. New consumers need a defined
call contract and acceptance tests before advertising an integration.
