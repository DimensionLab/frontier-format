# Frontier JSON Schemas

JSON Schema (draft 2020-12) for every file kind the Frontier editor validates. Generated
from the source models by the monorepo's schema exporter; a drift test fails when a file
here is older than its source. Licence: CC-BY-4.0 like the rest of the specification.

| Schema | Source of truth | How |
|--------|-----------------|-----|
| `frontier-train-config` | `frontier_train.config.Config`, `frontier_train.transient.config.TransientConfig` | pydantic `model_json_schema()`; `task: transient` selects the transient branch |
| `recipe` | `frontier_prep.recipes.Recipe` + every op's `Params` from the op registry | pydantic; each step's `params` is tied to its `op` with `if`/`then` |
| `hub-entry` | `frontier_hub.schema.Entry` | pydantic |
| `benchmark-spec` | `frontier_eval.benchmarks.BenchmarkSpec` | pydantic |
| `frontier-config` | the `frontier-config` settings registry | `frontier config schema --json-schema` |
| `plugin-manifest` | `frontier_connectors.plugins.manifest.PluginManifest` | dataclass converted field by field |
| `runspec`, `jobspec` | the `frontier-cloud` `RunSpec` / `JobSpec` Rust types | **hand-maintained** (`x-frontier-hand-maintained: true`); update when the Rust types change |
| `pipeline-v1` | the canvas `Pipeline` Rust type | hand-maintained |

Regenerate in the monorepo: `pnpm nx run schemas:export`. Check: `pnpm nx run schemas:test`.
