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
| `inspect <dataset>`            | Print the manifest summary, splits and a table of samples                                                         |
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
