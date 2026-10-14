# SPDX-License-Identifier: Apache-2.0
"""frontier-data: dataset ingestion, LOD, quality checks and splits for CFD surrogate training.

The package implements the ``frontier-dataset`` v1 layout and its v1.1 additions (mesh /
grid / point-cloud samples, transient arrays), specified in ``docs/spec/dataset-format.md``:
a directory with ``manifest.json``, one ``samples/<id>/`` folder per CFD case (``meta.json``,
``surface.zarr``, ``forces.json``) and a ``quality/`` report.
"""

from frontier_data.arrays import (
    Sample,
    SampleAppender,
    write_grid_sample,
    write_mesh_sample,
    write_points_sample,
    write_surface_sample,
)
from frontier_data.dataset import (
    FORMAT_MINOR,
    Dataset,
    DatasetWriter,
    Manifest,
    SampleMeta,
    TimeInfo,
)
from frontier_data.mesh import SurfaceMesh
from frontier_data.pipeline import Node, Pipeline, PipelineEvent, PipelineRunner
from frontier_data.splits import find_trajectory_leakage, group_split
from frontier_data.validate import validate_dataset

FORMAT_NAME = "frontier-dataset"
FORMAT_VERSION = 1

__all__ = [
    "FORMAT_MINOR",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "Dataset",
    "DatasetWriter",
    "Manifest",
    "Node",
    "Pipeline",
    "PipelineEvent",
    "PipelineRunner",
    "Sample",
    "SampleAppender",
    "SampleMeta",
    "SurfaceMesh",
    "TimeInfo",
    "find_trajectory_leakage",
    "group_split",
    "validate_dataset",
    "write_grid_sample",
    "write_mesh_sample",
    "write_points_sample",
    "write_surface_sample",
]
