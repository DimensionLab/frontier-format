# SPDX-License-Identifier: Apache-2.0
"""Converters from public transient datasets to ``frontier-dataset`` v1.1.

* :mod:`frontier_data.ingest.the_well` – The Well (Polymathic AI, HDF5 on the HF Hub) ->
  transient ``grid.zarr`` samples.
* :mod:`frontier_data.ingest.meshgraphnets` – DeepMind MeshGraphNets TFRecords (parsed without
  TensorFlow, streamed over HTTP) -> transient ``mesh.zarr`` samples.

Both expose a pure conversion function (local raw files -> dataset) that hub converters can
call, and an ``ingest_*`` function that also downloads a bounded subset.
"""

from frontier_data.ingest.meshgraphnets import (
    convert_meshgraphnets,
    ingest_meshgraphnets,
    read_tfrecord_trajectories,
)
from frontier_data.ingest.the_well import convert_the_well, ingest_the_well, read_well_file

__all__ = [
    "convert_meshgraphnets",
    "convert_the_well",
    "ingest_meshgraphnets",
    "ingest_the_well",
    "read_tfrecord_trajectories",
    "read_well_file",
]
