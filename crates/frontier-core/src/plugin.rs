// SPDX-License-Identifier: Apache-2.0
//! Plugin system contract.
//!
//! Frontier is a minimal core plus plugins. A plugin is anything that implements
//! [`Plugin`] and ships a [`PluginManifest`]. Concrete plugin families (simulation
//! connectors, metric sinks, visualizers, schedulers) extend this with their own trait
//! in their own crate, e.g. `frontier_connectors::Connector`.

use serde::{Deserialize, Serialize};

/// Families of plugins the core knows how to route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PluginKind {
    /// Imports simulation results (OpenFOAM, Fluent, STAR-CCM+, VTK, CGNS, ...) into datasets.
    Connector,
    /// Receives metrics, artifacts and events from training runs.
    MetricSink,
    /// Renders datasets, fields, meshes or model topology.
    Visualizer,
    /// Submits and tracks jobs on a compute backend (SLURM, Kubernetes, local, cloud).
    Scheduler,
    /// Exports trained models to deployment targets (ONNX, TensorRT, NIM, edge).
    Exporter,
}

/// Static description of a plugin, loaded before the plugin itself is instantiated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Unique, stable, kebab-case identifier, e.g. `connector-openfoam`.
    pub id: String,
    pub name: String,
    pub version: String,
    pub kind: PluginKind,
    /// Core data-model version this plugin was built against.
    pub data_model_version: u32,
    #[serde(default)]
    pub description: String,
}

/// Minimal behaviour every plugin provides.
pub trait Plugin: Send + Sync {
    fn manifest(&self) -> &PluginManifest;
}
