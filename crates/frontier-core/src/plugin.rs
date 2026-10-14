// SPDX-License-Identifier: Apache-2.0
//! Plugin system contract.
//!
//! Frontier is a minimal core plus plugins. A plugin is anything that implements
//! [`Plugin`] and ships a [`PluginManifest`]. Concrete plugin families (simulation
//! connectors, metric sinks, visualizers, schedulers) extend this with their own trait
//! in their own crate, e.g. `frontier_connectors::Connector`.

use serde::{Deserialize, Serialize};

/// Public Rust trait contract; independent of the application and plugin release versions.
pub const PLUGIN_API_MAJOR: u32 = 1;

fn default_api_major() -> u32 {
    PLUGIN_API_MAJOR
}

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
    #[serde(default = "default_api_major")]
    pub api_major: u32,
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

impl PluginManifest {
    pub fn check_compatibility(&self) -> Result<(), String> {
        if self.api_major != PLUGIN_API_MAJOR {
            return Err(format!(
                "{} requires plugin API {}, host provides {}",
                self.id, self.api_major, PLUGIN_API_MAJOR
            ));
        }
        if self.data_model_version != crate::DATA_MODEL_VERSION {
            return Err(format!(
                "{} requires data model {}, host provides {}",
                self.id,
                self.data_model_version,
                crate::DATA_MODEL_VERSION
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_plugin_release_and_legacy_manifest() {
        let mut manifest: PluginManifest = serde_json::from_value(serde_json::json!({
            "id": "community-reader", "name": "Community reader", "version": "17.3.0",
            "kind": "connector", "data_model_version": crate::DATA_MODEL_VERSION
        }))
        .unwrap();
        assert!(manifest.check_compatibility().is_ok());
        manifest.api_major = 2;
        assert!(manifest.check_compatibility().is_err());
        manifest.api_major = 1;
        manifest.data_model_version = crate::DATA_MODEL_VERSION + 1;
        assert!(manifest.check_compatibility().is_err());
    }
}
