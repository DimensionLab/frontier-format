// SPDX-License-Identifier: Apache-2.0
//! What a worker reports about the machine it runs on.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    #[serde(default)]
    pub memory_mb: u64,
    /// Stable device identity: the CUDA uuid (`GPU-...`), or `<node_id>:<index>` for devices
    /// without one (Apple, software adapters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<String>,
    /// The driver's index (`nvidia-smi -i`, `CUDA_VISIBLE_DEVICES` order).
    #[serde(default)]
    pub index: u32,
    /// `nvidia`, `apple`, `amd`, `intel`, `software`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub vendor: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_used_mb: Option<u64>,
    /// SM / core utilisation over the last sample, percent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utilization_pct: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pci_bus_id: Option<String>,
    /// Links to the other GPUs of the node (`nvidia-smi topo -m`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<crate::compute::GpuLink>,
    /// Graphics API the device is reachable through (`Metal`, `Vulkan`), for the renderers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_backend: Option<String>,
}

impl GpuInfo {
    pub fn new(name: impl Into<String>, memory_mb: u64) -> Self {
        Self {
            name: name.into(),
            memory_mb,
            uuid: None,
            index: 0,
            vendor: String::new(),
            memory_used_mb: None,
            utilization_pct: None,
            pci_bus_id: None,
            links: Vec::new(),
            adapter_backend: None,
        }
    }

    /// The identity used by pools and allocations.
    pub fn device_id(&self, node_id: &str) -> String {
        self.uuid
            .clone()
            .unwrap_or_else(|| format!("{node_id}:{}", self.index))
    }

    /// Number of NVLink peers.
    pub fn nvlink_peers(&self) -> usize {
        self.links.iter().filter(|l| l.is_nvlink()).count()
    }
}

/// Heartbeat payload (`POST /api/v1/workers/heartbeat`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub hostname: String,
    #[serde(default)]
    pub os: String,
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub worker_version: String,
    #[serde(default)]
    pub gpus: Vec<GpuInfo>,
    /// Which JobSpec.backend this worker serves (`local`, `runpod`, ...).
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Free-form labels (`gpu=h100`, `site=eu`) matched against JobSpec.labels.
    #[serde(default)]
    pub labels: Vec<String>,
    /// How many new jobs the worker can take right now (0 = just a liveness ping).
    #[serde(default = "default_capacity")]
    pub capacity: u32,
    #[serde(default)]
    pub running_jobs: Vec<crate::ids::JobId>,
    #[serde(default)]
    pub slurm_job_id: Option<String>,
    #[serde(default)]
    pub cuda_visible_devices: Option<String>,
    /// Stable per machine (machine-id, or a hash of the hostname): workers in several
    /// containers of one node share it, and pools name nodes by it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub node_id: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cpus: u32,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub memory_mb: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_version: Option<String>,
    /// `docker`, `apptainer`, `podman`, `slurm` when the agent runs inside one; image when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    /// Allocations this worker is currently serving; the server renews their leases.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allocations: Vec<String>,
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}
fn is_zero_u64(n: &u64) -> bool {
    *n == 0
}

fn default_backend() -> String {
    "local".to_string()
}

fn default_capacity() -> u32 {
    1
}

/// Response of the heartbeat: specs the worker now owns (already marked `running` server-side).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub jobs: Vec<crate::job::JobSpec>,
}
