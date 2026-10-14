// SPDX-License-Identifier: Apache-2.0
//! Compute allocation: what a job asks for (`ComputeRequest`), what the control plane hands
//! a worker (`Allocation`), and the device topology a worker reports. The pools themselves
//! are configuration (`frontier-config::compute`); see the JobSpec section of
//! docs/spec/api.md.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// What a device is used for. A pool carries a set of roles; a job asks for one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Training,
    Inference,
    Rendering,
    Prep,
    Any,
}

impl Role {
    pub const ALL: [Role; 5] = [
        Role::Training,
        Role::Inference,
        Role::Rendering,
        Role::Prep,
        Role::Any,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Role::Training => "training",
            Role::Inference => "inference",
            Role::Rendering => "rendering",
            Role::Prep => "prep",
            Role::Any => "any",
        }
    }

    /// Whether a pool with roles `have` serves a job asking for `self`.
    pub fn served_by(self, have: &[Role]) -> bool {
        have.iter()
            .any(|r| *r == Role::Any || *r == self || self == Role::Any)
    }
}

impl std::str::FromStr for Role {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "training" | "train" => Ok(Role::Training),
            "inference" | "infer" | "serving" | "serve" => Ok(Role::Inference),
            "rendering" | "render" => Ok(Role::Rendering),
            "prep" | "data" | "ingest" => Ok(Role::Prep),
            "any" | "" | "auto" => Ok(Role::Any),
            other => Err(format!(
                "unknown role `{other}`; expected training, inference, rendering, prep or any"
            )),
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How many GPUs a job wants, or exactly which ones.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GpuRequest {
    /// Per node.
    Count(u32),
    /// Explicit device uuids (`GPU-...`) or `node_id:index` pairs; one node only.
    Devices(Vec<String>),
}

impl Default for GpuRequest {
    fn default() -> Self {
        GpuRequest::Count(0)
    }
}

impl GpuRequest {
    /// GPUs per node this request amounts to.
    pub fn per_node(&self) -> u32 {
        match self {
            GpuRequest::Count(n) => *n,
            GpuRequest::Devices(d) => d.len() as u32,
        }
    }

    pub fn is_explicit(&self) -> bool {
        matches!(self, GpuRequest::Devices(d) if !d.is_empty())
    }
}

/// The compute part of a job or run: all optional, `auto` by default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ComputeRequest {
    /// Pool name; `None` or `auto` lets the policy choose among the pools serving `role`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pool: Option<String>,
    /// Defaults to `training` for train jobs, `prep` for data jobs, `any` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// GPUs per node (count) or explicit devices.
    #[serde(default, skip_serializing_if = "is_default_gpus")]
    pub gpus: GpuRequest,
    /// Nodes for a multi-node job (1 = single node). 0 means "unset" and reads as 1.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub nodes: u32,
    /// Minimum free memory per GPU.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub gpu_memory_gb: u32,
    /// Refuse to share a device, even inside a `shared` pool.
    #[serde(default, skip_serializing_if = "is_false")]
    pub exclusive: bool,
}

fn is_default_gpus(g: &GpuRequest) -> bool {
    *g == GpuRequest::Count(0)
}
fn is_zero(n: &u32) -> bool {
    *n == 0
}
fn is_false(b: &bool) -> bool {
    !*b
}

impl ComputeRequest {
    pub fn is_auto(&self) -> bool {
        *self == Self::default()
    }

    pub fn nodes(&self) -> u32 {
        self.nodes.max(1)
    }

    /// Short human form: `auto`, `pool=training · 2 GPUs`, `4 nodes × 8 GPUs (training)`.
    pub fn summary(&self) -> String {
        if self.is_auto() {
            return "auto".into();
        }
        let mut parts = Vec::new();
        if let Some(p) = &self.pool {
            parts.push(format!("pool {p}"));
        }
        let gpus = self.gpus.per_node();
        match (self.nodes(), gpus) {
            (1, 0) => {}
            (1, g) => parts.push(format!("{g} GPU{}", if g == 1 { "" } else { "s" })),
            (n, 0) => parts.push(format!("{n} nodes")),
            (n, g) => parts.push(format!(
                "{n} nodes × {g} GPU{}",
                if g == 1 { "" } else { "s" }
            )),
        }
        if self.gpus.is_explicit() {
            parts.push("explicit devices".into());
        }
        if let Some(r) = self.role {
            parts.push(r.to_string());
        }
        if self.gpu_memory_gb > 0 {
            parts.push(format!("≥{} GB", self.gpu_memory_gb));
        }
        if self.exclusive {
            parts.push("exclusive".into());
        }
        parts.join(" · ")
    }
}

/// A GPU-to-GPU link as `nvidia-smi topo -m` reports it (`NV4`, `PIX`, `SYS`, ...).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuLink {
    /// Index of the peer GPU on the same node.
    pub peer: u32,
    /// `NV#` (NVLink with # lanes), `PIX`/`PXB`/`PHB` (PCIe), `NODE`/`SYS` (across sockets).
    pub kind: String,
}

impl GpuLink {
    pub fn is_nvlink(&self) -> bool {
        self.kind.starts_with("NV")
    }
}

/// The devices of one node that an allocation holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeAllocation {
    pub node_id: String,
    pub hostname: String,
    /// Rank of this node in a multi-node job (0 = master).
    #[serde(default)]
    pub node_rank: u32,
    /// Device uuids, in `CUDA_VISIBLE_DEVICES` order.
    #[serde(default)]
    pub gpu_uuids: Vec<String>,
    /// The same devices as the worker's `nvidia-smi` indices.
    #[serde(default)]
    pub gpu_indices: Vec<u32>,
    /// Whether the worker of this node has received the job.
    #[serde(default)]
    pub claimed: bool,
    /// CPUs the job may use on this node (0 = unconstrained).
    #[serde(default)]
    pub cpus: u32,
}

/// Where the processes of a multi-node job find each other (torchrun `--rdzv`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rendezvous {
    pub master_addr: String,
    pub master_port: u16,
    pub nnodes: u32,
    /// Processes per node (= GPUs per node, at least 1).
    pub nproc_per_node: u32,
    /// `nnodes * nproc_per_node`.
    pub world_size: u32,
    /// torchrun `--rdzv_id`.
    pub id: String,
}

/// A reservation of devices for one holder: a job, or the server's render workers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Allocation {
    pub id: String,
    /// Job id, or `server:rendering` for the render workers.
    pub holder: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<crate::ids::JobId>,
    pub pool: String,
    pub role: Role,
    pub nodes: Vec<NodeAllocation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rendezvous: Option<Rendezvous>,
    #[serde(default)]
    pub exclusive: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Released when no heartbeat renews it before this time.
    pub lease_until: chrono::DateTime<chrono::Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub released_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub released_reason: Option<String>,
}

impl Allocation {
    pub fn gpu_count(&self) -> usize {
        self.nodes.iter().map(|n| n.gpu_uuids.len()).sum()
    }

    pub fn node(&self, node_id: &str) -> Option<&NodeAllocation> {
        self.nodes.iter().find(|n| n.node_id == node_id)
    }

    /// Environment a job on `node_id` gets: `CUDA_VISIBLE_DEVICES`, the rendezvous, NCCL and
    /// thread defaults, and `FRONTIER_ALLOCATION` (this allocation as JSON).
    pub fn env_for(&self, node_id: &str) -> BTreeMap<String, String> {
        let mut env = BTreeMap::new();
        let Some(node) = self.node(node_id) else {
            return env;
        };
        env.insert("FRONTIER_ALLOCATION_ID".into(), self.id.clone());
        env.insert("FRONTIER_POOL".into(), self.pool.clone());
        env.insert("FRONTIER_ROLE".into(), self.role.to_string());
        if !node.gpu_uuids.is_empty() {
            // CUDA accepts uuids here, which survives any reordering of indices by the driver.
            env.insert("CUDA_VISIBLE_DEVICES".into(), node.gpu_uuids.join(","));
            env.insert(
                "FRONTIER_GPU_INDICES".into(),
                node.gpu_indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        let nproc = node.gpu_uuids.len().max(1) as u32;
        if node.cpus > 0 {
            env.insert(
                "OMP_NUM_THREADS".into(),
                (node.cpus / nproc).max(1).to_string(),
            );
        } else {
            env.insert("OMP_NUM_THREADS".into(), "1".into());
        }
        if let Some(r) = &self.rendezvous {
            env.insert("MASTER_ADDR".into(), r.master_addr.clone());
            env.insert("MASTER_PORT".into(), r.master_port.to_string());
            env.insert("NNODES".into(), r.nnodes.to_string());
            env.insert("NODE_RANK".into(), node.node_rank.to_string());
            env.insert("NPROC_PER_NODE".into(), r.nproc_per_node.to_string());
            env.insert("WORLD_SIZE".into(), r.world_size.to_string());
            if r.nproc_per_node == 1 {
                // One process per node: a plain `python -m ...` can init the process group
                // from the environment without torchrun.
                env.insert("RANK".into(), node.node_rank.to_string());
                env.insert("LOCAL_RANK".into(), "0".into());
            }
            env.insert(
                "FRONTIER_TORCHRUN_ARGS".into(),
                format!(
                    "--nnodes {} --node_rank {} --nproc_per_node {} --rdzv_backend c10d --rdzv_endpoint {}:{} --rdzv_id {}",
                    r.nnodes, node.node_rank, r.nproc_per_node, r.master_addr, r.master_port, r.id
                ),
            );
            env.insert("NCCL_DEBUG".into(), "WARN".into());
            env.insert("TORCH_NCCL_ASYNC_ERROR_HANDLING".into(), "1".into());
            if r.nnodes > 1 {
                // Cross-node traffic: let NCCL pick the interface that reaches the master.
                env.insert("NCCL_SOCKET_FAMILY".into(), "AF_INET".into());
            }
        }
        if let Ok(json) = serde_json::to_string(self) {
            env.insert("FRONTIER_ALLOCATION".into(), json);
        }
        env
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_parse_and_match() {
        assert_eq!("train".parse::<Role>().unwrap(), Role::Training);
        assert_eq!("auto".parse::<Role>().unwrap(), Role::Any);
        assert!("gpu".parse::<Role>().is_err());
        assert!(Role::Training.served_by(&[Role::Any]));
        assert!(Role::Training.served_by(&[Role::Inference, Role::Training]));
        assert!(!Role::Rendering.served_by(&[Role::Training]));
        assert!(Role::Any.served_by(&[Role::Training]));
    }

    #[test]
    fn request_serialises_sparsely() {
        let r = ComputeRequest::default();
        assert_eq!(serde_json::to_string(&r).unwrap(), "{}");
        assert_eq!(r.summary(), "auto");
        let r: ComputeRequest =
            serde_json::from_str(r#"{"pool":"training","gpus":2,"nodes":4}"#).unwrap();
        assert_eq!(r.gpus, GpuRequest::Count(2));
        assert_eq!(r.summary(), "pool training · 4 nodes × 2 GPUs");
        let r: ComputeRequest = serde_json::from_str(r#"{"gpus":["GPU-a","GPU-b"]}"#).unwrap();
        assert!(r.gpus.is_explicit());
        assert_eq!(r.gpus.per_node(), 2);
    }

    #[test]
    fn env_for_a_node() {
        let now = chrono::Utc::now();
        let a = Allocation {
            id: "alloc-1".into(),
            holder: "job".into(),
            job_id: None,
            pool: "auto".into(),
            role: Role::Training,
            nodes: vec![
                NodeAllocation {
                    node_id: "n0".into(),
                    hostname: "h0".into(),
                    node_rank: 0,
                    gpu_uuids: vec!["GPU-a".into(), "GPU-b".into()],
                    gpu_indices: vec![0, 1],
                    claimed: false,
                    cpus: 16,
                },
                NodeAllocation {
                    node_id: "n1".into(),
                    hostname: "h1".into(),
                    node_rank: 1,
                    gpu_uuids: vec!["GPU-c".into(), "GPU-d".into()],
                    gpu_indices: vec![0, 1],
                    claimed: false,
                    cpus: 16,
                },
            ],
            rendezvous: Some(Rendezvous {
                master_addr: "h0".into(),
                master_port: 29500,
                nnodes: 2,
                nproc_per_node: 2,
                world_size: 4,
                id: "alloc-1".into(),
            }),
            exclusive: false,
            created_at: now,
            lease_until: now,
            released_at: None,
            released_reason: None,
        };
        let env = a.env_for("n1");
        assert_eq!(env["CUDA_VISIBLE_DEVICES"], "GPU-c,GPU-d");
        assert_eq!(env["NODE_RANK"], "1");
        assert_eq!(env["WORLD_SIZE"], "4");
        assert_eq!(env["OMP_NUM_THREADS"], "8");
        assert!(
            !env.contains_key("RANK"),
            "torchrun assigns ranks when nproc > 1"
        );
        assert!(env["FRONTIER_TORCHRUN_ARGS"].contains("--node_rank 1"));
        assert!(env["FRONTIER_ALLOCATION"].contains("\"alloc-1\""));
        assert!(a.env_for("nope").is_empty());
    }
}
