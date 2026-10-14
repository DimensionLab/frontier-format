// SPDX-License-Identifier: Apache-2.0
//! Job specs and job records (see docs/spec/api.md, "JobSpec").

use crate::compute::{Allocation, ComputeRequest};
use crate::ids::{JobId, RunId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    /// Whether moving from `self` to `next` is a legal state-machine step
    /// (`queued -> running -> succeeded|failed`, cancel allowed from any non-terminal state).
    pub fn can_transition_to(self, next: JobState) -> bool {
        use JobState::*;
        match (self, next) {
            (a, b) if a == b => true,
            (Queued, Running) | (Queued, Cancelled) => true,
            (Running, Succeeded) | (Running, Failed) | (Running, Cancelled) => true,
            // a worker may report a terminal state without an explicit `running` first
            (Queued, Succeeded) | (Queued, Failed) => true,
            _ => false,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl std::str::FromStr for JobState {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "queued" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "succeeded" => Ok(Self::Succeeded),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            other => Err(format!("unknown job state `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Resources {
    #[serde(default)]
    pub gpus: u32,
    #[serde(default = "default_gpu_type")]
    pub gpu_type: String,
    #[serde(default)]
    pub cpus: u32,
    #[serde(default)]
    pub memory_gb: u32,
}

fn default_gpu_type() -> String {
    "any".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mount {
    pub uri: String,
    pub path: String,
    #[serde(default)]
    pub read_only: bool,
}

/// What to run. Sent by clients to `POST /api/v1/jobs` and handed to workers by
/// `POST /api/v1/workers/heartbeat`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobSpec {
    /// Assigned by the server on submission; clients may leave it out.
    #[serde(default)]
    pub id: JobId,
    pub name: String,
    /// Container image. `None` means run `command` as a bare process on the worker host.
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub resources: Resources,
    #[serde(default)]
    pub mounts: Vec<Mount>,
    /// `local | runpod | anyscale | slurm | k8s`. Workers only claim jobs for their own backend.
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Required worker labels (`gpu=h100`, `site=eu`); a job is only claimed by a worker
    /// that advertises every one of them.
    #[serde(default)]
    pub labels: Vec<String>,
    /// Working directory inside the container / on the host (defaults to the worker's).
    #[serde(default)]
    pub workdir: Option<String>,
    /// Run this job will log to, if known up front.
    #[serde(default)]
    pub run_id: Option<RunId>,
    /// Pool, role, GPUs and nodes (docs/spec/api.md, JobSpec); `{}` = automatic.
    #[serde(default, skip_serializing_if = "ComputeRequest::is_auto")]
    pub compute: ComputeRequest,
    /// Filled in by the control plane when a worker claims the job: the devices it holds and
    /// the rendezvous for multi-node jobs. Never set by clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocation: Option<Allocation>,
}

fn default_backend() -> String {
    "local".to_string()
}

impl JobSpec {
    pub fn new(name: impl Into<String>, command: Vec<String>) -> Self {
        Self {
            id: JobId::new(),
            name: name.into(),
            image: None,
            command,
            env: BTreeMap::new(),
            resources: Resources::default(),
            mounts: Vec::new(),
            backend: default_backend(),
            labels: Vec::new(),
            workdir: None,
            run_id: None,
            compute: ComputeRequest::default(),
            allocation: None,
        }
    }
}

/// Server-side record of a submitted job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: JobId,
    pub spec: JobSpec,
    pub state: JobState,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub finished_at: Option<DateTime<Utc>>,
    /// Hostname of the worker that claimed the job.
    #[serde(default)]
    pub worker: Option<String>,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Body of `POST /api/v1/jobs/{id}/status`. `log` accepts a single string or a list of lines.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JobStatusUpdate {
    #[serde(default)]
    pub state: Option<JobState>,
    #[serde(default, deserialize_with = "deserialize_log_lines")]
    pub log: Vec<String>,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub error: Option<String>,
}

fn deserialize_log_lines<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Lines {
        One(String),
        Many(Vec<String>),
        Missing(()),
    }
    Ok(match Lines::deserialize(d)? {
        Lines::One(s) => s.lines().map(str::to_owned).collect(),
        Lines::Many(v) => v,
        Lines::Missing(()) => Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_parses_contract_example() {
        let json = r#"{"name":"train-ahmed","image":"registry.example.com/frontier-trainer:1.0",
            "command":["python","-m","frontier_train"],"env":{"FRONTIER_RUN_ID":"x"},
            "resources":{"gpus":1,"gpu_type":"any","cpus":8,"memory_gb":32},
            "mounts":[{"uri":"s3://b/dataset","path":"/data"}],"backend":"runpod"}"#;
        let spec: JobSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.resources.gpus, 1);
        assert_eq!(spec.backend, "runpod");
        assert_eq!(spec.mounts[0].path, "/data");
    }

    #[test]
    fn status_log_accepts_string_or_list() {
        let a: JobStatusUpdate =
            serde_json::from_str(r#"{"state":"running","log":"a\nb"}"#).unwrap();
        assert_eq!(a.log, vec!["a", "b"]);
        let b: JobStatusUpdate = serde_json::from_str(r#"{"log":["x"]}"#).unwrap();
        assert_eq!(b.log, vec!["x"]);
        let c: JobStatusUpdate = serde_json::from_str(r#"{"state":"failed"}"#).unwrap();
        assert!(c.log.is_empty());
    }

    #[test]
    fn transitions() {
        assert!(JobState::Queued.can_transition_to(JobState::Running));
        assert!(JobState::Running.can_transition_to(JobState::Failed));
        assert!(!JobState::Succeeded.can_transition_to(JobState::Running));
        assert!(!JobState::Running.can_transition_to(JobState::Queued));
    }
}
