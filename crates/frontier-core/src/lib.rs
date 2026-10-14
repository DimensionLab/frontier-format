// SPDX-License-Identifier: Apache-2.0
//! Core domain model for DimensionLab Frontier.
//!
//! This crate is intentionally small and dependency-light. Everything else in the
//! platform (connectors, visualization, telemetry, the worker, the server, the UI)
//! builds on the types and traits defined here. Keep it that way: the "strong minimal
//! core" is what lets every other product in the monorepo stand on its own.

pub mod artifact;
pub mod compute;
pub mod dataset;
pub mod error;
pub mod ids;
pub mod job;
pub mod joblog;
pub mod node;
pub mod notification;
pub mod plugin;
pub mod prep;
pub mod project;
pub mod run;

pub use artifact::Artifact;
pub use compute::{
    Allocation, ComputeRequest, GpuLink, GpuRequest, NodeAllocation, Rendezvous, Role,
};
pub use dataset::{Dataset, DatasetManifest, SampleEntry};
pub use error::{Error, Result};
pub use ids::{ArtifactId, DatasetId, JobId, ProjectId, RunId};
pub use job::{Job, JobSpec, JobState, JobStatusUpdate, Mount, Resources};
pub use node::{GpuInfo, HeartbeatResponse, NodeInfo};
pub use plugin::{Plugin, PluginKind, PluginManifest};
pub use prep::{
    AgentSession, ApprovalDecision, ApprovalRecord, ApprovalRequest, Assertion, LineageEntry,
    PrepMode, PrepRequest, ProvenanceRecord, SessionEvent, SessionSummary, VersionRecord,
    VersionStore,
};
pub use project::Project;
pub use run::{Metric, MetricRecord, MetricValue, Run, RunState, Step};

/// Semantic version of the core data model. Bump on any breaking change to the
/// wire format produced by `frontier-data` or consumed by the Python SDK.
pub const DATA_MODEL_VERSION: u32 = 1;
