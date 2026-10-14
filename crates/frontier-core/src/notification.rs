// SPDX-License-Identifier: Apache-2.0
//! Notifications: what the server tells a person about things that happened while they
//! were looking elsewhere, and what the app shows as toasts and in its notification
//! centre. One vocabulary for both sides (`docs/spec/api.md`, "Notifications").

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// How much a notification asks of the reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    #[default]
    Info,
    Success,
    Warning,
    Error,
}

impl Level {
    pub fn as_str(self) -> &'static str {
        match self {
            Level::Info => "info",
            Level::Success => "success",
            Level::Warning => "warning",
            Level::Error => "error",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "info" => Level::Info,
            "success" => Level::Success,
            "warning" => Level::Warning,
            "error" => Level::Error,
            _ => return None,
        })
    }

    /// Whether the reader has to act or at least know: such a toast stays until dismissed
    /// and is announced at once.
    pub fn is_urgent(self) -> bool {
        matches!(self, Level::Warning | Level::Error)
    }
}

/// What happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotificationKind {
    /// A board's pipeline run ended with every step done.
    PipelineRunFinished,
    /// A board's pipeline run ended with a failed step.
    PipelineRunFailed,
    /// One step of a pipeline run failed (`body` is its one-line reason).
    StepFailed,
    /// A training run finished.
    RunFinished,
    /// A training run failed.
    RunFailed,
    /// A job that is not a step of a pipeline run failed.
    JobFailed,
    /// A job has been queued for longer than is usual.
    JobQueuedLong,
    /// A dataset import or a data-preparation step finished: the dataset can be used.
    DatasetReady,
    /// A quality gate refused a model.
    GateFailed,
    /// An evaluation of a model finished.
    EvaluationReady,
    /// A report was written.
    ReportReady,
    /// An inference wrote its predictions.
    PredictionsReady,
    /// An inference failed (`body` is its one-line reason).
    InferenceFailed,
    /// A model version moved to another stage.
    ModelPromoted,
    /// A model version was deployed.
    ModelDeployed,
    /// A data-preparation step or an AI agent's plan waits for a person's decision.
    ApprovalRequested,
    ApprovalGranted,
    ApprovalRefused,
    /// A worker stopped answering while it held work.
    WorkerOffline,
    /// A pool has no free capacity for a queued job.
    PoolNoCapacity,
    /// A cloud pod is about to be released because nothing uses it.
    PodIdleRelease,
    /// Spending reached a configured threshold.
    CostThreshold,
    /// A stored credential expires soon.
    CredentialExpiring,
    /// A stored credential was tested and refused.
    CredentialTestFailed,
    /// The app lost its server (made by the app itself).
    ConnectionLost,
    /// The app reached its server again (made by the app itself).
    ConnectionRestored,
    /// A newer version of Frontier is available.
    UpdateAvailable,
}

impl NotificationKind {
    pub const ALL: &'static [NotificationKind] = &[
        Self::PipelineRunFinished,
        Self::PipelineRunFailed,
        Self::StepFailed,
        Self::RunFinished,
        Self::RunFailed,
        Self::JobFailed,
        Self::JobQueuedLong,
        Self::DatasetReady,
        Self::GateFailed,
        Self::EvaluationReady,
        Self::ReportReady,
        Self::PredictionsReady,
        Self::InferenceFailed,
        Self::ModelPromoted,
        Self::ModelDeployed,
        Self::ApprovalRequested,
        Self::ApprovalGranted,
        Self::ApprovalRefused,
        Self::WorkerOffline,
        Self::PoolNoCapacity,
        Self::PodIdleRelease,
        Self::CostThreshold,
        Self::CredentialExpiring,
        Self::CredentialTestFailed,
        Self::ConnectionLost,
        Self::ConnectionRestored,
        Self::UpdateAvailable,
    ];

    /// The wire name (`step_failed`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PipelineRunFinished => "pipeline_run_finished",
            Self::PipelineRunFailed => "pipeline_run_failed",
            Self::StepFailed => "step_failed",
            Self::RunFinished => "run_finished",
            Self::RunFailed => "run_failed",
            Self::JobFailed => "job_failed",
            Self::JobQueuedLong => "job_queued_long",
            Self::DatasetReady => "dataset_ready",
            Self::GateFailed => "gate_failed",
            Self::EvaluationReady => "evaluation_ready",
            Self::ReportReady => "report_ready",
            Self::PredictionsReady => "predictions_ready",
            Self::InferenceFailed => "inference_failed",
            Self::ModelPromoted => "model_promoted",
            Self::ModelDeployed => "model_deployed",
            Self::ApprovalRequested => "approval_requested",
            Self::ApprovalGranted => "approval_granted",
            Self::ApprovalRefused => "approval_refused",
            Self::WorkerOffline => "worker_offline",
            Self::PoolNoCapacity => "pool_no_capacity",
            Self::PodIdleRelease => "pod_idle_release",
            Self::CostThreshold => "cost_threshold",
            Self::CredentialExpiring => "credential_expiring",
            Self::CredentialTestFailed => "credential_test_failed",
            Self::ConnectionLost => "connection_lost",
            Self::ConnectionRestored => "connection_restored",
            Self::UpdateAvailable => "update_available",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|k| k.as_str() == s)
    }

    /// The group a person switches on or off (`notifications.types.<group>`).
    pub fn group(self) -> &'static str {
        match self {
            Self::PipelineRunFinished
            | Self::PipelineRunFailed
            | Self::RunFinished
            | Self::RunFailed => "runs",
            Self::StepFailed => "steps",
            Self::JobFailed | Self::JobQueuedLong => "jobs",
            Self::DatasetReady => "datasets",
            Self::GateFailed
            | Self::EvaluationReady
            | Self::ReportReady
            | Self::PredictionsReady
            | Self::InferenceFailed => "evaluations",
            Self::ModelPromoted | Self::ModelDeployed => "models",
            Self::ApprovalRequested | Self::ApprovalGranted | Self::ApprovalRefused => "approvals",
            Self::WorkerOffline | Self::PoolNoCapacity | Self::PodIdleRelease => "compute",
            Self::CostThreshold => "cost",
            Self::CredentialExpiring | Self::CredentialTestFailed => "credentials",
            Self::ConnectionLost | Self::ConnectionRestored => "connection",
            Self::UpdateAvailable => "updates",
        }
    }

    /// The level a notification of this kind has unless its maker says otherwise.
    pub fn level(self) -> Level {
        match self {
            Self::PipelineRunFinished
            | Self::RunFinished
            | Self::DatasetReady
            | Self::EvaluationReady
            | Self::ReportReady
            | Self::PredictionsReady
            | Self::ModelPromoted
            | Self::ModelDeployed
            | Self::ApprovalGranted
            | Self::ConnectionRestored => Level::Success,
            Self::PipelineRunFailed
            | Self::InferenceFailed
            | Self::StepFailed
            | Self::RunFailed
            | Self::JobFailed
            | Self::GateFailed
            | Self::CredentialTestFailed
            | Self::ConnectionLost => Level::Error,
            Self::JobQueuedLong
            | Self::ApprovalRequested
            | Self::ApprovalRefused
            | Self::WorkerOffline
            | Self::PoolNoCapacity
            | Self::PodIdleRelease
            | Self::CostThreshold
            | Self::CredentialExpiring => Level::Warning,
            Self::UpdateAvailable => Level::Info,
        }
    }
}

/// The groups of [`NotificationKind::group`], in the order settings list them.
pub const GROUPS: &[&str] = &[
    "runs",
    "steps",
    "jobs",
    "datasets",
    "evaluations",
    "models",
    "approvals",
    "compute",
    "cost",
    "credentials",
    "connection",
    "updates",
];

/// The thing a notification is about, so that the app can go to it. `kind` is one of
/// `pipeline_run`, `step`, `run`, `job`, `dataset`, `project`, `approval`, `model`,
/// `worker`, `credential`, `server`; the other fields depend on it (`board`, `run`, `job`,
/// `step`, `session`, `version` ...).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Subject {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id: String,
    #[serde(flatten)]
    pub more: BTreeMap<String, String>,
}

impl Subject {
    pub fn new(kind: &str, id: impl Into<String>) -> Self {
        Self {
            kind: kind.to_string(),
            id: id.into(),
            more: BTreeMap::new(),
        }
    }

    pub fn with(mut self, key: &str, value: impl Into<String>) -> Self {
        let value = value.into();
        if !value.is_empty() {
            self.more.insert(key.to_string(), value);
        }
        self
    }

    pub fn get(&self, key: &str) -> &str {
        self.more.get(key).map(String::as_str).unwrap_or("")
    }
}

/// One notification of one person.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Notification {
    /// What `POST /notifications/read` takes. The server's are the sequence number as
    /// text; the app's own start with `local-`.
    pub id: String,
    /// Position in the person's stream: larger is newer. `0` for the app's own.
    #[serde(default)]
    pub seq: u64,
    pub kind: NotificationKind,
    pub level: Level,
    /// What happened, in sentence case, without ids or paths.
    pub title: String,
    /// One short line more: the reason, or what to do.
    #[serde(default)]
    pub body: String,
    /// Project id, when the subject belongs to one.
    #[serde(default)]
    pub project: Option<String>,
    pub subject: Subject,
    /// Login name of who caused it (who submitted the run), when a person did.
    #[serde(default)]
    pub actor: Option<String>,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub read: bool,
}

impl Notification {
    /// When it was made, as milliseconds since the Unix epoch.
    pub fn created_ms(&self) -> i64 {
        self.created_at.timestamp_millis()
    }
}

/// A time from milliseconds since the Unix epoch, for a client that makes notifications of
/// its own from its own clock (the epoch itself for a value out of range).
pub fn time_from_ms(ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(ms).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_have_stable_names_and_a_known_group() {
        for kind in NotificationKind::ALL {
            let name = serde_json::to_value(kind).unwrap();
            assert_eq!(name, kind.as_str(), "serde and as_str disagree");
            assert_eq!(NotificationKind::parse(kind.as_str()), Some(*kind));
            assert!(GROUPS.contains(&kind.group()), "{kind:?}");
        }
        for group in GROUPS {
            assert!(
                NotificationKind::ALL.iter().any(|k| k.group() == *group),
                "no kind in {group}"
            );
        }
        assert_eq!(NotificationKind::parse("nothing"), None);
    }

    #[test]
    fn a_notification_round_trips_with_its_subject_fields_flat() {
        let n = Notification {
            id: "7".into(),
            seq: 7,
            kind: NotificationKind::StepFailed,
            level: NotificationKind::StepFailed.level(),
            title: "Step failed: train".into(),
            body: "CUDA out of memory".into(),
            project: Some("p1".into()),
            subject: Subject::new("step", "train")
                .with("run", "r1")
                .with("job", "j1")
                .with("board", ""),
            actor: Some("ada".into()),
            created_at: DateTime::parse_from_rfc3339("2026-10-12T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            read: false,
        };
        let v = serde_json::to_value(&n).unwrap();
        assert_eq!(v["kind"], "step_failed");
        assert_eq!(v["level"], "error");
        assert_eq!(v["subject"]["type"], "step");
        assert_eq!(v["subject"]["job"], "j1");
        assert!(v["subject"].get("board").is_none(), "empty fields are left out");
        assert_eq!(serde_json::from_value::<Notification>(v).unwrap(), n);
        assert!(Level::Error.is_urgent() && !Level::Success.is_urgent());
        assert_eq!(Level::parse("warning"), Some(Level::Warning));
        assert_eq!(time_from_ms(n.created_ms()), n.created_at);
    }
}
