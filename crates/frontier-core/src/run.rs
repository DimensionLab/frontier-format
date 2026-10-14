// SPDX-License-Identifier: Apache-2.0

use crate::ids::{ProjectId, RunId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Training step index. Monotonic within a run.
pub type Step = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Pending,
    Running,
    Finished,
    Failed,
    Killed,
}

/// A training / fine-tuning / evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: RunId,
    pub project: ProjectId,
    pub name: String,
    pub state: RunState,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub config: serde_json::Value,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Final summary written by the SDK at `finish()` (best metrics, wall time, ...).
    #[serde(default)]
    pub summary: serde_json::Value,
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,
}

impl Run {
    pub fn new(project: ProjectId, name: impl Into<String>) -> Self {
        Self {
            id: RunId::new(),
            project,
            name: name.into(),
            state: RunState::Pending,
            created_at: Utc::now(),
            config: serde_json::Value::Null,
            tags: Vec::new(),
            summary: serde_json::Value::Null,
            updated_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetricValue {
    Float(f64),
    Int(i64),
    Text(String),
}

/// One logged scalar at one step. Physics-specific metrics (drag/lift coefficients,
/// field L2 errors, spectra) are logged with namespaced keys like `cfd/cd_rel_err`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub run: RunId,
    pub step: Step,
    pub timestamp: DateTime<Utc>,
    pub values: BTreeMap<String, MetricValue>,
}

/// Wire format of one metric record as posted by the SDK to
/// `POST /api/v1/runs/{id}/metrics`: `{step, timestamp?, values:{key:number}}`.
/// The run is implied by the URL; a `run` key in the body is ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    pub step: Step,
    #[serde(default)]
    pub timestamp: Option<DateTime<Utc>>,
    pub values: BTreeMap<String, MetricValue>,
}

impl MetricRecord {
    pub fn into_metric(self, run: RunId) -> Metric {
        Metric {
            run,
            step: self.step,
            timestamp: self.timestamp.unwrap_or_else(Utc::now),
            values: self.values,
        }
    }
}

impl From<Metric> for MetricRecord {
    fn from(m: Metric) -> Self {
        Self {
            step: m.step,
            timestamp: Some(m.timestamp),
            values: m.values,
        }
    }
}

impl RunState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Finished => "finished",
            Self::Failed => "failed",
            Self::Killed => "killed",
        }
    }
}

impl std::str::FromStr for RunState {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "running" => Ok(Self::Running),
            "finished" => Ok(Self::Finished),
            "failed" => Ok(Self::Failed),
            "killed" => Ok(Self::Killed),
            other => Err(format!("unknown run state `{other}`")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_roundtrips_through_json() {
        let run = Run::new(ProjectId::new(), "baseline-transolver");
        let json = serde_json::to_string(&run).unwrap();
        let back: Run = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, run.id);
        assert_eq!(back.state, RunState::Pending);
    }

    #[test]
    fn metric_record_ignores_run_and_missing_timestamp() {
        let rec: MetricRecord =
            serde_json::from_str(r#"{"run":"local-abc","step":3,"values":{"train/loss":0.5}}"#)
                .unwrap();
        assert_eq!(rec.step, 3);
        assert!(rec.timestamp.is_none());
        let m = rec.into_metric(RunId::new());
        assert_eq!(m.values["train/loss"], MetricValue::Float(0.5));
    }
}
