// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A file attached to a run (checkpoint, report, log). `url` is server-relative, e.g.
/// `/api/v1/runs/<id>/artifacts/<name>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    pub size: u64,
    pub url: String,
    #[serde(default)]
    pub content_type: Option<String>,
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
}
