// SPDX-License-Identifier: Apache-2.0

use crate::ids::ProjectId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A project groups runs (one dataset family / one modelling effort).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: ProjectId,
    pub name: String,
    pub created_at: DateTime<Utc>,
}

impl Project {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: ProjectId::new(),
            name: name.into(),
            created_at: Utc::now(),
        }
    }
}
