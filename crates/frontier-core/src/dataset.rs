// SPDX-License-Identifier: Apache-2.0
//! `frontier-dataset` manifest: v1 (docs/spec/dataset-format.md, "Dataset layout") and the additive
//! v1.1 keys (docs/spec/dataset-format.md section 1: `minor`, `sample_kind`, `temporal`, `time`,
//! `domain`, `license`, `citation`, `hub_id`). The v1.1 keys live in `extra`, so a v1 reader
//! round-trips them untouched; the accessors below apply the v1 defaults.

use crate::ids::DatasetId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DATASET_FORMAT: &str = "frontier-dataset";
pub const DATASET_FORMAT_VERSION: u32 = 1;

/// One entry of `manifest.json#samples`. `path` is relative to the dataset root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleEntry {
    pub id: String,
    pub path: String,
    /// Any additional per-sample keys the ingest wrote (conditions, tags, ...).
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// `manifest.json` of a dataset directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub source: serde_json::Value,
    #[serde(default)]
    pub samples: Vec<SampleEntry>,
    #[serde(default)]
    pub splits: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub fields: Vec<serde_json::Value>,
    #[serde(default)]
    pub conditions: Vec<serde_json::Value>,
    /// Anything else the manifest carries (description, license, ...).
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl DatasetManifest {
    fn extra_str(&self, key: &str) -> Option<&str> {
        self.extra.get(key).and_then(|v| v.as_str())
    }

    /// Minor format version (`0` for v1 manifests).
    pub fn minor(&self) -> u32 {
        self.extra
            .get("minor")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
    }

    /// `surface | mesh | grid | point_cloud`; v1 manifests are surfaces.
    pub fn sample_kind(&self) -> &str {
        self.extra_str("sample_kind").unwrap_or("surface")
    }

    /// `steady | transient`; v1 manifests are steady.
    pub fn temporal(&self) -> &str {
        self.extra_str("temporal").unwrap_or("steady")
    }

    pub fn is_transient(&self) -> bool {
        self.temporal() == "transient"
    }

    /// Number of time steps declared in `time.n_steps` (1 for steady data).
    pub fn n_steps(&self) -> u64 {
        self.extra
            .get("time")
            .and_then(|t| t.get("n_steps"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
    }

    /// `cfd | structural | thermal | multiphysics | materials | geophysical | other`.
    pub fn domain(&self) -> Option<&str> {
        self.extra_str("domain")
    }

    /// SPDX id (a plain string in the manifest, or `{spdx: ...}` as in the hub catalog).
    pub fn license(&self) -> Option<&str> {
        match self.extra.get("license") {
            Some(serde_json::Value::String(s)) => Some(s.as_str()),
            Some(v) => v.get("spdx").and_then(|s| s.as_str()),
            None => None,
        }
    }

    pub fn citation(&self) -> Option<&str> {
        self.extra_str("citation")
    }

    /// Id of the hub catalog entry this dataset was pulled from.
    pub fn hub_id(&self) -> Option<&str> {
        self.extra_str("hub_id")
    }
}

fn default_format() -> String {
    DATASET_FORMAT.to_string()
}

fn default_version() -> u32 {
    DATASET_FORMAT_VERSION
}

/// A registered dataset: a manifest plus where it lives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub id: DatasetId,
    pub name: String,
    /// As given at registration: a local path, `file://...`, `s3://...`.
    pub uri: String,
    pub created_at: DateTime<Utc>,
    pub manifest: DatasetManifest,
}

impl Dataset {
    pub fn n_samples(&self) -> usize {
        self.manifest.samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_manifest_gets_v1_defaults() {
        let m: DatasetManifest =
            serde_json::from_str(r#"{"format":"frontier-dataset","version":1,"name":"a"}"#)
                .unwrap();
        assert_eq!(m.minor(), 0);
        assert_eq!(m.sample_kind(), "surface");
        assert_eq!(m.temporal(), "steady");
        assert_eq!(m.n_steps(), 1);
        assert!(m.domain().is_none() && m.license().is_none() && m.hub_id().is_none());
    }

    #[test]
    fn v1_1_keys_are_read_and_round_trip() {
        let text = r#"{"format":"frontier-dataset","version":1,"minor":1,"name":"karman",
            "sample_kind":"grid","temporal":"transient",
            "time":{"n_steps":100,"dt":0.01,"units":"s","uniform":true},
            "domain":"cfd","license":"CC-BY-4.0","citation":"Doe 2026","hub_id":"karman-2d"}"#;
        let m: DatasetManifest = serde_json::from_str(text).unwrap();
        assert_eq!((m.minor(), m.sample_kind(), m.n_steps()), (1, "grid", 100));
        assert!(m.is_transient());
        assert_eq!(m.domain(), Some("cfd"));
        assert_eq!(m.license(), Some("CC-BY-4.0"));
        assert_eq!(m.hub_id(), Some("karman-2d"));
        let back = serde_json::to_value(&m).unwrap();
        assert_eq!(back["time"]["dt"], 0.01);
        assert_eq!(back["sample_kind"], "grid");

        let spdx: DatasetManifest =
            serde_json::from_str(r#"{"license":{"spdx":"MIT","commercial_use":true}}"#).unwrap();
        assert_eq!(spdx.license(), Some("MIT"));
    }
}
