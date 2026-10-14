// SPDX-License-Identifier: Apache-2.0
//! Data preparation: dataset versions, provenance, assertions, approvals and agent sessions
//! (`docs/spec/data-prep.md` sections 3 to 5 and 8; session records are the harness's
//! `<id>.jsonl` files).
//!
//! [`VersionStore`] is a read-only view of the `.frontier/` directory `frontier_data.versions`
//! writes (JSON files plus hardlinked objects, no database): the control plane and the CLI
//! read logs, provenance, assertions and lineage from it directly. Writes (commit, revert,
//! approve, diff with array statistics) stay with the Python implementation, which owns the
//! content-addressed ids.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The directory inside a dataset that holds its versions.
pub const STORE_DIR: &str = ".frontier";

/// `plan`, `preview` or `run` of an op (`docs/spec/data-prep.md` section 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrepMode {
    Plan,
    Preview,
    Run,
}

impl PrepMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::Preview => "preview",
            Self::Run => "run",
        }
    }
}

impl std::str::FromStr for PrepMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "plan" => Ok(Self::Plan),
            "preview" => Ok(Self::Preview),
            "run" => Ok(Self::Run),
            other => Err(format!("unknown prep mode `{other}` (plan, preview, run)")),
        }
    }
}

/// A person's decision on a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    #[default]
    Approve,
    Reject,
}

impl ApprovalDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Approve => "approve",
            Self::Reject => "reject",
        }
    }
}

/// What a client sends to approve (or reject) a plan: `POST /api/v1/prep/{op}/run` carries
/// one, `POST /api/v1/agent/sessions/{id}/approvals/{plan_id}` creates one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ApprovalRequest {
    /// `plan_id` of the plan being approved (hash of op, op version, params and input
    /// version); optional for a run submitted by a person who did not plan first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<String>,
    /// Reviewer identity; the `X-Frontier-Actor` header fills it in when absent.
    #[serde(default)]
    pub approved_by: String,
    #[serde(default)]
    pub decision: ApprovalDecision,
    #[serde(default)]
    pub note: String,
}

/// An approval as the control plane records it (audit table and session approvals).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRecord {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<String>,
    pub decision: ApprovalDecision,
    pub reviewer: String,
    #[serde(default)]
    pub note: String,
    pub at: chrono::DateTime<chrono::Utc>,
    /// `HMAC(server secret, session_id:plan_id)`, minted on approval only; what the harness
    /// passes to `<op>.run` as `approval_token`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

/// Body of `POST /api/v1/prep/{op}/plan|preview|run`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PrepRequest {
    /// Dataset id (registered) or name.
    pub dataset: String,
    #[serde(default)]
    pub params: Value,
    /// Preview sample selection (`{"max_samples": 2, "ids": [...]}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Required for `run` unless `policy.ai.autonomous_op_classes` covers the op.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval: Option<ApprovalRequest>,
    /// Compute pool to run in (`{}` = automatic, role `prep`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pool: Option<String>,
}

/// One file of a version tree: content hash and size.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeEntry(pub String, pub u64);

/// The tree of a version: dataset-level files and one subtree hash per sample.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct VersionTree {
    #[serde(default)]
    pub files: BTreeMap<String, TreeEntry>,
    #[serde(default)]
    pub samples: BTreeMap<String, String>,
}

/// `versions/<id>.json` as written by `frontier_data.versions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VersionFile {
    pub id: String,
    #[serde(default)]
    pub parent: Option<String>,
    pub created_at: String,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub actor: Value,
    #[serde(default)]
    pub op_record: Option<String>,
    #[serde(default)]
    pub stats: Value,
    #[serde(default)]
    pub tree: VersionTree,
}

/// A version as the API and the CLI show it: the record without its tree, plus tags and
/// the op that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VersionRecord {
    pub id: String,
    #[serde(default)]
    pub parent: Option<String>,
    pub created_at: String,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub actor: Value,
    #[serde(default)]
    pub op_record: Option<String>,
    /// Op id of the producing record (`prep.fields.map`), when there is one.
    #[serde(default)]
    pub op: Option<String>,
    #[serde(default)]
    pub stats: Value,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub n_samples: usize,
}

/// One line of `provenance.jsonl` (`docs/spec/data-prep.md` section 4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub at: String,
    #[serde(default)]
    pub actor: Value,
    #[serde(flatten)]
    pub data: Map<String, Value>,
}

impl ProvenanceRecord {
    pub fn str(&self, key: &str) -> Option<&str> {
        self.data.get(key).and_then(Value::as_str)
    }
}

/// A quality assertion attached to a version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assertion {
    pub id: String,
    pub version: String,
    pub check: String,
    pub status: String,
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub detail: Value,
    #[serde(default)]
    pub op_record: Option<String>,
    #[serde(default)]
    pub at: String,
}

/// A version in which a target changed (`lineage`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineageEntry {
    pub version: String,
    pub created_at: String,
    pub message: String,
    pub op: Option<String>,
    pub op_record: Option<String>,
    pub params: Option<Value>,
    pub n_files: usize,
    pub added: usize,
    pub removed: usize,
    pub modified: usize,
}

/// Provenance as a graph: versions and op records are nodes; `parent` and
/// `input_version -> output_version` are edges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProvenanceGraph {
    pub nodes: Vec<Value>,
    pub edges: Vec<Value>,
    pub head: Option<String>,
    pub tags: BTreeMap<String, String>,
}

/// A recorded agent session as the control plane tracks it (the record itself is the
/// `.jsonl` file the harness writes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSession {
    pub id: String,
    #[serde(default)]
    pub dataset: Option<String>,
    #[serde(default)]
    pub dataset_root: Option<String>,
    pub task: String,
    pub provider: String,
    pub model: String,
    pub harness: String,
    /// `queued | running | succeeded | failed | cancelled` (the job's state), or `replay`.
    pub state: String,
    #[serde(default)]
    pub job_id: Option<String>,
    pub record_path: String,
    pub created_by: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// One event of a session record (`session_start`, `tool_call`, `approval_request`, ...).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEvent {
    pub seq: u64,
    #[serde(default)]
    pub t: f64,
    pub kind: String,
    #[serde(flatten)]
    pub data: Map<String, Value>,
}

/// Counts and outcomes of a session record (the shape `frontier_agent.record.summary` prints).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SessionSummary {
    pub id: String,
    pub task: Option<String>,
    pub model: Option<Value>,
    pub harness: Option<String>,
    pub tool_calls: usize,
    pub tools_used: Vec<String>,
    pub approvals_requested: usize,
    pub approvals_granted: usize,
    pub ops_run: Vec<String>,
    pub judgments: usize,
    pub errors: usize,
    pub outcome: Option<String>,
    pub final_text: Option<String>,
    pub usage: Option<Value>,
    pub duration_s: Option<f64>,
    pub n_events: usize,
}

/// Read a session record file: one JSON object per line, blank lines ignored, a bad line
/// ends the read (the harness may be mid-write).
pub fn read_session_events(path: &Path) -> std::io::Result<Vec<SessionEvent>> {
    let text = std::fs::read_to_string(path)?;
    Ok(parse_session_events(&text))
}

pub fn parse_session_events(text: &str) -> Vec<SessionEvent> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map_while(|l| serde_json::from_str::<SessionEvent>(l).ok())
        .collect()
}

pub fn summarize_session(id: &str, events: &[SessionEvent]) -> SessionSummary {
    let of = |kind: &'static str| events.iter().filter(move |e| e.kind == kind);
    let start = of("session_start").next();
    let end = of("session_end").next();
    let get = |e: Option<&SessionEvent>, key: &str| e.and_then(|e| e.data.get(key).cloned());
    let calls: Vec<&SessionEvent> = of("tool_call").collect();
    SessionSummary {
        id: id.to_string(),
        task: get(start, "task").and_then(|v| v.as_str().map(str::to_string)),
        model: get(start, "model"),
        harness: get(start, "harness").and_then(|v| v.as_str().map(str::to_string)),
        tool_calls: calls.len(),
        tools_used: calls
            .iter()
            .filter_map(|c| c.data.get("tool").and_then(Value::as_str))
            .map(str::to_string)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        approvals_requested: of("approval_request").count(),
        approvals_granted: of("approval")
            .filter(|e| e.data.get("granted").and_then(Value::as_bool) == Some(true))
            .count(),
        ops_run: of("op_run")
            .filter(|e| e.data.get("approved").and_then(Value::as_bool) == Some(true))
            .filter_map(|e| e.data.get("op").and_then(Value::as_str))
            .map(str::to_string)
            .collect(),
        judgments: of("judgment").count(),
        errors: of("error").count(),
        outcome: get(end, "outcome").and_then(|v| v.as_str().map(str::to_string)),
        final_text: get(end, "final").and_then(|v| v.as_str().map(str::to_string)),
        usage: get(end, "usage"),
        duration_s: get(end, "duration_s").and_then(|v| v.as_f64()),
        n_events: events.len(),
    }
}

/// Read-only access to a dataset's `.frontier/` version store.
#[derive(Debug, Clone)]
pub struct VersionStore {
    root: PathBuf,
    dir: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("{0} has no version store (run `frontier prep versions init` first)")]
    NotInitialized(PathBuf),
    #[error("unknown version `{0}`")]
    UnknownVersion(String),
    #[error("ambiguous version `{0}`: {1:?}")]
    Ambiguous(String, Vec<String>),
    #[error("{0}: history is shorter")]
    TooShort(String),
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{}: {source}", path.display())]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

impl VersionStore {
    /// The store of the dataset at `root` (`<root>/.frontier`), whether or not it exists yet.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let dir = root.join(STORE_DIR);
        Self { root, dir }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn initialized(&self) -> bool {
        self.dir.join("HEAD").is_file()
    }

    fn read(&self, path: &Path) -> Result<String, StoreError> {
        std::fs::read_to_string(path).map_err(|source| StoreError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn read_json<T: serde::de::DeserializeOwned>(&self, path: &Path) -> Result<T, StoreError> {
        serde_json::from_str(&self.read(path)?).map_err(|source| StoreError::Json {
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn head(&self) -> Result<String, StoreError> {
        let p = self.dir.join("HEAD");
        if !p.is_file() {
            return Err(StoreError::NotInitialized(self.root.clone()));
        }
        Ok(self.read(&p)?.trim().to_string())
    }

    /// `refs/<name>` -> version id.
    pub fn tags(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        let Ok(entries) = std::fs::read_dir(self.dir.join("refs")) else {
            return out;
        };
        for e in entries.flatten() {
            if e.path().is_file()
                && let Ok(text) = std::fs::read_to_string(e.path())
            {
                out.insert(
                    e.file_name().to_string_lossy().into_owned(),
                    text.trim().to_string(),
                );
            }
        }
        out
    }

    fn version_path(&self, id: &str) -> PathBuf {
        self.dir.join("versions").join(format!("{id}.json"))
    }

    /// `HEAD`, `HEAD~n`, a tag, a full id or a unique id prefix -> version id.
    pub fn resolve(&self, reference: &str) -> Result<String, StoreError> {
        let r = reference.trim();
        if r.is_empty() || r.eq_ignore_ascii_case("head") {
            return self.head();
        }
        if let Some(n) = r.strip_prefix("HEAD~") {
            let n: usize = if n.is_empty() {
                1
            } else {
                n.parse()
                    .map_err(|_| StoreError::UnknownVersion(reference.to_string()))?
            };
            let mut v = self.version_file(&self.head()?)?;
            for _ in 0..n {
                let parent = v
                    .parent
                    .clone()
                    .ok_or_else(|| StoreError::TooShort(reference.to_string()))?;
                v = self.version_file(&parent)?;
            }
            return Ok(v.id);
        }
        if let Some(id) = self.tags().get(r) {
            return Ok(id.clone());
        }
        if self.version_path(r).is_file() {
            return Ok(r.to_string());
        }
        let mut candidates = Vec::new();
        if let Ok(entries) = std::fs::read_dir(self.dir.join("versions")) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if let Some(stem) = name.strip_suffix(".json")
                    && stem.starts_with(r)
                {
                    candidates.push(stem.to_string());
                }
            }
        }
        match candidates.len() {
            1 => Ok(candidates.remove(0)),
            0 => Err(StoreError::UnknownVersion(reference.to_string())),
            _ => Err(StoreError::Ambiguous(reference.to_string(), candidates)),
        }
    }

    /// The full version file (with its tree) by resolved id.
    pub fn version_file(&self, id: &str) -> Result<VersionFile, StoreError> {
        let p = self.version_path(id);
        if !p.is_file() {
            return Err(StoreError::UnknownVersion(id.to_string()));
        }
        self.read_json(&p)
    }

    fn record_of(
        &self,
        v: &VersionFile,
        tags: &BTreeMap<String, String>,
        op: Option<String>,
    ) -> VersionRecord {
        VersionRecord {
            id: v.id.clone(),
            parent: v.parent.clone(),
            created_at: v.created_at.clone(),
            message: v.message.clone(),
            actor: v.actor.clone(),
            op_record: v.op_record.clone(),
            op,
            stats: v.stats.clone(),
            tags: tags
                .iter()
                .filter(|(_, id)| **id == v.id)
                .map(|(n, _)| n.clone())
                .collect(),
            n_samples: v.tree.samples.len(),
        }
    }

    /// One version by reference, with its tags and producing op.
    pub fn version(&self, reference: &str) -> Result<VersionRecord, StoreError> {
        let id = self.resolve(reference)?;
        let v = self.version_file(&id)?;
        let ops = self.op_index()?;
        let op = v.op_record.as_ref().and_then(|r| ops.get(r).cloned());
        Ok(self.record_of(&v, &self.tags(), op))
    }

    /// HEAD and its ancestors, newest first.
    pub fn log(&self, limit: Option<usize>) -> Result<Vec<VersionRecord>, StoreError> {
        let tags = self.tags();
        let ops = self.op_index()?;
        let mut out = Vec::new();
        for v in self.chain()? {
            if limit.is_some_and(|n| out.len() >= n) {
                break;
            }
            let op = v.op_record.as_ref().and_then(|r| ops.get(r).cloned());
            out.push(self.record_of(&v, &tags, op));
        }
        Ok(out)
    }

    /// The version files of the chain, newest first (for lineage and effective assertions).
    /// A version that names itself (or an ancestor) as parent ends the walk: the chain is
    /// read, never trusted to be acyclic.
    fn chain(&self) -> Result<Vec<VersionFile>, StoreError> {
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        let mut cur = Some(self.head()?);
        while let Some(id) = cur {
            if !seen.insert(id.clone()) {
                break;
            }
            let v = self.version_file(&id)?;
            cur = v.parent.clone();
            out.push(v);
        }
        Ok(out)
    }

    /// Every provenance record, in file order; `kind` filters.
    pub fn provenance(&self, kind: Option<&str>) -> Result<Vec<ProvenanceRecord>, StoreError> {
        let p = self.dir.join("provenance.jsonl");
        if !p.is_file() {
            return Ok(Vec::new());
        }
        let text = self.read(&p)?;
        let mut out = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let rec: ProvenanceRecord =
                serde_json::from_str(line).map_err(|source| StoreError::Json {
                    path: p.clone(),
                    source,
                })?;
            if kind.is_none_or(|k| rec.kind == k) {
                out.push(rec);
            }
        }
        Ok(out)
    }

    /// `op record id -> op id` for every op record.
    fn op_index(&self) -> Result<BTreeMap<String, String>, StoreError> {
        Ok(self
            .provenance(Some("op"))?
            .into_iter()
            .filter_map(|r| r.str("op").map(|op| (r.id.clone(), op.to_string())))
            .collect())
    }

    pub fn record(&self, id: &str) -> Result<Option<ProvenanceRecord>, StoreError> {
        Ok(self.provenance(None)?.into_iter().find(|r| r.id == id))
    }

    fn assertion_of(r: ProvenanceRecord) -> Option<Assertion> {
        Some(Assertion {
            version: r.str("version")?.to_string(),
            check: r.str("check")?.to_string(),
            status: r.str("status").unwrap_or("unknown").to_string(),
            target: r.str("target").unwrap_or("dataset").to_string(),
            detail: r.data.get("detail").cloned().unwrap_or(Value::Null),
            op_record: r.str("op_record").map(str::to_string),
            at: r.at.clone(),
            id: r.id,
        })
    }

    /// Assertions recorded on exactly `reference`.
    pub fn assertions(&self, reference: &str) -> Result<Vec<Assertion>, StoreError> {
        let id = self.resolve(reference)?;
        Ok(self
            .provenance(Some("assertion"))?
            .into_iter()
            .filter(|r| r.str("version") == Some(id.as_str()))
            .filter_map(Self::assertion_of)
            .collect())
    }

    /// Assertions that still hold at `reference`: the latest per (check, target) walking
    /// back the chain, kept only while the content they were made on is unchanged (the
    /// sample's subtree for `samples/<id>` targets, the whole sample set otherwise).
    pub fn effective_assertions(&self, reference: &str) -> Result<Vec<Assertion>, StoreError> {
        let head_id = self.resolve(reference)?;
        let head = self.version_file(&head_id)?;
        let mut by_version: BTreeMap<String, Vec<Assertion>> = BTreeMap::new();
        for a in self
            .provenance(Some("assertion"))?
            .into_iter()
            .filter_map(Self::assertion_of)
        {
            by_version.entry(a.version.clone()).or_default().push(a);
        }
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        let mut visited = BTreeSet::new();
        let mut cur = Some(head_id);
        while let Some(id) = cur {
            if !visited.insert(id.clone()) {
                break;
            }
            let v = self.version_file(&id)?;
            cur = v.parent.clone();
            for a in by_version.remove(&v.id).unwrap_or_default() {
                let key = (a.check.clone(), a.target.clone());
                if seen.contains(&key) {
                    continue;
                }
                if let Some(sid) = a.target.strip_prefix("samples/") {
                    let sid = sid.split('/').next().unwrap_or_default();
                    if v.tree.samples.get(sid) != head.tree.samples.get(sid) {
                        continue;
                    }
                } else if v.tree.samples != head.tree.samples {
                    continue;
                }
                seen.insert(key);
                out.push(a);
            }
        }
        Ok(out)
    }

    pub fn approvals(&self, reference: Option<&str>) -> Result<Vec<ProvenanceRecord>, StoreError> {
        let id = reference.map(|r| self.resolve(r)).transpose()?;
        Ok(self
            .provenance(Some("approval"))?
            .into_iter()
            .filter(|r| id.as_deref().is_none_or(|id| r.str("version") == Some(id)))
            .collect())
    }

    /// `trees/<hash>.json`: the files of one sample subtree.
    fn subtree(&self, hash: &str) -> Result<BTreeMap<String, TreeEntry>, StoreError> {
        self.read_json(&self.dir.join("trees").join(format!("{hash}.json")))
    }

    /// Every file of a version with its content hash (`samples/<id>/...` expanded).
    pub fn flatten(&self, v: &VersionFile) -> Result<BTreeMap<String, String>, StoreError> {
        let mut out: BTreeMap<String, String> = v
            .tree
            .files
            .iter()
            .map(|(p, e)| (p.clone(), e.0.clone()))
            .collect();
        for (sid, hash) in &v.tree.samples {
            for (rel, e) in self.subtree(hash)? {
                out.insert(format!("samples/{sid}/{rel}"), e.0);
            }
        }
        Ok(out)
    }

    /// Versions (oldest first) in which `target` (a path or a prefix such as
    /// `samples/run_1`) changed, with the op record that produced each.
    pub fn lineage(&self, target: &str) -> Result<Vec<LineageEntry>, StoreError> {
        let target = target.trim_end_matches('/');
        let prefix = format!("{target}/");
        let records: BTreeMap<String, ProvenanceRecord> = self
            .provenance(Some("op"))?
            .into_iter()
            .map(|r| (r.id.clone(), r))
            .collect();
        let mut out = Vec::new();
        let mut prev: BTreeMap<String, String> = BTreeMap::new();
        // subtree reads are cached: a sample unchanged across versions keeps its hash
        let mut subtrees: BTreeMap<String, BTreeMap<String, TreeEntry>> = BTreeMap::new();
        for v in self.chain()?.into_iter().rev() {
            let mut cur: BTreeMap<String, String> = BTreeMap::new();
            for (p, e) in &v.tree.files {
                if p == target || p.starts_with(&prefix) {
                    cur.insert(p.clone(), e.0.clone());
                }
            }
            for (sid, hash) in &v.tree.samples {
                let sample_dir = format!("samples/{sid}");
                if !(sample_dir == target
                    || sample_dir.starts_with(&prefix)
                    || target.starts_with(&format!("{sample_dir}/")))
                {
                    continue;
                }
                if !subtrees.contains_key(hash) {
                    subtrees.insert(hash.clone(), self.subtree(hash)?);
                }
                for (rel, e) in &subtrees[hash] {
                    let p = format!("{sample_dir}/{rel}");
                    if p == target || p.starts_with(&prefix) {
                        cur.insert(p, e.0.clone());
                    }
                }
            }
            if cur != prev {
                let rec = v.op_record.as_ref().and_then(|r| records.get(r));
                out.push(LineageEntry {
                    version: v.id.clone(),
                    created_at: v.created_at.clone(),
                    message: v.message.clone(),
                    op: rec.and_then(|r| r.str("op")).map(str::to_string),
                    op_record: v.op_record.clone(),
                    params: rec.and_then(|r| r.data.get("params").cloned()),
                    n_files: cur.len(),
                    added: cur.keys().filter(|p| !prev.contains_key(*p)).count(),
                    removed: prev.keys().filter(|p| !cur.contains_key(*p)).count(),
                    modified: cur
                        .iter()
                        .filter(|(p, h)| prev.get(*p).is_some_and(|ph| ph != *h))
                        .count(),
                });
            }
            prev = cur;
        }
        Ok(out)
    }

    /// The provenance DAG: version nodes (with tags, message, actor, stats) and op nodes;
    /// edges `parent -> version` and `input_version -> op -> output_version`. Assertions and
    /// approvals are counted per version, not expanded.
    pub fn graph(&self) -> Result<ProvenanceGraph, StoreError> {
        let records = self.provenance(None)?;
        let tags = self.tags();
        let mut g = ProvenanceGraph {
            head: self.head().ok(),
            tags: tags.clone(),
            ..Default::default()
        };
        let mut counts: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
        let mut outputs: BTreeMap<String, String> = BTreeMap::new();
        for r in &records {
            match r.kind.as_str() {
                "assertion" => {
                    if let Some(v) = r.str("version") {
                        let c = counts.entry(v.to_string()).or_default();
                        match r.str("status") {
                            Some("pass") => c.0 += 1,
                            Some("warn") => c.1 += 1,
                            Some("skip") => {}
                            _ => c.2 += 1,
                        }
                    }
                }
                "note" => {
                    if let (Some(op), Some(out)) = (r.str("ref"), r.str("output_version")) {
                        outputs.insert(op.to_string(), out.to_string());
                    }
                }
                _ => {}
            }
        }
        let approvals: BTreeMap<String, usize> = records
            .iter()
            .filter(|r| r.kind == "approval")
            .fold(BTreeMap::new(), |mut m, r| {
                if let Some(v) = r.str("version") {
                    *m.entry(v.to_string()).or_default() += 1;
                }
                m
            });
        for r in &records {
            match r.kind.as_str() {
                "version" => {
                    let Some(id) = r.str("version") else { continue };
                    let (pass, warn, fail) = counts.get(id).copied().unwrap_or_default();
                    g.nodes.push(serde_json::json!({
                        "kind": "version", "id": id, "at": r.at, "actor": r.actor,
                        "message": r.data.get("message"), "stats": r.data.get("stats"),
                        "tags": tags.iter().filter(|(_, v)| *v == id).map(|(n, _)| n).collect::<Vec<_>>(),
                        "assertions": {"pass": pass, "warn": warn, "fail": fail},
                        "approvals": approvals.get(id).copied().unwrap_or(0),
                    }));
                    if let Some(parent) = r.str("parent") {
                        g.edges
                            .push(serde_json::json!({"from": parent, "to": id, "kind": "parent"}));
                    }
                }
                "op" => {
                    let output = outputs.get(&r.id).cloned();
                    g.nodes.push(serde_json::json!({
                        "kind": "op", "id": r.id, "at": r.at, "actor": r.actor,
                        "op": r.str("op"), "op_version": r.str("op_version"),
                        "params": r.data.get("params"), "summary": r.str("summary"),
                        "determinism": r.str("determinism"), "n_changes": r.data.get("n_changes"),
                        "input_version": r.str("input_version"), "output_version": output,
                    }));
                    if let Some(input) = r.str("input_version") {
                        g.edges
                            .push(serde_json::json!({"from": input, "to": r.id, "kind": "input"}));
                    }
                    if let Some(out) = output {
                        g.edges
                            .push(serde_json::json!({"from": r.id, "to": out, "kind": "output"}));
                    }
                }
                _ => {}
            }
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn store() -> (tempfile::TempDir, VersionStore) {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path().join(STORE_DIR);
        write(&d.join("HEAD"), "v_b\n");
        write(&d.join("refs/baseline"), "v_a\n");
        write(
            &d.join("trees/t1.json"),
            r#"{"meta.json":["h1",10],"surface.zarr/p/c/0":["h2",20]}"#,
        );
        write(
            &d.join("trees/t2.json"),
            r#"{"meta.json":["h1",10],"surface.zarr/p/c/0":["h3",20]}"#,
        );
        write(
            &d.join("versions/v_a.json"),
            r#"{"id":"v_a","parent":null,"created_at":"2026-10-11T00:00:00+00:00","message":"baseline","actor":{"kind":"human","name":"m"},"op_record":"r0","stats":{"n_files":3},"tree":{"files":{"manifest.json":["m1",5]},"samples":{"run_1":"t1"}}}"#,
        );
        write(
            &d.join("versions/v_b.json"),
            r#"{"id":"v_b","parent":"v_a","created_at":"2026-10-11T00:01:00+00:00","message":"convert","actor":{"kind":"human","name":"m"},"op_record":"r1","stats":{"n_files":3},"tree":{"files":{"manifest.json":["m2",5]},"samples":{"run_1":"t2"}}}"#,
        );
        write(
            &d.join("provenance.jsonl"),
            concat!(
                r#"{"kind":"op","op":"prep.baseline","op_version":"1.0.0","params":{},"input_version":null,"actor":{"kind":"system","name":"init"},"at":"t","id":"r0"}"#,
                "\n",
                r#"{"kind":"version","version":"v_a","parent":null,"message":"baseline","op_record":"r0","actor":{"kind":"human","name":"m"},"stats":{"n_files":3},"at":"t","id":"x1"}"#,
                "\n",
                r#"{"kind":"assertion","version":"v_a","check":"watertight","status":"pass","target":"samples/run_1","detail":{},"at":"t","actor":{},"id":"a1"}"#,
                "\n",
                r#"{"kind":"assertion","version":"v_a","check":"schema","status":"pass","target":"dataset","detail":{},"at":"t","actor":{},"id":"a2"}"#,
                "\n",
                r#"{"kind":"op","op":"prep.units.convert","op_version":"1.0.0","params":{"conversions":{"p":"Pa"}},"input_version":"v_a","actor":{"kind":"human","name":"m"},"at":"t","id":"r1"}"#,
                "\n",
                r#"{"kind":"version","version":"v_b","parent":"v_a","message":"convert","op_record":"r1","actor":{"kind":"human","name":"m"},"stats":{"n_files":3},"at":"t","id":"x2"}"#,
                "\n",
                r#"{"kind":"note","ref":"r1","output_version":"v_b","note":"output","at":"t","actor":{},"id":"n1"}"#,
                "\n",
                r#"{"kind":"approval","version":"v_b","approved_by":"alice","decision":"approve","note":"","actor":{"kind":"human","name":"alice"},"at":"t","id":"ap1"}"#,
                "\n",
            ),
        );
        let s = VersionStore::new(tmp.path());
        (tmp, s)
    }

    #[test]
    fn log_resolves_tags_and_ops() {
        let (_t, s) = store();
        assert!(s.initialized());
        let log = s.log(None).unwrap();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].id, "v_b");
        assert_eq!(log[0].op.as_deref(), Some("prep.units.convert"));
        assert_eq!(log[1].tags, vec!["baseline"]);
        assert_eq!(s.resolve("HEAD~1").unwrap(), "v_a");
        assert_eq!(s.resolve("baseline").unwrap(), "v_a");
        assert!(
            s.resolve("v_")
                .unwrap_err()
                .to_string()
                .contains("ambiguous")
        );
        assert_eq!(s.version("v_b").unwrap().n_samples, 1);
    }

    #[test]
    fn effective_assertions_follow_content() {
        let (_t, s) = store();
        // run_1 changed between v_a and v_b: its watertight check no longer applies; the
        // dataset-level one does not either (the sample set changed)
        assert!(s.effective_assertions("HEAD").unwrap().is_empty());
        let at_a = s.effective_assertions("v_a").unwrap();
        assert_eq!(at_a.len(), 2);
        assert_eq!(s.assertions("v_a").unwrap().len(), 2);
        assert_eq!(s.approvals(Some("HEAD")).unwrap().len(), 1);
    }

    #[test]
    fn lineage_and_graph() {
        let (_t, s) = store();
        let l = s.lineage("samples/run_1").unwrap();
        assert_eq!(l.len(), 2);
        assert_eq!(l[1].modified, 1);
        assert_eq!(l[1].op.as_deref(), Some("prep.units.convert"));
        assert_eq!(s.lineage("manifest.json").unwrap().len(), 2);
        assert_eq!(s.lineage("samples/run_1/meta.json").unwrap().len(), 1);
        let g = s.graph().unwrap();
        assert_eq!(g.nodes.len(), 4);
        assert_eq!(g.edges.len(), 3);
        assert_eq!(g.head.as_deref(), Some("v_b"));
    }

    #[test]
    fn a_self_parented_version_does_not_loop() {
        let (_t, s) = store();
        let d = s.dir().join("versions");
        std::fs::write(
            d.join("v_c.json"),
            r#"{"id":"v_c","parent":"v_c","created_at":"t","message":"revert","actor":{},"op_record":null,"stats":{},"tree":{"files":{},"samples":{}}}"#,
        )
        .unwrap();
        std::fs::write(s.dir().join("HEAD"), "v_c\n").unwrap();
        assert_eq!(s.log(None).unwrap().len(), 1);
        assert!(s.effective_assertions("HEAD").unwrap().is_empty());
        assert_eq!(s.lineage("manifest.json").unwrap().len(), 0);
    }

    #[test]
    fn session_summary() {
        let text = concat!(
            r#"{"seq":0,"t":1.0,"kind":"session_start","task":"rename","model":{"provider":"local"},"harness":"builtin"}"#,
            "\n",
            r#"{"seq":1,"t":1.0,"kind":"tool_call","tool":"prep.fields.rename.plan"}"#,
            "\n",
            r#"{"seq":2,"t":1.0,"kind":"approval_request","plan_id":"p"}"#,
            "\n",
            r#"{"seq":3,"t":1.0,"kind":"approval","granted":true}"#,
            "\n",
            r#"{"seq":4,"t":1.0,"kind":"op_run","op":"prep.fields.rename","approved":true}"#,
            "\n",
            r#"{"seq":5,"t":1.0,"kind":"session_end","outcome":"done","final":"ok","duration_s":2.5}"#,
            "\n",
            "{\"seq\":6,\"kind\":\"partial",
            "\n",
        );
        let events = parse_session_events(text);
        assert_eq!(events.len(), 6);
        let s = summarize_session("s1", &events);
        assert_eq!(s.tool_calls, 1);
        assert_eq!(s.approvals_granted, 1);
        assert_eq!(s.ops_run, vec!["prep.fields.rename"]);
        assert_eq!(s.outcome.as_deref(), Some("done"));
        assert_eq!(s.duration_s, Some(2.5));
    }
}
