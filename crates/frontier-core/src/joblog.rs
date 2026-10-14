// SPDX-License-Identifier: Apache-2.0
//! Log records for job logs: what the worker and the server themselves write into the
//! log of a job, as JSON lines
//! (`{"ts", "level", "logger", "step"?, "message", "fields"?}`), the form the tools a job
//! runs use too (`frontier_data.joblog`) and the log pages read (`docs/spec/api.md`, "Job
//! and step logs as pages"). What a process prints on its own console is not this: that
//! stays text for the person at the terminal.

use serde_json::{Map, Value, json};

/// One record as its line (no line breaks). `fields` is left out when it is not an
/// object with entries.
pub fn record(level: &str, logger: &str, message: &str, fields: Value) -> String {
    let mut out = Map::new();
    out.insert(
        "ts".into(),
        json!(chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)),
    );
    out.insert("level".into(), json!(level));
    out.insert("logger".into(), json!(logger));
    out.insert("message".into(), json!(message));
    if fields.as_object().is_some_and(|f| !f.is_empty()) {
        out.insert("fields".into(), fields);
    }
    Value::Object(out).to_string()
}

/// A line written as `[logger-name] words` (how the worker and the server marked their
/// own lines) as a record of that logger; any other line as it is. The level is `error`
/// for words that say something could not be done, else `info`.
pub fn from_bracketed(line: &str) -> String {
    let Some(rest) = line.strip_prefix('[') else {
        return line.to_string();
    };
    let Some((name, words)) = rest.split_once("] ") else {
        return line.to_string();
    };
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return line.to_string();
    }
    let level = if words.starts_with("could not") || words.starts_with("failed") {
        "error"
    } else if words.starts_with("timeout") {
        "warn"
    } else {
        "info"
    };
    record(level, &name.replace('-', "_"), words, Value::Null)
}

/// Whether a JSON value is a log record, as opposed to the JSON a command printed as its
/// result: an object with `ts`, `level` and `message`.
pub fn is_record(v: &Value) -> bool {
    v.as_object().is_some_and(|o| {
        ["ts", "level", "message"]
            .iter()
            .all(|k| o.contains_key(*k))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_is_one_json_line_with_the_keys_the_log_pages_read() {
        let line = record(
            "warn",
            "frontier_worker",
            "lease lost\nretrying",
            json!({"job": "j1"}),
        );
        assert!(!line.contains('\n'));
        let v: Value = serde_json::from_str(&line).unwrap();
        assert!(is_record(&v));
        assert_eq!(v["level"], "warn");
        assert_eq!(v["logger"], "frontier_worker");
        assert_eq!(v["message"], "lease lost\nretrying");
        assert_eq!(v["fields"]["job"], "j1");
        assert!(v["ts"].as_str().unwrap().ends_with('Z'));
        let plain: Value = serde_json::from_str(&record("info", "x", "m", Value::Null)).unwrap();
        assert!(plain.get("fields").is_none());
    }

    #[test]
    fn bracketed_lines_become_records_and_other_lines_stay() {
        let v: Value =
            serde_json::from_str(&from_bracketed("[frontier-worker] started: python -m x"))
                .unwrap();
        assert_eq!(v["logger"], "frontier_worker");
        assert_eq!(v["message"], "started: python -m x");
        assert_eq!(v["level"], "info");
        let failed: Value = serde_json::from_str(&from_bracketed(
            "[frontier-server] could not register /d: no manifest",
        ))
        .unwrap();
        assert_eq!(failed["level"], "error");
        for other in ["epoch 3", "[1, 2]", "[not a name] x", "[x]"] {
            assert_eq!(from_bracketed(other), other);
        }
    }

    #[test]
    fn a_result_is_not_a_record() {
        assert!(!is_record(&json!({"ok": true, "message": "done"})));
        assert!(!is_record(&json!([1, 2])));
    }
}
