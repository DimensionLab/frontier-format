// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::fmt;

/// Hierarchical address of a logged entity, e.g. `/run/<id>/metrics/loss`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EntityPath(Vec<String>);

impl EntityPath {
    pub fn root() -> Self {
        Self(Vec::new())
    }

    pub fn parse(s: &str) -> Self {
        Self(
            s.split('/')
                .filter(|p| !p.is_empty())
                .map(str::to_owned)
                .collect(),
        )
    }

    pub fn join(&self, part: impl Into<String>) -> Self {
        let mut parts = self.0.clone();
        parts.push(part.into());
        Self(parts)
    }

    pub fn parts(&self) -> &[String] {
        &self.0
    }

    pub fn is_ancestor_of(&self, other: &EntityPath) -> bool {
        other.0.len() > self.0.len() && other.0.starts_with(&self.0)
    }
}

impl fmt::Display for EntityPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "/{}", self.0.join("/"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_display() {
        let p = EntityPath::parse("/run/abc/metrics//loss/");
        assert_eq!(p.parts(), &["run", "abc", "metrics", "loss"]);
        assert_eq!(p.to_string(), "/run/abc/metrics/loss");
        assert!(EntityPath::parse("/run/abc").is_ancestor_of(&p));
    }
}
