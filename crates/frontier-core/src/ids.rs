// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

macro_rules! typed_id {
    ($name:ident) => {
        /// Globally unique, time-ordered (UUIDv7) identifier.
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl std::str::FromStr for $name {
            type Err = uuid::Error;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Uuid::parse_str(s).map(Self)
            }
        }
    };
}

typed_id!(ProjectId);
typed_id!(RunId);
typed_id!(DatasetId);
typed_id!(JobId);
typed_id!(ArtifactId);
