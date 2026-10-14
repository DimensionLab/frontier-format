// SPDX-License-Identifier: Apache-2.0
//! Data model for Frontier.
//!
//! Design (inspired by Rerun's chunk store): every piece of data is addressed by an
//! *entity path* (`/run/<id>/metrics/loss`, `/dataset/<id>/sample/42/mesh`) and indexed
//! by one or more *timelines* (training `step`, wall-clock `time`, simulation `t`).
//! Columns are Apache Arrow so the same buffers flow zero-copy between Rust, Python
//! (via `frontier-py`), the wgpu renderer, and the wire (Arrow IPC / Flight).

pub mod entity;
pub mod field;
pub mod frame;
pub mod mesh;
pub mod metrics;
pub mod surface;

pub use entity::EntityPath;
pub use field::{FieldDescriptor, FieldLocation, MeshDescriptor};
pub use frame::{FieldFrame, FieldStats};
pub use mesh::VolumeMesh;
pub use metrics::MetricsBatch;
pub use surface::{SurfaceError, SurfaceField, SurfaceMesh};
