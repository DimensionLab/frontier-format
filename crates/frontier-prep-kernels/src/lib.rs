// SPDX-License-Identifier: Apache-2.0
//! Deterministic data-preparation kernels: the Rust ports of the reference kernels in
//! `frontier_prep.ops.mesh` / `frontier_prep.ops.units` (port specification:
//! `docs/spec/data-prep.md` section 11).
//!
//! Contract shared by every kernel:
//!
//! - the output is a pure function of the input arrays: no hash map decides an output, and
//!   every sort has a total order (ties broken by the original index);
//! - rayon is used only for element-wise maps and for sorts with a total order, so the
//!   result is identical for any thread count;
//! - float reductions (`area`, `boundary_length`, `signed_volume`) follow numpy's pairwise
//!   summation ([`sum::pairwise`]) on values computed without fused multiply-add, which makes
//!   them bitwise equal to the Python reference.
//!
//! Arrays are flat row-major slices: `faces` holds `3 * M` vertex indices, `points`
//! `3 * N` coordinates.

pub mod error;
pub mod merge;
pub mod orient;
pub mod real;
pub mod scale;
pub mod sum;
pub mod tables;
pub mod topology;

pub use error::KernelError;
pub use merge::{Merged, merge_vertices};
pub use orient::{duplicate_faces, orient_consistently};
pub use real::Real;
pub use scale::scale_block;
pub use tables::{EdgeTables, components_from_slots};
pub use topology::{Audit, face_areas, topology};

/// Version of the kernel implementations, recorded in provenance by the Python ops.
pub const KERNELS_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Revision of the port specification the kernels implement. Bumped when a rule that decides
/// an output changes (the Python reference changes in the same commit).
pub const SPEC_REVISION: u32 = 1;
