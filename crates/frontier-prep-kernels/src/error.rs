// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

/// Why a kernel refused its input. Kernels never panic on malformed arrays.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// A flat array whose length is not a multiple of its row width.
    #[error("{name} has {len} elements, not a multiple of {width}")]
    Shape {
        name: &'static str,
        len: usize,
        width: usize,
    },
    /// A face references a negative vertex index.
    #[error("face {face} references negative vertex index {index}")]
    NegativeIndex { face: usize, index: i64 },
    /// A face references a vertex beyond the point array.
    #[error("face {face} references vertex {index}, but there are {n_points} points")]
    IndexOutOfRange {
        face: usize,
        index: i64,
        n_points: usize,
    },
    /// A slot id handed to the component labelling is negative.
    #[error("slot {slot} carries negative id {id}")]
    NegativeSlotId { slot: usize, id: i64 },
    /// `inv` does not have three slots per face.
    #[error("expected {expected} slot ids (3 per face), got {got}")]
    SlotCount { expected: usize, got: usize },
}
