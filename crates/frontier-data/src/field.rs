// SPDX-License-Identifier: Apache-2.0
//! Descriptors for simulation meshes and fields (the "samples" of a physics dataset).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldLocation {
    Point,
    Cell,
    Face,
}

/// A mesh as stored in a Frontier dataset: unstructured, possibly polyhedral.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshDescriptor {
    pub n_points: u64,
    pub n_cells: u64,
    pub n_faces: Option<u64>,
    pub dimension: u8,
    /// Axis-aligned bounding box `[xmin, ymin, zmin, xmax, ymax, zmax]`.
    pub bounds: [f64; 6],
}

/// A scalar / vector / tensor field attached to a mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDescriptor {
    /// e.g. `U`, `p`, `wallShearStress`, `nut`.
    pub name: String,
    pub location: FieldLocation,
    /// 1 = scalar, 3 = vector, 6 = symmetric tensor, 9 = tensor.
    pub components: u8,
    /// SI unit string, e.g. `m/s`, `m^2/s^2`.
    #[serde(default)]
    pub unit: String,
    /// Simulation time this field was written at (None for steady-state).
    pub time: Option<f64>,
}
