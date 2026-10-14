// SPDX-License-Identifier: Apache-2.0
//! Triangulated surface sample (points, faces, per-point / per-face fields) and its Arrow
//! wire format, served by `GET /api/v1/datasets/{id}/samples/{sid}/surface?lod=N`.
//!
//! Arrow layout: **one `RecordBatch` with exactly one row**, so a whole mesh is a single
//! IPC message and variable-length parts (N points vs M faces) live in list columns:
//!
//! | column            | type                                   | notes                         |
//! |-------------------|----------------------------------------|-------------------------------|
//! | `points`          | `List<FixedSizeList<Float32, 3>>`      | xyz per vertex                |
//! | `faces`           | `List<FixedSizeList<Int32, 3>>`        | triangles, vertex indices     |
//! | `cell_area`       | `List<Float32>` (nullable)             | one per face                  |
//! | `field:<name>`    | `List<Float32>` (C = 1) or `List<FixedSizeList<Float32, C>>` | field metadata `location` (`point`/`cell`), `components` |
//!
//! Schema metadata carries `sample_id`, `lod`, `n_points`, `n_faces` and anything else the
//! server adds. [`SurfaceMesh::from_record_batch`] is the reference decoder.

use crate::field::FieldLocation;
use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, ListArray, RecordBatch,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SurfaceError {
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("invalid surface: {0}")]
    Invalid(String),
}

/// Flat, row-major `(n, components)` field values.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceField {
    pub location: FieldLocation,
    pub components: usize,
    pub values: Vec<f32>,
}

impl SurfaceField {
    pub fn len(&self) -> usize {
        self.values.len().checked_div(self.components).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceMesh {
    /// `n_points * 3`, xyz interleaved.
    pub points: Vec<f32>,
    /// `n_faces * 3`, vertex indices.
    pub faces: Vec<i32>,
    pub cell_area: Option<Vec<f32>>,
    pub fields: BTreeMap<String, SurfaceField>,
    pub metadata: BTreeMap<String, String>,
}

pub(crate) const FIELD_PREFIX: &str = "field:";

impl SurfaceMesh {
    pub fn n_points(&self) -> usize {
        self.points.len() / 3
    }

    pub fn n_faces(&self) -> usize {
        self.faces.len() / 3
    }

    pub fn validate(&self) -> Result<(), SurfaceError> {
        if !self.points.len().is_multiple_of(3) {
            return Err(SurfaceError::Invalid(
                "points length not a multiple of 3".into(),
            ));
        }
        if !self.faces.len().is_multiple_of(3) {
            return Err(SurfaceError::Invalid(
                "faces length not a multiple of 3".into(),
            ));
        }
        let n_points = self.n_points() as i32;
        if self.faces.iter().any(|&i| i < 0 || i >= n_points) {
            return Err(SurfaceError::Invalid("face index out of range".into()));
        }
        if let Some(a) = &self.cell_area
            && a.len() != self.n_faces()
        {
            return Err(SurfaceError::Invalid("cell_area length != n_faces".into()));
        }
        for (name, f) in &self.fields {
            if f.components == 0 || !f.values.len().is_multiple_of(f.components) {
                return Err(SurfaceError::Invalid(format!(
                    "field {name}: bad components"
                )));
            }
            let expected = match f.location {
                FieldLocation::Point => self.n_points(),
                FieldLocation::Cell | FieldLocation::Face => self.n_faces(),
            };
            if f.len() != expected {
                return Err(SurfaceError::Invalid(format!(
                    "field {name}: {} rows, expected {expected}",
                    f.len()
                )));
            }
        }
        Ok(())
    }

    pub fn to_record_batch(&self) -> Result<RecordBatch, SurfaceError> {
        self.validate()?;
        let mut fields = Vec::new();
        let mut columns: Vec<ArrayRef> = Vec::new();

        let (f, a) = list_of_fixed_f32("points", &self.points, 3)?;
        fields.push(f);
        columns.push(a);
        let (f, a) = list_of_fixed_i32("faces", &self.faces, 3)?;
        fields.push(f);
        columns.push(a);

        let area = self.cell_area.clone().unwrap_or_default();
        let (f, a) = list_f32("cell_area", &area, self.cell_area.is_some())?;
        fields.push(f);
        columns.push(a);

        for (name, field) in &self.fields {
            let col = format!("{FIELD_PREFIX}{name}");
            let (mut f, a) = if field.components == 1 {
                list_f32(&col, &field.values, true)?
            } else {
                list_of_fixed_f32(&col, &field.values, field.components as i32)?
            };
            let loc = match field.location {
                FieldLocation::Point => "point",
                FieldLocation::Cell => "cell",
                FieldLocation::Face => "face",
            };
            f = f.with_metadata(HashMap::from([
                ("location".to_string(), loc.to_string()),
                ("components".to_string(), field.components.to_string()),
            ]));
            fields.push(f);
            columns.push(a);
        }

        let mut meta: HashMap<String, String> = self.metadata.clone().into_iter().collect();
        meta.insert("n_points".into(), self.n_points().to_string());
        meta.insert("n_faces".into(), self.n_faces().to_string());
        let schema = Arc::new(Schema::new_with_metadata(fields, meta));
        Ok(RecordBatch::try_new(schema, columns)?)
    }

    pub fn from_record_batch(batch: &RecordBatch) -> Result<Self, SurfaceError> {
        if batch.num_rows() != 1 {
            return Err(SurfaceError::Invalid(format!(
                "surface batch must have 1 row, got {}",
                batch.num_rows()
            )));
        }
        let mut mesh = SurfaceMesh {
            metadata: batch
                .schema()
                .metadata()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            ..Default::default()
        };
        for (i, field) in batch.schema().fields().iter().enumerate() {
            let col = batch.column(i);
            match field.name().as_str() {
                "points" => mesh.points = read_list_f32(col)?,
                "faces" => mesh.faces = read_list_i32(col)?,
                "cell_area" => {
                    mesh.cell_area = if col.is_null(0) {
                        None
                    } else {
                        Some(read_list_f32(col)?)
                    }
                }
                name if name.starts_with(FIELD_PREFIX) => {
                    let md = field.metadata();
                    let components: usize = md
                        .get("components")
                        .and_then(|c| c.parse().ok())
                        .unwrap_or(1);
                    let location = match md.get("location").map(String::as_str) {
                        Some("cell") => FieldLocation::Cell,
                        Some("face") => FieldLocation::Face,
                        _ => FieldLocation::Point,
                    };
                    mesh.fields.insert(
                        name[FIELD_PREFIX.len()..].to_string(),
                        SurfaceField {
                            location,
                            components,
                            values: read_list_f32(col)?,
                        },
                    );
                }
                _ => {}
            }
        }
        mesh.validate()?;
        Ok(mesh)
    }

    /// Serialize as an Arrow IPC *stream* (`application/vnd.apache.arrow.stream`).
    pub fn to_ipc_stream(&self) -> Result<Vec<u8>, SurfaceError> {
        let batch = self.to_record_batch()?;
        let mut buf = Vec::new();
        {
            let mut w = StreamWriter::try_new(&mut buf, batch.schema().as_ref())?;
            w.write(&batch)?;
            w.finish()?;
        }
        Ok(buf)
    }

    pub fn from_ipc_stream(bytes: &[u8]) -> Result<Self, SurfaceError> {
        let reader = StreamReader::try_new(std::io::Cursor::new(bytes), None)?;
        let batch = reader
            .into_iter()
            .next()
            .ok_or_else(|| SurfaceError::Invalid("empty IPC stream".into()))??;
        Self::from_record_batch(&batch)
    }
}

pub(crate) fn list_f32(
    name: &str,
    values: &[f32],
    valid: bool,
) -> Result<(Field, ArrayRef), SurfaceError> {
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let values: ArrayRef = Arc::new(Float32Array::from(values.to_vec()));
    let nulls = (!valid).then(|| arrow::buffer::NullBuffer::new_null(1));
    let list = ListArray::try_new(
        item.clone(),
        OffsetBuffer::from_lengths([values.len()]),
        values,
        nulls,
    )?;
    Ok((Field::new(name, DataType::List(item), true), Arc::new(list)))
}

pub(crate) fn list_of_fixed_f32(
    name: &str,
    values: &[f32],
    size: i32,
) -> Result<(Field, ArrayRef), SurfaceError> {
    let inner: ArrayRef = Arc::new(Float32Array::from(values.to_vec()));
    list_of_fixed(name, inner, DataType::Float32, size)
}

pub(crate) fn list_of_fixed_i32(
    name: &str,
    values: &[i32],
    size: i32,
) -> Result<(Field, ArrayRef), SurfaceError> {
    let inner: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
    list_of_fixed(name, inner, DataType::Int32, size)
}

pub(crate) fn list_of_fixed(
    name: &str,
    inner: ArrayRef,
    dt: DataType,
    size: i32,
) -> Result<(Field, ArrayRef), SurfaceError> {
    let item = Arc::new(Field::new("item", dt, false));
    let n = inner.len() / size as usize;
    let fixed = FixedSizeListArray::try_new(item.clone(), size, inner, None)?;
    let fixed_field = Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(item, size),
        false,
    ));
    let list = ListArray::try_new(
        fixed_field.clone(),
        OffsetBuffer::from_lengths([n]),
        Arc::new(fixed),
        None,
    )?;
    Ok((
        Field::new(name, DataType::List(fixed_field), true),
        Arc::new(list),
    ))
}

/// Flatten `List<Float32>` or `List<FixedSizeList<Float32, C>>` (row 0) to a Vec.
pub(crate) fn read_list_f32(col: &ArrayRef) -> Result<Vec<f32>, SurfaceError> {
    let inner = row0(col)?;
    let inner = match inner.data_type() {
        DataType::FixedSizeList(_, _) => inner
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| SurfaceError::Invalid("expected FixedSizeList".into()))?
            .values()
            .clone(),
        _ => inner,
    };
    let arr = inner
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            SurfaceError::Invalid(format!("expected Float32, got {}", inner.data_type()))
        })?;
    Ok(arr.values().to_vec())
}

pub(crate) fn read_list_i32(col: &ArrayRef) -> Result<Vec<i32>, SurfaceError> {
    let inner = row0(col)?;
    let inner = match inner.data_type() {
        DataType::FixedSizeList(_, _) => inner
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| SurfaceError::Invalid("expected FixedSizeList".into()))?
            .values()
            .clone(),
        _ => inner,
    };
    let arr = inner.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
        SurfaceError::Invalid(format!("expected Int32, got {}", inner.data_type()))
    })?;
    Ok(arr.values().to_vec())
}

pub(crate) fn row0(col: &ArrayRef) -> Result<ArrayRef, SurfaceError> {
    let list = col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| SurfaceError::Invalid(format!("expected List, got {}", col.data_type())))?;
    Ok(list.value(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SurfaceMesh {
        let mut fields = BTreeMap::new();
        fields.insert(
            "p".to_string(),
            SurfaceField {
                location: FieldLocation::Point,
                components: 1,
                values: vec![1., 2., 3., 4.],
            },
        );
        fields.insert(
            "wallShearStress".to_string(),
            SurfaceField {
                location: FieldLocation::Point,
                components: 3,
                values: (0..12).map(|x| x as f32).collect(),
            },
        );
        SurfaceMesh {
            points: vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            faces: vec![0, 1, 2, 0, 2, 3],
            cell_area: Some(vec![0.5, 0.5]),
            fields,
            metadata: BTreeMap::from([("sample_id".into(), "run_1".into())]),
        }
    }

    #[test]
    fn ipc_roundtrip() {
        let mesh = sample();
        let bytes = mesh.to_ipc_stream().unwrap();
        let back = SurfaceMesh::from_ipc_stream(&bytes).unwrap();
        assert_eq!(back.points, mesh.points);
        assert_eq!(back.faces, mesh.faces);
        assert_eq!(back.cell_area, mesh.cell_area);
        assert_eq!(back.fields, mesh.fields);
        assert_eq!(back.metadata["sample_id"], "run_1");
        assert_eq!(back.metadata["n_points"], "4");
    }

    #[test]
    fn missing_cell_area_is_null() {
        let mut mesh = sample();
        mesh.cell_area = None;
        let back = SurfaceMesh::from_ipc_stream(&mesh.to_ipc_stream().unwrap()).unwrap();
        assert!(back.cell_area.is_none());
    }

    #[test]
    fn rejects_bad_face_index() {
        let mut mesh = sample();
        mesh.faces[0] = 99;
        assert!(mesh.to_record_batch().is_err());
    }
}
