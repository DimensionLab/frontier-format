// SPDX-License-Identifier: Apache-2.0
//! One time step of one field: the unit a timeline scrubs through.
//!
//! Served by `GET .../samples/{sid}/grid?field=&t=` (regular grids) and
//! `GET .../samples/{sid}/field?field=&t=` (per-point / per-cell values of a surface or
//! mesh whose geometry the client already holds).
//!
//! Arrow layout: **one `RecordBatch`, one row**, one column `values: List<Float32>` holding
//! the row-major array. Everything else is schema metadata (all values are strings):
//!
//! | key          | value                                                              |
//! |--------------|--------------------------------------------------------------------|
//! | `field`      | field name                                                         |
//! | `shape`      | JSON list, shape of `values` after stride / slice, e.g. `[64,32,2]` |
//! | `dims`       | JSON list naming each axis of `shape`, e.g. `["x","y","component"]`  |
//! | `full_shape` | JSON list, shape of one step as stored (before stride / slice)       |
//! | `t`          | time index                                                         |
//! | `time`       | physical time of that index (`times[t]`), `0` for steady data        |
//! | `n_steps`    | number of steps of the sample                                      |
//! | `range`      | JSON `[min, max]` over the finite values of this frame             |
//! | `location`   | `point` / `cell` / `grid`                                          |

use crate::surface::{SurfaceError, list_f32, read_list_f32};
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FieldFrame {
    pub field: String,
    /// Row-major values with shape [`FieldFrame::shape`].
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
    pub dims: Vec<String>,
    /// Shape of one stored step before stride / slice were applied.
    pub full_shape: Vec<usize>,
    pub t: usize,
    pub time: f64,
    pub n_steps: usize,
    /// `[min, max]` over finite values; `[0, 0]` when there are none.
    pub range: [f32; 2],
    /// `point`, `cell` or `grid`.
    pub location: String,
    /// Extra schema metadata (sample id, slice, stride, ...).
    pub metadata: BTreeMap<String, String>,
}

/// `[min, max]` over the finite entries of `values`.
pub fn finite_range(values: &[f32]) -> [f32; 2] {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if lo > hi { [0.0, 0.0] } else { [lo, hi] }
}

impl FieldFrame {
    /// Number of components (size of the `component` axis, 1 when there is none).
    pub fn components(&self) -> usize {
        self.dims
            .iter()
            .position(|d| d == "component")
            .and_then(|i| self.shape.get(i).copied())
            .unwrap_or(1)
    }

    /// Sizes of the non-component axes, in order.
    pub fn spatial_shape(&self) -> Vec<usize> {
        self.dims
            .iter()
            .zip(&self.shape)
            .filter(|(d, _)| d.as_str() != "component")
            .map(|(_, n)| *n)
            .collect()
    }

    pub fn validate(&self) -> Result<(), SurfaceError> {
        let n: usize = self.shape.iter().product();
        if n != self.values.len() {
            return Err(SurfaceError::Invalid(format!(
                "frame {}: shape {:?} does not match {} values",
                self.field,
                self.shape,
                self.values.len()
            )));
        }
        if self.dims.len() != self.shape.len() {
            return Err(SurfaceError::Invalid(format!(
                "frame {}: {} dims for a rank-{} shape",
                self.field,
                self.dims.len(),
                self.shape.len()
            )));
        }
        Ok(())
    }

    pub fn to_ipc_stream(&self) -> Result<Vec<u8>, SurfaceError> {
        self.validate()?;
        let (f, a) = list_f32("values", &self.values, true)?;
        let mut meta: HashMap<String, String> = self.metadata.clone().into_iter().collect();
        meta.insert("field".into(), self.field.clone());
        meta.insert("shape".into(), usize_list_json(&self.shape));
        meta.insert("dims".into(), str_list_json(&self.dims));
        meta.insert("full_shape".into(), usize_list_json(&self.full_shape));
        meta.insert("t".into(), self.t.to_string());
        meta.insert("time".into(), self.time.to_string());
        meta.insert("n_steps".into(), self.n_steps.to_string());
        meta.insert(
            "range".into(),
            format!("[{},{}]", self.range[0], self.range[1]),
        );
        meta.insert("location".into(), self.location.clone());
        let schema = Arc::new(Schema::new_with_metadata(vec![f], meta));
        let batch = RecordBatch::try_new(schema, vec![a])?;
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
        let schema = batch.schema();
        let idx = schema
            .index_of("values")
            .map_err(|_| SurfaceError::Invalid("frame has no `values` column".into()))?;
        let values = read_list_f32(batch.column(idx))?;
        let md = schema.metadata();
        let get = |k: &str| md.get(k).map(String::as_str).unwrap_or("");
        let frame = FieldFrame {
            field: get("field").to_string(),
            shape: parse_usize_list(get("shape")),
            dims: parse_str_list(get("dims")),
            full_shape: parse_usize_list(get("full_shape")),
            t: get("t").parse().unwrap_or(0),
            time: get("time").parse().unwrap_or(0.0),
            n_steps: get("n_steps").parse().unwrap_or(1),
            range: {
                let r = parse_f32_list(get("range"));
                if r.len() == 2 {
                    [r[0], r[1]]
                } else {
                    finite_range(&values)
                }
            },
            location: get("location").to_string(),
            metadata: md.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            values,
        };
        frame.validate()?;
        Ok(frame)
    }
}

// Tiny JSON list codecs: the crate stays free of serde_json.
fn usize_list_json(v: &[usize]) -> String {
    let items: Vec<String> = v.iter().map(|v| v.to_string()).collect();
    format!("[{}]", items.join(","))
}

fn str_list_json(v: &[String]) -> String {
    let items: Vec<String> = v
        .iter()
        .map(|v| format!("\"{}\"", v.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect();
    format!("[{}]", items.join(","))
}

fn list_items(s: &str) -> impl Iterator<Item = &str> {
    s.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
}

fn parse_usize_list(s: &str) -> Vec<usize> {
    list_items(s).filter_map(|x| x.parse().ok()).collect()
}

fn parse_f32_list(s: &str) -> Vec<f32> {
    list_items(s).filter_map(|x| x.parse().ok()).collect()
}

fn parse_str_list(s: &str) -> Vec<String> {
    list_items(s)
        .map(|x| x.trim_matches('"').to_string())
        .collect()
}

/// Per-step statistics of one field, the basis of a colour range that stays put while a
/// timeline plays. Body of `GET .../samples/{sid}/field-stats?field=`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FieldStats {
    pub field: String,
    /// `null` = the value itself (scalars) or the vector magnitude; otherwise a component index.
    #[serde(default)]
    pub component: Option<usize>,
    pub n_steps: usize,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
    pub mean: Vec<f32>,
    /// 2nd / 98th percentile per step (from a strided subsample of at most 65,536 values).
    pub p02: Vec<f32>,
    pub p98: Vec<f32>,
    pub global: GlobalStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct GlobalStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    /// Smallest per-step `p02` and largest per-step `p98`: a robust range over all time.
    pub p02: f32,
    pub p98: f32,
}

impl FieldStats {
    /// Accumulate one step. `values` are the scalars that will be displayed.
    pub fn push_step(&mut self, values: &[f32]) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for &v in values {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
                sum += v as f64;
                n += 1;
            }
        }
        if n == 0 {
            lo = 0.0;
            hi = 0.0;
        }
        let stride = values.len().div_ceil(65_536).max(1);
        let mut sample: Vec<f32> = values
            .iter()
            .step_by(stride)
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        sample.sort_unstable_by(f32::total_cmp);
        let pick = |q: f32| -> f32 {
            if sample.is_empty() {
                0.0
            } else {
                sample[((sample.len() - 1) as f32 * q).round() as usize]
            }
        };
        self.min.push(lo);
        self.max.push(hi);
        self.mean
            .push(if n == 0 { 0.0 } else { (sum / n as f64) as f32 });
        self.p02.push(pick(0.02));
        self.p98.push(pick(0.98));
        self.n_steps = self.min.len();
    }

    /// Fill [`FieldStats::global`] from the per-step vectors.
    pub fn finish(&mut self) {
        let fold = |v: &[f32], init: f32, f: fn(f32, f32) -> f32| v.iter().copied().fold(init, f);
        if self.min.is_empty() {
            self.global = GlobalStats::default();
            return;
        }
        self.global = GlobalStats {
            min: fold(&self.min, f32::INFINITY, f32::min),
            max: fold(&self.max, f32::NEG_INFINITY, f32::max),
            mean: self.mean.iter().sum::<f32>() / self.mean.len() as f32,
            p02: fold(&self.p02, f32::INFINITY, f32::min),
            p98: fold(&self.p98, f32::NEG_INFINITY, f32::max),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip_keeps_shape_dims_and_range() {
        let frame = FieldFrame {
            field: "u".into(),
            values: (0..24).map(|x| x as f32).collect(),
            shape: vec![4, 3, 2],
            dims: vec!["x".into(), "y".into(), "component".into()],
            full_shape: vec![8, 6, 2],
            t: 3,
            time: 0.03,
            n_steps: 6,
            range: [0.0, 23.0],
            location: "grid".into(),
            metadata: BTreeMap::from([("stride".into(), "2".into())]),
        };
        let back = FieldFrame::from_ipc_stream(&frame.to_ipc_stream().unwrap()).unwrap();
        assert_eq!(back.values, frame.values);
        assert_eq!(back.shape, vec![4, 3, 2]);
        assert_eq!(back.dims, frame.dims);
        assert_eq!(back.full_shape, vec![8, 6, 2]);
        assert_eq!((back.t, back.n_steps), (3, 6));
        assert!((back.time - 0.03).abs() < 1e-12);
        assert_eq!(back.range, [0.0, 23.0]);
        assert_eq!(back.components(), 2);
        assert_eq!(back.spatial_shape(), vec![4, 3]);
        assert_eq!(back.metadata["stride"], "2");
    }

    #[test]
    fn frame_rejects_shape_mismatch() {
        let frame = FieldFrame {
            values: vec![0.0; 5],
            shape: vec![2, 3],
            dims: vec!["x".into(), "y".into()],
            ..Default::default()
        };
        assert!(frame.to_ipc_stream().is_err());
    }

    #[test]
    fn stats_ignore_non_finite_and_track_global_range() {
        let mut s = FieldStats::default();
        s.push_step(&[1.0, f32::NAN, 3.0]);
        s.push_step(&[-2.0, 0.0, f32::INFINITY]);
        s.finish();
        assert_eq!(s.n_steps, 2);
        assert_eq!(s.min, vec![1.0, -2.0]);
        assert_eq!(s.max, vec![3.0, 0.0]);
        assert_eq!(s.mean, vec![2.0, -1.0]);
        assert_eq!((s.global.min, s.global.max), (-2.0, 3.0));
        assert!(s.global.p02 >= -2.0 && s.global.p98 <= 3.0);
    }

    #[test]
    fn finite_range_of_empty_or_nan_is_zero() {
        assert_eq!(finite_range(&[]), [0.0, 0.0]);
        assert_eq!(finite_range(&[f32::NAN]), [0.0, 0.0]);
        assert_eq!(finite_range(&[2.0, -1.0]), [-1.0, 2.0]);
    }
}
