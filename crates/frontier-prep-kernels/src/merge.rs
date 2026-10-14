// SPDX-License-Identifier: Apache-2.0
//! Coincident-vertex merge, the seam merge (port spec 6b).

use std::cmp::Ordering;

use rayon::prelude::*;

use crate::error::KernelError;
use crate::real::{Real, numpy_cmp};
use crate::tables::check_faces_in;

/// Result of [`merge_vertices`].
#[derive(Debug, Clone, PartialEq)]
pub struct Merged {
    /// `(K, 3)` flat float32: the first occurrence of every group, groups in lexicographic
    /// order of their key rows.
    pub points: Vec<f32>,
    /// `(M', 3)` flat int32: remapped faces, those with two equal corners dropped.
    pub faces: Vec<i32>,
    /// `(N,)` group of every input vertex.
    pub point_map: Vec<i64>,
}

/// A key row with the vertex it came from.
#[derive(Clone, Copy)]
struct Row<K> {
    key: [K; 3],
    index: u64,
}

/// Group rows that compare equal, groups numbered in sorted row order; returns
/// `(first, inverse)` like `frontier_prep.ops.mesh._group_rows`. `cmp` orders two key
/// components, `differ` decides a group boundary between neighbours in the sorted order
/// (the two disagree only for NaN: all NaNs sort together and every one is its own group).
fn group_rows<K: Copy + Send + Sync>(
    mut rows: Vec<Row<K>>,
    cmp: impl Fn(K, K) -> Ordering + Sync,
    differ: impl Fn(K, K) -> bool,
) -> (Vec<u64>, Vec<i64>) {
    // ties fall back to the vertex index: the order is total (deterministic for any thread
    // count) and the first row of a group is its lowest index
    rows.par_sort_unstable_by(|a, b| {
        cmp(a.key[0], b.key[0])
            .then_with(|| cmp(a.key[1], b.key[1]))
            .then_with(|| cmp(a.key[2], b.key[2]))
            .then_with(|| a.index.cmp(&b.index))
    });
    let mut first = Vec::new();
    let mut inverse = vec![0i64; rows.len()];
    for (i, row) in rows.iter().enumerate() {
        let new = i == 0 || (0..3).any(|c| differ(rows[i - 1].key[c], row.key[c]));
        if new {
            first.push(row.index);
        }
        inverse[row.index as usize] = first.len() as i64 - 1;
    }
    (first, inverse)
}

/// Lexicographic grouping of integer rows (`(first, inverse)`): used for duplicate faces.
pub(crate) fn group_int_rows(
    keys: impl IndexedParallelIterator<Item = [i64; 3]>,
) -> (Vec<u64>, Vec<i64>) {
    let rows: Vec<Row<i64>> = keys
        .enumerate()
        .map(|(i, key)| Row {
            key,
            index: i as u64,
        })
        .collect();
    group_rows(rows, |a, b| a.cmp(&b), |a, b| a != b)
}

/// Merge coincident vertices: exactly when `tol <= 0`, else on a grid of size `tol`
/// (`round_half_even(points / tol)` per axis, the division in the point dtype).
///
/// Groups are the distinct key rows in lexicographic order; a group's representative is its
/// lowest vertex index, whose coordinates are kept verbatim. Exact keys compare numerically
/// (`-0.0 == 0.0`); a NaN coordinate makes its vertex a group of its own, sorted after every
/// number.
///
/// Returns `Ok(None)` when the quantised keys are not representable as int64 (non-finite
/// coordinates or a grid too fine for the coordinate range, or a `tol` that is not a positive
/// finite number in the point dtype): numpy's float-to-int conversion is platform-defined
/// there, so the caller keeps the reference implementation for such input.
pub fn merge_vertices<P: Real>(
    points: &[P],
    faces: &[i64],
    tol: f64,
) -> Result<Option<Merged>, KernelError> {
    if !points.len().is_multiple_of(3) {
        return Err(KernelError::Shape {
            name: "points",
            len: points.len(),
            width: 3,
        });
    }
    let n = points.len() / 3;
    check_faces_in(faces, n)?;
    let row = |i: usize| [points[3 * i], points[3 * i + 1], points[3 * i + 2]];

    let (first, inverse) = if tol <= 0.0 {
        let rows: Vec<Row<P>> = (0..n)
            .into_par_iter()
            .map(|i| Row {
                key: row(i),
                index: i as u64,
            })
            .collect();
        group_rows(rows, numpy_cmp, |a, b| a != b)
    } else {
        // a Python float divisor takes the array dtype (float32 points divide in float32)
        let t = P::from_f64(tol);
        if !(t > P::ZERO && t.is_finite()) {
            return Ok(None);
        }
        const LIMIT: f64 = 9.223_372_036_854_775e18;
        let quantise = |x: P| -> Option<i64> {
            let q = (x / t).round_ties_even().to_f64();
            (q.is_finite() && q.abs() < LIMIT).then_some(q as i64)
        };
        let rows: Option<Vec<Row<i64>>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let [x, y, z] = row(i);
                Some(Row {
                    key: [quantise(x)?, quantise(y)?, quantise(z)?],
                    index: i as u64,
                })
            })
            .collect();
        let Some(rows) = rows else {
            return Ok(None);
        };
        group_rows(rows, |a, b| a.cmp(&b), |a, b| a != b)
    };

    let new_points: Vec<f32> = first
        .par_iter()
        .flat_map_iter(|&i| row(i as usize).map(Real::to_f32))
        .collect();
    let new_faces: Vec<i32> = faces
        .par_chunks_exact(3)
        .flat_map_iter(|f| {
            let g = [
                inverse[f[0] as usize],
                inverse[f[1] as usize],
                inverse[f[2] as usize],
            ];
            let keep = g[0] != g[1] && g[1] != g[2] && g[0] != g[2];
            keep.then_some(g.map(|x| x as i32)).into_iter().flatten()
        })
        .collect();
    Ok(Some(Merged {
        points: new_points,
        faces: new_faces,
        point_map: inverse,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn merged<P: Real>(points: &[P], faces: &[i64], tol: f64) -> Merged {
        merge_vertices(points, faces, tol).unwrap().unwrap()
    }

    // -- port spec 6b, test vectors 1-3 ------------------------------------------------------

    #[test]
    fn vector_1_seam() {
        let p: [f32; 18] = [
            0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.,
        ];
        let m = merged(&p, &[0, 1, 2, 3, 5, 4], 0.0);
        assert_eq!(m.point_map, [0, 2, 1, 2, 1, 3]);
        assert_eq!(m.points, [0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0.]);
        assert_eq!(m.faces, [0, 2, 1, 2, 3, 1]);
        // the merged mesh has 4 open edges: the seam edge (1, 2) now has two faces
        let f64s: Vec<i64> = m.faces.iter().map(|&x| i64::from(x)).collect();
        let audit = crate::topology(&m.points, &f64s).unwrap();
        assert_eq!(audit.open_edges, 4);
    }

    #[test]
    fn vector_2_tolerance_and_collapse() {
        let p: [f32; 12] = [0., 0., 0., 0.2, 0., 0., 0., 1., 0., 1., 0., 0.];
        let m = merged(&p, &[0, 1, 2, 1, 3, 2], 0.5);
        assert_eq!(m.point_map, [0, 0, 1, 2]);
        assert_eq!(m.points, [0., 0., 0., 0., 1., 0., 1., 0., 0.]);
        assert_eq!(m.faces, [0, 2, 1]);
    }

    #[test]
    fn vector_3_negative_zero() {
        let p: [f32; 9] = [0., 0., 0., -0.0, 0., 0., 2., 1., 0.];
        let m = merged(&p, &[0, 1, 2], 0.0);
        assert_eq!(m.point_map, [0, 0, 1]);
        assert_eq!(m.points, [0., 0., 0., 2., 1., 0.]);
        assert!(m.faces.is_empty());
        // the representative keeps its own bits: vertex 0 is +0.0
        assert!(m.points[0].is_sign_positive());
    }

    // -- behaviour beyond the vectors ------------------------------------------------------------

    #[test]
    fn empty_input() {
        let m = merged::<f32>(&[], &[], 0.0);
        assert!(m.points.is_empty() && m.faces.is_empty() && m.point_map.is_empty());
        let m = merged::<f32>(&[], &[], 0.1);
        assert!(m.point_map.is_empty());
    }

    #[test]
    fn nan_rows_sort_last_and_never_merge() {
        // numpy lexsort puts NaN after every number; `!=` between sorted neighbours makes
        // each NaN row its own group even when the rows are identical
        let nan = f32::NAN;
        let p = [nan, 0., 0., 1., 0., 0., nan, 0., 0., 1., 0., 0., 0., 5., 5.];
        let m = merged(&p, &[0, 1, 4, 2, 3, 4], 0.0);
        assert_eq!(m.point_map, [2, 1, 3, 1, 0]);
        assert_eq!(m.points.len(), 12);
        assert!(m.points[6].is_nan() && m.points[9].is_nan());
        assert_eq!(m.faces, [2, 1, 0, 3, 1, 0]);
    }

    #[test]
    fn non_representable_keys_are_left_to_the_reference() {
        let p = [f32::NAN, 0., 0., 1., 0., 0., 0., 1., 0.];
        assert_eq!(merge_vertices(&p, &[0, 1, 2], 0.5).unwrap(), None);
        let p = [f32::INFINITY, 0., 0., 1., 0., 0., 0., 1., 0.];
        assert_eq!(merge_vertices(&p, &[0, 1, 2], 0.5).unwrap(), None);
        let p = [1.0e30_f32, 0., 0., 1., 0., 0., 0., 1., 0.];
        assert_eq!(merge_vertices(&p, &[0, 1, 2], 1.0e-9).unwrap(), None);
        // a tolerance that underflows to zero in float32
        assert_eq!(merge_vertices(&p, &[0, 1, 2], 1.0e-60).unwrap(), None);
        assert_eq!(merge_vertices(&p, &[0, 1, 2], f64::NAN).unwrap(), None);
    }

    #[test]
    fn ties_round_to_even_and_divide_in_the_point_dtype() {
        // 0.5 / 1 -> 0, 1.5 -> 2, 2.5 -> 2, -0.5 -> -0 -> 0
        let p: [f32; 12] = [0.5, 0., 0., 1.5, 0., 0., 2.5, 0., 0., -0.5, 0., 0.];
        let m = merged(&p, &[0, 1, 2], 1.0);
        assert_eq!(m.point_map, [0, 1, 1, 0]);
        // 0.35f32 / 0.1f32 rounds to 3.5 in float32 (-> 4, ties to even); the same value
        // divided in float64 is 3.4999999... (-> 3). numpy divides in the array dtype.
        let p32: [f32; 6] = [0.35, 0., 0., 0.3, 0., 0.];
        let p64: [f64; 6] = [f64::from(0.35_f32), 0., 0., 0.3, 0., 0.];
        assert_eq!(merged(&p32, &[], 0.1).point_map, [1, 0]);
        assert_eq!(merged(&p64, &[], 0.1).point_map, [0, 0]);
    }

    #[test]
    fn out_of_range_face_is_an_error() {
        let p: [f32; 3] = [0., 0., 0.];
        assert!(matches!(
            merge_vertices(&p, &[0, 0, 1], 0.0),
            Err(KernelError::IndexOutOfRange { .. })
        ));
    }

    #[test]
    fn identical_for_any_thread_count() {
        // many exact duplicates: the tie-break on the index decides the representative
        let p: Vec<f32> = (0..3 * 50_000)
            .map(|i| ((i * 7_919_usize) % 97) as f32 * 0.25)
            .collect();
        let f = crate::tables::tests::soup(40_000, 50_000, 11);
        let run = |threads, tol| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| merged(&p, &f, tol))
        };
        for tol in [0.0, 0.6] {
            let reference = run(1, tol);
            for threads in [2, 7] {
                assert_eq!(run(threads, tol), reference);
            }
        }
    }
}
