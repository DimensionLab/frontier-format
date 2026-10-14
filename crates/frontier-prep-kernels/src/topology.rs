// SPDX-License-Identifier: Apache-2.0
//! Topology audit of a triangle surface (port spec 6a).

use rayon::prelude::*;

use crate::error::KernelError;
use crate::real::Real;
use crate::sum::numpy_sum;
use crate::tables::{EdgeTables, check_faces_in, components_from_slots};

/// The audit of a mesh with at least one face. Field names and meanings are those of the
/// Python `topology()` dict; `genus_estimate`, `watertight` and `oriented` are derived.
#[derive(Debug, Clone, PartialEq)]
pub struct Audit {
    pub n_points: usize,
    pub n_faces: usize,
    pub n_edges: usize,
    pub open_edges: usize,
    pub nonmanifold_edges: usize,
    pub inconsistent_edges: usize,
    pub degenerate_faces: usize,
    pub components: usize,
    pub euler: i64,
    pub area: f64,
    pub boundary_length: f64,
    /// `[xmin, xmax, ymin, ymax, zmin, zmax]`: float32 extrema widened to float64.
    pub bounds: [f64; 6],
    pub signed_volume: f64,
}

impl Audit {
    /// `(2 * components - euler) / 2` for a closed surface, `None` when edges are open.
    pub fn genus_estimate(&self) -> Option<f64> {
        (self.open_edges == 0).then(|| (2.0 * self.components as f64 - self.euler as f64) / 2.0)
    }

    pub fn watertight(&self) -> bool {
        self.open_edges == 0 && self.nonmanifold_edges == 0
    }

    pub fn oriented(&self) -> bool {
        self.inconsistent_edges == 0
    }
}

/// The three corners of face `i` as float64, each coordinate converted by `conv`.
#[inline]
fn corners<P: Real>(
    points: &[P],
    faces: &[i64],
    i: usize,
    conv: impl Fn(P) -> f64,
) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (c, corner) in out.iter_mut().enumerate() {
        let v = 3 * faces[3 * i + c] as usize;
        *corner = [conv(points[v]), conv(points[v + 1]), conv(points[v + 2])];
    }
    out
}

/// `np.cross(a, b)`: products and differences as separate IEEE operations (no FMA).
#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Area of one face from float32 coordinates: `0.5 * |(p1 - p0) x (p2 - p0)|` in float64,
/// the squares added left to right.
#[inline]
fn face_area<P: Real>(points: &[P], faces: &[i64], i: usize) -> f64 {
    // `SurfaceMesh` stores float32 points whatever the caller passed
    let [p0, p1, p2] = corners(points, faces, i, |x| f64::from(x.to_f32()));
    let c = cross(sub(p1, p0), sub(p2, p0));
    0.5 * ((c[0] * c[0] + c[1] * c[1]) + c[2] * c[2]).sqrt()
}

/// Per-face areas (float64, `(M,)`), the values of `SurfaceMesh.face_areas()`.
pub fn face_areas<P: Real>(points: &[P], faces: &[i64]) -> Result<Vec<f64>, KernelError> {
    let m = check_inputs(points, faces)?;
    Ok(areas_unchecked(points, faces, m))
}

fn areas_unchecked<P: Real>(points: &[P], faces: &[i64], m: usize) -> Vec<f64> {
    (0..m)
        .into_par_iter()
        .map(|i| face_area(points, faces, i))
        .collect()
}

fn check_inputs<P: Real>(points: &[P], faces: &[i64]) -> Result<usize, KernelError> {
    if !points.len().is_multiple_of(3) {
        return Err(KernelError::Shape {
            name: "points",
            len: points.len(),
            width: 3,
        });
    }
    check_faces_in(faces, points.len() / 3)
}

/// Minimum and maximum of one coordinate as numpy computes them on float32: NaN wins, and
/// `-0.0` orders below `+0.0`.
fn column_bounds<P: Real>(points: &[P], axis: usize) -> (f32, f32) {
    let mut it = points.iter().skip(axis).step_by(3).map(|x| x.to_f32());
    let Some(first) = it.next() else {
        return (0.0, 0.0);
    };
    let (mut lo, mut hi) = (first, first);
    let mut nan = first.is_nan();
    for x in it {
        nan |= x.is_nan();
        if x < lo || (x == lo && x.is_sign_negative()) {
            lo = x;
        }
        if x > hi || (x == hi && x.is_sign_positive()) {
            hi = x;
        }
    }
    if nan { (f32::NAN, f32::NAN) } else { (lo, hi) }
}

/// Audit `faces` (flat `3M` indices, `M > 0` expected by callers that mirror the Python
/// dict; an empty mesh yields an all-zero audit) over `points` (flat `3N` coordinates).
pub fn topology<P: Real>(points: &[P], faces: &[i64]) -> Result<Audit, KernelError> {
    let m = check_inputs(points, faces)?;
    let n_points = points.len() / 3;
    let t = EdgeTables::build(faces)?;
    let n_edges = t.n_edges();

    let open_edges = t.counts.par_iter().filter(|&&c| c == 1).count();
    let nonmanifold_edges = t.counts.par_iter().filter(|&&c| c > 2).count();

    // directed edges used more than once: per edge, slots going min -> max and max -> min
    let inconsistent_edges: usize = if m == 0 {
        0
    } else {
        (0..n_edges)
            .into_par_iter()
            .map(|e| {
                let lo = t.uniq[2 * e];
                let run = t.run(e);
                let n_fwd = run
                    .iter()
                    .filter(|&&k| {
                        let k = k as usize;
                        faces[3 * (k % m) + k / m] == lo
                    })
                    .count();
                usize::from(n_fwd > 1) + usize::from(run.len() - n_fwd > 1)
            })
            .sum()
    };

    let areas = areas_unchecked(points, faces, m);
    let degenerate_faces = (0..m)
        .into_par_iter()
        .filter(|&i| {
            let (a, b, c) = (faces[3 * i], faces[3 * i + 1], faces[3 * i + 2]);
            a == b || b == c || a == c || areas[i] <= 0.0
        })
        .count();
    let area = numpy_sum(&areas);
    drop(areas);

    let components = components_from_slots(m, &t.inv)?
        .into_iter()
        .max()
        .map_or(0, |x| x as usize + 1);

    let mut used = vec![false; t.n as usize];
    for &v in faces {
        used[v as usize] = true;
    }
    let n_used = used.iter().filter(|&&u| u).count();
    let euler = n_used as i64 - n_edges as i64 + m as i64;

    // open edges in edge order, in the arithmetic of the point dtype (the reference
    // subtracts the point rows as they are: float32 for dataset arrays)
    let lengths: Vec<P> = (0..n_edges)
        .into_par_iter()
        .filter(|&e| t.counts[e] == 1)
        .map(|e| {
            let (u, v) = (3 * t.uniq[2 * e] as usize, 3 * t.uniq[2 * e + 1] as usize);
            let d = [
                points[u] - points[v],
                points[u + 1] - points[v + 1],
                points[u + 2] - points[v + 2],
            ];
            ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt()
        })
        .collect();
    let boundary_length = if lengths.is_empty() {
        0.0
    } else {
        numpy_sum(&lengths).to_f64()
    };

    let mut bounds = [0.0; 6];
    for axis in 0..3 {
        let (lo, hi) = column_bounds(points, axis);
        bounds[2 * axis] = f64::from(lo);
        bounds[2 * axis + 1] = f64::from(hi);
    }

    // dot(p0, cross(p1, p2)) per face, the products added left to right
    let triple: Vec<f64> = (0..m)
        .into_par_iter()
        .map(|i| {
            let [a, b, c] = corners(points, faces, i, Real::to_f64);
            let x = cross(b, c);
            (a[0] * x[0] + a[1] * x[1]) + a[2] * x[2]
        })
        .collect();
    let signed_volume = numpy_sum(&triple) / 6.0;

    Ok(Audit {
        n_points,
        n_faces: m,
        n_edges,
        open_edges,
        nonmanifold_edges,
        inconsistent_edges,
        degenerate_faces,
        components,
        euler,
        area,
        boundary_length,
        bounds,
        signed_volume,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQUARE: [f32; 12] = [0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.];

    // -- port spec 6a, test vectors 1-3 ------------------------------------------------------

    #[test]
    fn vector_1_two_triangles() {
        let a = topology(&SQUARE, &[0, 1, 2, 2, 1, 3]).unwrap();
        assert_eq!((a.n_points, a.n_faces, a.n_edges), (4, 2, 5));
        assert_eq!((a.open_edges, a.nonmanifold_edges, a.euler), (4, 0, 1));
        assert_eq!((a.components, a.inconsistent_edges), (1, 0));
        assert_eq!(a.degenerate_faces, 0);
        assert_eq!(a.area, 1.0);
        assert_eq!(a.boundary_length, 4.0);
        assert_eq!(a.signed_volume, 0.0);
        assert_eq!(a.bounds, [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        assert!(!a.watertight() && a.oriented());
        assert_eq!(a.genus_estimate(), None);
    }

    #[test]
    fn vector_2_same_winding() {
        let a = topology(&SQUARE, &[0, 1, 2, 1, 2, 3]).unwrap();
        assert_eq!(a.inconsistent_edges, 1);
        assert!(!a.oriented());
    }

    #[test]
    fn vector_3_nonmanifold_fan() {
        let p: [f32; 15] = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., -1., 0.];
        let a = topology(&p, &[0, 1, 2, 0, 1, 3, 1, 0, 4]).unwrap();
        assert_eq!((a.open_edges, a.nonmanifold_edges, a.euler), (6, 1, 1));
        assert_eq!((a.components, a.inconsistent_edges), (1, 1));
        assert_eq!(a.area, 1.5);
        // 3 + 3 * sqrt(2) summed in float32 (the point dtype), in edge order
        assert_eq!(a.boundary_length, 7.242_640_972_137_451);
        assert_eq!(a.signed_volume, 0.0);
    }

    // -- behaviour beyond the vectors ------------------------------------------------------------

    #[test]
    fn closed_tetrahedron() {
        let p: [f32; 12] = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let f = [0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2];
        let a = topology(&p, &f).unwrap();
        assert!(a.watertight() && a.oriented());
        assert_eq!((a.euler, a.components), (2, 1));
        assert_eq!(a.genus_estimate(), Some(0.0));
        assert_eq!(a.boundary_length, 0.0);
        assert_eq!(a.signed_volume, 1.0 / 6.0);
    }

    #[test]
    fn float64_points_use_float64_for_boundary_and_volume() {
        let p64: Vec<f64> = SQUARE.iter().map(|&x| f64::from(x) * 0.1).collect();
        let a = topology(&p64, &[0, 1, 2, 2, 1, 3]).unwrap();
        // 4 edges of length 0.1 in float64; the float32 path would give 0.4000000059604645
        assert_eq!(a.boundary_length, 0.4);
        // areas and bounds always go through float32
        assert_eq!(a.bounds[1], f64::from(0.1_f32));
    }

    #[test]
    fn degenerate_and_repeated_faces() {
        // face 1 repeats an index, face 2 is a zero-area sliver (collinear points)
        let p: [f32; 15] = [0., 0., 0., 1., 0., 0., 0., 1., 0., 2., 0., 0., 3., 0., 0.];
        let a = topology(&p, &[0, 1, 2, 1, 1, 2, 0, 1, 3]).unwrap();
        assert_eq!(a.degenerate_faces, 2);
    }

    #[test]
    fn nan_coordinates_propagate() {
        let mut p = SQUARE;
        p[3] = f32::NAN;
        let a = topology(&p, &[0, 1, 2, 2, 1, 3]).unwrap();
        assert!(a.area.is_nan() && a.boundary_length.is_nan() && a.signed_volume.is_nan());
        assert!(a.bounds[0].is_nan() && a.bounds[1].is_nan());
        assert_eq!(a.bounds[2..], [0.0, 1.0, 0.0, 0.0]);
        // NaN areas are not `<= 0`
        assert_eq!(a.degenerate_faces, 0);
        assert_eq!(a.open_edges, 4);
    }

    #[test]
    fn signed_zero_bounds() {
        let p: [f32; 6] = [0.0, -0.0, 0.0, -0.0, 0.0, 0.0];
        assert!(column_bounds(&p, 0).0.is_sign_negative());
        assert!(column_bounds(&p, 0).1.is_sign_positive());
        assert!(column_bounds(&p, 1).0.is_sign_negative());
        assert!(column_bounds(&p, 1).1.is_sign_positive());
        let q: [f32; 6] = [-0.0; 6];
        assert!(column_bounds(&q, 2).1.is_sign_negative());
    }

    #[test]
    fn out_of_range_index_is_an_error() {
        assert_eq!(
            topology(&SQUARE, &[0, 1, 4]),
            Err(KernelError::IndexOutOfRange {
                face: 0,
                index: 4,
                n_points: 4
            })
        );
        assert!(matches!(
            face_areas(&SQUARE[..5], &[0, 0, 0]),
            Err(KernelError::Shape { .. })
        ));
    }

    #[test]
    fn empty_mesh_is_all_zero() {
        let a = topology(&SQUARE, &[]).unwrap();
        assert_eq!(
            (a.n_points, a.n_faces, a.n_edges, a.components),
            (4, 0, 0, 0)
        );
        assert_eq!(a.area, 0.0);
    }

    #[test]
    fn identical_for_any_thread_count() {
        let f = crate::tables::tests::soup(30_000, 4_000, 5);
        let p: Vec<f32> = (0..3 * 4_000)
            .map(|i| ((i * 2_654_435_761_usize) % 10_007) as f32 * 1.0e-3)
            .collect();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| topology(&p, &f).unwrap())
        };
        let reference = run(1);
        for threads in [2, 5, 8] {
            let a = run(threads);
            assert_eq!(a, reference);
            assert_eq!(a.area.to_bits(), reference.area.to_bits());
            assert_eq!(a.signed_volume.to_bits(), reference.signed_volume.to_bits());
        }
    }
}
