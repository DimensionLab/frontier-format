// SPDX-License-Identifier: Apache-2.0
//! Orientation consistency and duplicate faces (port spec 6c).

use std::collections::VecDeque;

use rayon::prelude::*;

use crate::error::KernelError;
use crate::merge::group_int_rows;
use crate::tables::{EdgeTables, check_faces};

const NONE: u32 = u32::MAX;

/// Flip faces so neighbours traverse shared manifold edges in opposite directions.
/// Returns the faces as flat int32 and the number of flipped faces.
///
/// Rules (identical to the Python reference): adjacency is over edges with exactly two
/// slots; the relation value of a face pair is the sum over their shared manifold edges of
/// 1 (opposite traversal) or 2 (same traversal) and the pair disagrees iff that sum is 2;
/// the root of each component is its lowest face index and keeps its winding; every other
/// face takes `flip[pred] ^ disagree(pred, face)` with `pred` its predecessor in a
/// breadth-first walk that visits neighbours in ascending face index.
pub fn orient_consistently(faces: &[i64]) -> Result<(Vec<i32>, usize), KernelError> {
    let m = check_faces(faces)?;
    let mut out: Vec<i32> = faces.par_iter().map(|&v| v as i32).collect();
    if m == 0 {
        return Ok((out, 0));
    }
    let t = EdgeTables::build(faces)?;
    orient_with(&t, faces, &mut out).map(|n| (out, n))
}

fn orient_with(t: &EdgeTables, faces: &[i64], out: &mut [i32]) -> Result<usize, KernelError> {
    let m = t.m;
    // neighbour across each of a face's three edge slots (NONE unless the edge is manifold)
    // and whether the two faces traverse that edge the same way
    let wide = m > NONE as usize;
    let mut nbr = vec![NONE; if wide { 0 } else { 3 * m }];
    let mut nbr_wide = vec![u64::MAX; if wide { 3 * m } else { 0 }];
    let mut same = vec![false; 3 * m];
    for e in 0..t.n_edges() {
        if t.counts[e] != 2 {
            continue;
        }
        let run = t.run(e);
        let (ka, kb) = (run[0] as usize, run[1] as usize);
        let (a, sa, b, sb) = (ka % m, ka / m, kb % m, kb / m);
        let s = faces[3 * a + sa] == faces[3 * b + sb];
        same[3 * a + sa] = s;
        same[3 * b + sb] = s;
        if wide {
            nbr_wide[3 * a + sa] = b as u64;
            nbr_wide[3 * b + sb] = a as u64;
        } else {
            nbr[3 * a + sa] = b as u32;
            nbr[3 * b + sb] = a as u32;
        }
    }
    let neighbour = |i: usize| -> Option<usize> {
        if wide {
            (nbr_wide[i] != u64::MAX).then(|| nbr_wide[i] as usize)
        } else {
            (nbr[i] != NONE).then(|| nbr[i] as usize)
        }
    };

    let mut visited = vec![false; m];
    let mut flip = vec![false; m];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for root in 0..m {
        if visited[root] {
            continue;
        }
        visited[root] = true;
        queue.push_back(root);
        while let Some(u) = queue.pop_front() {
            // up to three (neighbour, relation sum) pairs, merged per neighbour, ascending
            let mut adj = [(usize::MAX, 0u8); 3];
            let mut len = 0;
            for s in 0..3 {
                let Some(v) = neighbour(3 * u + s) else {
                    continue;
                };
                let val = if same[3 * u + s] { 2 } else { 1 };
                if let Some(slot) = adj[..len].iter_mut().find(|x| x.0 == v) {
                    slot.1 += val;
                } else {
                    adj[len] = (v, val);
                    len += 1;
                }
            }
            adj[..len].sort_unstable_by_key(|x| x.0);
            for &(v, rel) in &adj[..len] {
                if !visited[v] {
                    visited[v] = true;
                    flip[v] = flip[u] ^ (rel == 2);
                    queue.push_back(v);
                }
            }
        }
    }
    let mut n_flipped = 0;
    for (i, _) in flip.iter().enumerate().filter(|x| *x.1) {
        out.swap(3 * i + 1, 3 * i + 2);
        n_flipped += 1;
    }
    Ok(n_flipped)
}

/// Mask of faces whose vertex set repeats an earlier face's (the first occurrence is kept).
pub fn duplicate_faces(faces: &[i64]) -> Result<Vec<bool>, KernelError> {
    if !faces.len().is_multiple_of(3) {
        return Err(KernelError::Shape {
            name: "faces",
            len: faces.len(),
            width: 3,
        });
    }
    let keys = faces.par_chunks_exact(3).map(|f| {
        let mut k = [f[0], f[1], f[2]];
        k.sort_unstable();
        k
    });
    let (first, _) = group_int_rows(keys);
    let mut dup = vec![true; faces.len() / 3];
    for i in first {
        dup[i as usize] = false;
    }
    Ok(dup)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orient(f: &[i64]) -> (Vec<i32>, usize) {
        orient_consistently(f).unwrap()
    }

    // -- port spec 6c, test vectors 1-4 and the duplicate-face vector ------------------------

    #[test]
    fn vector_1_flipped_interior_face() {
        let (f, n) = orient(&[0, 1, 2, 1, 3, 2, 2, 4, 3, 3, 5, 4]);
        assert_eq!(f, [0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4]);
        assert_eq!(n, 1);
    }

    #[test]
    fn vector_2_flipped_root_keeps_its_winding() {
        let (f, n) = orient(&[0, 2, 1, 1, 3, 2, 2, 3, 4, 3, 5, 4]);
        assert_eq!(f, [0, 2, 1, 1, 2, 3, 2, 4, 3, 3, 4, 5]);
        assert_eq!(n, 3);
    }

    #[test]
    fn vector_3_two_components() {
        let (f, n) = orient(&[4, 6, 5, 0, 1, 2, 1, 3, 2, 4, 7, 5]);
        assert_eq!(f, [4, 6, 5, 0, 1, 2, 1, 3, 2, 4, 5, 7]);
        assert_eq!(n, 1);
    }

    #[test]
    fn vector_4_nonmanifold_fan_connects_nothing() {
        let (f, n) = orient(&[0, 1, 2, 0, 1, 3, 1, 0, 4]);
        assert_eq!(f, [0, 1, 2, 0, 1, 3, 1, 0, 4]);
        assert_eq!(n, 0);
    }

    #[test]
    fn vector_duplicate_faces() {
        let d = duplicate_faces(&[0, 1, 2, 2, 0, 1, 1, 2, 3, 0, 2, 1]).unwrap();
        assert_eq!(d, [false, true, false, true]);
    }

    // -- behaviour beyond the vectors ------------------------------------------------------------

    #[test]
    fn empty_and_single() {
        assert_eq!(orient(&[]), (vec![], 0));
        assert_eq!(orient(&[3, 1, 2]), (vec![3, 1, 2], 0));
        assert!(duplicate_faces(&[]).unwrap().is_empty());
    }

    // Expected values below were produced by the Python reference kernel
    // (`packages/frontier-prep/tests/_reference_kernels.py`).

    #[test]
    fn relation_sum_rule_on_duplicate_faces() {
        // faces 0 and 1 cover the same vertices; face 2 makes edge (0, 1) non-manifold, so
        // the pair shares two manifold edges. Opposite winding on both gives 1 + 1 = 2,
        // which the rule reads as "disagree": face 1 is flipped onto face 0's winding.
        assert_eq!(
            orient(&[0, 1, 2, 0, 2, 1, 0, 1, 3]),
            (vec![0, 1, 2, 0, 1, 2, 0, 1, 3], 1)
        );
        assert_eq!(
            orient(&[0, 1, 2, 2, 1, 0, 0, 1, 3]),
            (vec![0, 1, 2, 2, 0, 1, 0, 1, 3], 1)
        );
        // same winding on both shared edges: 2 + 2 = 4, no flip
        assert_eq!(orient(&[0, 1, 2, 0, 1, 2, 0, 1, 3]).1, 0);
        // without the third face all three edges are manifold: sums 3 or 6, never 2
        assert_eq!(orient(&[0, 1, 2, 0, 2, 1]).1, 0);
        assert_eq!(orient(&[0, 1, 2, 0, 1, 2]).1, 0);
        // a repeated index puts two slots of one face on edge (0, 1): a self-pair, ignored
        assert_eq!(orient(&[0, 1, 0, 5, 6, 7]).1, 0);
    }

    #[test]
    fn mobius_strip_follows_the_bfs_tree() {
        // non-orientable: the walk order decides which faces flip, so it is the contract
        let f = [
            0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 0, 3, 0, 4,
        ];
        let expect = vec![
            0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 3, 7, 3, 0, 7, 3, 4, 0,
        ];
        assert_eq!(orient(&f), (expect, 3));
    }

    #[test]
    fn output_wraps_to_int32_like_numpy() {
        let big = i64::from(i32::MAX) + 2;
        let (f, _) = orient(&[0, 1, big]);
        assert_eq!(f[2], i32::MIN + 1);
    }

    #[test]
    fn identical_for_any_thread_count() {
        let f = crate::tables::tests::soup(30_000, 12_000, 21);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| (orient(&f), duplicate_faces(&f).unwrap()))
        };
        let reference = run(1);
        for threads in [2, 6] {
            assert_eq!(run(threads), reference);
        }
    }
}
