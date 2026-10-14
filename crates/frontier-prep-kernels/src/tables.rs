// SPDX-License-Identifier: Apache-2.0
//! Sorted-edge tables of a triangle mesh and face components (port spec 6a).

use rayon::prelude::*;

use crate::error::KernelError;

/// Sorted-edge tables, the same arrays as `frontier_prep.ops.mesh.TopologyTables`.
///
/// Slot `k` of the `3M` directed edge slots belongs to face `k % M` and edge slot `k / M`
/// (slot 0 is `(f0, f1)`, 1 is `(f1, f2)`, 2 is `(f2, f0)`). Edges are the distinct
/// undirected pairs `(min, max)` in lexicographic order; `order` lists the slots by edge and,
/// inside one edge, by ascending slot index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeTables {
    /// Number of faces `M`.
    pub m: usize,
    /// Packing base of the Python key: `max(faces) + 1` (1 for an empty mesh).
    pub n: i64,
    /// `(3M,)` slot indices sorted by (edge, slot index).
    pub order: Vec<i64>,
    /// `(3M,)` edge id of each slot.
    pub inv: Vec<i64>,
    /// `(E,)` first position of each edge's run in `order`.
    pub start: Vec<i64>,
    /// `(E,)` number of slots (faces) on each edge.
    pub counts: Vec<i64>,
    /// `(E, 2)` flat: the undirected edges `(min, max)`.
    pub uniq: Vec<i64>,
}

/// One slot inside the bucket of its lower vertex: `(higher vertex, slot index)`.
type Entry = (u64, u64);

/// Above this many vertices per slot the counting sort's offset table would dominate; the
/// comparison sort takes over (identical output).
const SPARSE_FACTOR: usize = 4;
const SPARSE_SLACK: usize = 1 << 20;
/// Buckets up to this length are sorted by insertion (they arrive ordered by slot index).
const INSERTION_MAX: usize = 24;

impl EdgeTables {
    /// Build the tables of `faces` (flat `3M` indices, all non-negative).
    pub fn build(faces: &[i64]) -> Result<Self, KernelError> {
        let m = check_faces(faces)?;
        if m == 0 {
            return Ok(Self {
                m: 0,
                n: 1,
                order: Vec::new(),
                inv: Vec::new(),
                start: Vec::new(),
                counts: Vec::new(),
                uniq: Vec::new(),
            });
        }
        let max = faces.par_iter().copied().max().unwrap_or(0);
        let n = max as usize + 1;
        if n <= SPARSE_FACTOR * 3 * m + SPARSE_SLACK {
            Ok(Self::build_counting(faces, m, n))
        } else {
            Ok(Self::build_sorting(faces, m, n))
        }
    }

    /// Counting sort on the lower vertex, then a sort of every (small) bucket on
    /// `(higher vertex, slot)`. Linear in the number of slots.
    fn build_counting(faces: &[i64], m: usize, n: usize) -> Self {
        let slots = 3 * m;
        let mut offs = vec![0usize; n + 1];
        for k in 0..slots {
            let (lo, _) = slot_edge(faces, m, k);
            offs[lo as usize + 1] += 1;
        }
        for v in 0..n {
            offs[v + 1] += offs[v];
        }
        let mut cursor = offs[..n].to_vec();
        let mut entries: Vec<Entry> = vec![(0, 0); slots];
        for k in 0..slots {
            let (lo, hi) = slot_edge(faces, m, k);
            let c = &mut cursor[lo as usize];
            entries[*c] = (hi, k as u64);
            *c += 1;
        }
        drop(cursor);
        sort_buckets(&mut entries, &offs);
        let it = (0..n).flat_map(|lo| {
            entries[offs[lo]..offs[lo + 1]]
                .iter()
                .map(move |&(hi, k)| (lo as u64, hi, k))
        });
        Self::finish(it, m, n)
    }

    /// Comparison sort of `(lower, higher, slot)` triples: used when the vertex ids are
    /// sparse (a few faces indexing into a huge point array).
    fn build_sorting(faces: &[i64], m: usize, n: usize) -> Self {
        let mut triples: Vec<(u64, u64, u64)> = (0..3 * m)
            .into_par_iter()
            .map(|k| {
                let (lo, hi) = slot_edge(faces, m, k);
                (lo, hi, k as u64)
            })
            .collect();
        // the slot index makes every triple distinct: the order is total, so an unstable
        // parallel sort is deterministic
        triples.par_sort_unstable();
        Self::finish(triples.into_iter(), m, n)
    }

    fn finish(sorted: impl Iterator<Item = (u64, u64, u64)>, m: usize, n: usize) -> Self {
        let slots = 3 * m;
        let mut order = Vec::with_capacity(slots);
        let mut inv = vec![0i64; slots];
        let mut start: Vec<i64> = Vec::with_capacity(slots / 2 + 1);
        let mut counts: Vec<i64> = Vec::with_capacity(slots / 2 + 1);
        let mut uniq: Vec<i64> = Vec::with_capacity(slots + 2);
        let mut prev: Option<(u64, u64)> = None;
        for (pos, (lo, hi, k)) in sorted.enumerate() {
            if prev != Some((lo, hi)) {
                prev = Some((lo, hi));
                start.push(pos as i64);
                counts.push(0);
                uniq.push(lo as i64);
                uniq.push(hi as i64);
            }
            if let Some(c) = counts.last_mut() {
                *c += 1;
            }
            inv[k as usize] = counts.len() as i64 - 1;
            order.push(k as i64);
        }
        Self {
            m,
            n: n as i64,
            order,
            inv,
            start,
            counts,
            uniq,
        }
    }

    /// Number of distinct undirected edges.
    pub fn n_edges(&self) -> usize {
        self.counts.len()
    }

    /// The slots of edge `e`, in ascending slot index.
    #[inline]
    pub fn run(&self, e: usize) -> &[i64] {
        let s = self.start[e] as usize;
        &self.order[s..s + self.counts[e] as usize]
    }
}

/// `(min, max)` of the directed edge at slot `k`.
#[inline]
fn slot_edge(faces: &[i64], m: usize, k: usize) -> (u64, u64) {
    let (face, s) = (k % m, k / m);
    let u = faces[3 * face + s] as u64;
    let v = faces[3 * face + (s + 1) % 3] as u64;
    if u <= v { (u, v) } else { (v, u) }
}

/// Number of faces, after checking the shape and that no index is negative.
pub(crate) fn check_faces(faces: &[i64]) -> Result<usize, KernelError> {
    if !faces.len().is_multiple_of(3) {
        return Err(KernelError::Shape {
            name: "faces",
            len: faces.len(),
            width: 3,
        });
    }
    if let Some(pos) = faces.par_iter().position_first(|&v| v < 0) {
        return Err(KernelError::NegativeIndex {
            face: pos / 3,
            index: faces[pos],
        });
    }
    Ok(faces.len() / 3)
}

/// As [`check_faces`], and every index is below `n_points`.
pub(crate) fn check_faces_in(faces: &[i64], n_points: usize) -> Result<usize, KernelError> {
    let m = check_faces(faces)?;
    if let Some(pos) = faces
        .par_iter()
        .position_first(|&v| v as u64 >= n_points as u64)
    {
        return Err(KernelError::IndexOutOfRange {
            face: pos / 3,
            index: faces[pos],
            n_points,
        });
    }
    Ok(m)
}

/// Sort every bucket `entries[offs[v]..offs[v + 1]]` on `(hi, slot)`. Buckets are disjoint,
/// so groups of buckets are sorted in parallel without affecting the result.
fn sort_buckets(entries: &mut [Entry], offs: &[usize]) {
    let n = offs.len() - 1;
    let tasks = (rayon::current_num_threads() * 8).clamp(1, n.max(1));
    let mut parts: Vec<(usize, usize, &mut [Entry])> = Vec::with_capacity(tasks);
    let mut rest = entries;
    let mut consumed = 0usize;
    for t in 0..tasks {
        let (v0, v1) = (n * t / tasks, n * (t + 1) / tasks);
        let (head, tail) = rest.split_at_mut(offs[v1] - consumed);
        consumed = offs[v1];
        rest = tail;
        parts.push((v0, v1, head));
    }
    parts.into_par_iter().for_each(|(v0, v1, part)| {
        let base = offs[v0];
        for v in v0..v1 {
            let bucket = &mut part[offs[v] - base..offs[v + 1] - base];
            if bucket.len() <= INSERTION_MAX {
                insertion_sort(bucket);
            } else {
                bucket.sort_unstable();
            }
        }
    });
}

fn insertion_sort(a: &mut [Entry]) {
    for i in 1..a.len() {
        let x = a[i];
        let mut j = i;
        while j > 0 && a[j - 1] > x {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = x;
    }
}

/// Connected components of faces through shared slot ids (`inv[k]` is the id of slot `k`,
/// three slots per face, any non-negative ids), labelled by first appearance in face order:
/// face 0 has label 0, the next face in a new component gets 1, and so on. Same result as
/// the bipartite face/edge `connected_components` of the Python reference.
pub fn components_from_slots(m: usize, inv: &[i64]) -> Result<Vec<i64>, KernelError> {
    if inv.len() != 3 * m {
        return Err(KernelError::SlotCount {
            expected: 3 * m,
            got: inv.len(),
        });
    }
    if m == 0 {
        return Ok(Vec::new());
    }
    if let Some(slot) = inv.iter().position(|&id| id < 0) {
        return Err(KernelError::NegativeSlotId {
            slot,
            id: inv[slot],
        });
    }
    let ne = inv.iter().copied().max().unwrap_or(0) as usize + 1;
    const NONE: u64 = u64::MAX;
    let mut first = vec![NONE; ne];
    let mut uf = UnionFind::new(m);
    for (k, &id) in inv.iter().enumerate() {
        let face = k % m;
        let f0 = &mut first[id as usize];
        if *f0 == NONE {
            *f0 = face as u64;
        } else {
            uf.union(face, *f0 as usize);
        }
    }
    drop(first);
    let mut label_of_root = vec![-1i64; m];
    let mut labels = vec![0i64; m];
    let mut next = 0i64;
    for (face, label) in labels.iter_mut().enumerate() {
        let root = uf.find(face);
        if label_of_root[root] < 0 {
            label_of_root[root] = next;
            next += 1;
        }
        *label = label_of_root[root];
    }
    Ok(labels)
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.parent[p];
            x = self.parent[x];
        }
        x
    }

    /// The lower index becomes the root; the labels are renumbered afterwards, so the choice
    /// only bounds the tree depth in practice (path halving does the rest).
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra < rb {
            self.parent[rb] = ra;
        } else if rb < ra {
            self.parent[ra] = rb;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn flat(f: &[[i64; 3]]) -> Vec<i64> {
        f.iter().flatten().copied().collect()
    }

    // -- port spec 6a, test vectors 1-3 ------------------------------------------------------

    #[test]
    fn vector_1_two_triangles() {
        let t = EdgeTables::build(&flat(&[[0, 1, 2], [2, 1, 3]])).unwrap();
        assert_eq!(t.n, 4);
        assert_eq!(t.order, [0, 4, 1, 2, 3, 5]);
        assert_eq!(t.inv, [0, 2, 2, 3, 1, 4]);
        assert_eq!(t.start, [0, 1, 2, 4, 5]);
        assert_eq!(t.counts, [1, 1, 2, 1, 1]);
        assert_eq!(t.uniq, [0, 1, 0, 2, 1, 2, 1, 3, 2, 3]);
        assert_eq!(components_from_slots(2, &t.inv).unwrap(), [0, 0]);
    }

    #[test]
    fn vector_2_same_winding() {
        let t = EdgeTables::build(&flat(&[[0, 1, 2], [1, 2, 3]])).unwrap();
        assert_eq!(t.order, [0, 4, 1, 2, 5, 3]);
        assert_eq!(t.inv, [0, 2, 2, 4, 1, 3]);
        assert_eq!(t.counts, [1, 1, 2, 1, 1]);
    }

    #[test]
    fn vector_3_nonmanifold_fan() {
        let t = EdgeTables::build(&flat(&[[0, 1, 2], [0, 1, 3], [1, 0, 4]])).unwrap();
        assert_eq!(t.uniq, [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4]);
        assert_eq!(t.counts, [3, 1, 1, 1, 1, 1, 1]);
        assert_eq!(t.order, [0, 1, 2, 6, 7, 5, 3, 4, 8]);
        assert_eq!(t.inv, [0, 0, 0, 4, 5, 3, 1, 2, 6]);
        assert_eq!(components_from_slots(3, &t.inv).unwrap(), [0, 0, 0]);
    }

    // -- behaviour beyond the vectors ------------------------------------------------------------

    #[test]
    fn empty_mesh() {
        let t = EdgeTables::build(&[]).unwrap();
        assert_eq!((t.m, t.n, t.n_edges()), (0, 1, 0));
        assert!(components_from_slots(0, &[]).unwrap().is_empty());
    }

    #[test]
    fn malformed_input_is_an_error() {
        assert!(matches!(
            EdgeTables::build(&[0, 1]),
            Err(KernelError::Shape { .. })
        ));
        assert_eq!(
            EdgeTables::build(&[0, 1, 2, 0, -3, 1]),
            Err(KernelError::NegativeIndex { face: 1, index: -3 })
        );
        assert!(matches!(
            components_from_slots(1, &[0, 1]),
            Err(KernelError::SlotCount { .. })
        ));
        assert!(matches!(
            components_from_slots(1, &[0, -1, 2]),
            Err(KernelError::NegativeSlotId { .. })
        ));
    }

    /// A deterministic pseudo-random soup with repeated indices, duplicate and degenerate faces.
    pub(crate) fn soup(m: usize, n: i64, seed: u64) -> Vec<i64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..3 * m)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % n as u64) as i64
            })
            .collect()
    }

    #[test]
    fn counting_and_sorting_paths_agree() {
        for (m, n, seed) in [(1, 3, 1), (7, 4, 2), (200, 40, 3), (5000, 900, 4)] {
            let f = soup(m, n, seed);
            let nn = *f.iter().max().unwrap() as usize + 1;
            let a = EdgeTables::build_counting(&f, m, nn);
            let b = EdgeTables::build_sorting(&f, m, nn);
            assert_eq!(a, b);
            // the definition: slots sorted by (min, max, slot)
            let mut keys: Vec<(u64, u64, u64)> = (0..3 * m)
                .map(|k| {
                    let (lo, hi) = slot_edge(&f, m, k);
                    (lo, hi, k as u64)
                })
                .collect();
            keys.sort();
            let order: Vec<i64> = keys.iter().map(|x| x.2 as i64).collect();
            assert_eq!(a.order, order);
        }
    }

    #[test]
    fn sparse_vertex_ids_take_the_sorting_path() {
        let big = 1_i64 << 40;
        let f = [0, big, 5, big, 0, 7];
        let t = EdgeTables::build(&f).unwrap();
        assert_eq!(t.n, big + 1);
        assert_eq!(t.uniq, [0, 5, 0, 7, 0, big, 5, big, 7, big]);
        assert_eq!(t.counts, [1, 1, 2, 1, 1]);
    }

    #[test]
    fn identical_for_any_thread_count() {
        let f = soup(20_000, 3_000, 9);
        let reference = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| EdgeTables::build(&f).unwrap());
        for threads in [2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(pool.install(|| EdgeTables::build(&f).unwrap()), reference);
        }
    }

    #[test]
    fn components_are_labelled_by_first_appearance() {
        // faces 0 and 2 share an edge, face 1 is alone, face 3 joins face 1
        let f = flat(&[[0, 1, 2], [10, 11, 12], [2, 1, 3], [11, 10, 13]]);
        let t = EdgeTables::build(&f).unwrap();
        assert_eq!(components_from_slots(4, &t.inv).unwrap(), [0, 1, 0, 1]);
        // arbitrary slot ids (cut edges get fresh ones in the part-id op)
        assert_eq!(
            components_from_slots(2, &[5, 9, 7, 5, 100, 3]).unwrap(),
            [0, 0]
        );
        assert_eq!(
            components_from_slots(2, &[5, 9, 7, 6, 100, 3]).unwrap(),
            [0, 1]
        );
    }
}
