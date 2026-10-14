// SPDX-License-Identifier: Apache-2.0
//! numpy's pairwise summation, the one float accumulation order the audit must reproduce.

use crate::real::Real;

const BLOCK: usize = 128;

/// `pairwise_sum` of `numpy/_core/src/umath/loops_utils.h.src`: a plain loop below 8
/// elements, eight running partial sums up to 128 elements, recursion on halves (the left
/// half rounded down to a multiple of 8) above.
pub fn pairwise<T: Real>(a: &[T]) -> T {
    let n = a.len();
    if n < 8 {
        let mut res = T::ZERO;
        for &x in a {
            res = res + x;
        }
        res
    } else if n <= BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let body = n - n % 8;
        let mut i = 8;
        while i < body {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj = *rj + a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[body..] {
            res = res + x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// `np.sum(a)` of a contiguous 1-D float array: the reduction starts from the identity
/// `0.0` and adds the pairwise sum (so an all `-0.0` array sums to `+0.0`, as in numpy).
pub fn numpy_sum<T: Real>(a: &[T]) -> T {
    T::ZERO + pairwise(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_sizes_are_sequential() {
        let a = [1.0e16_f64, 1.0, -1.0e16, 1.0];
        // ((0 + 1e16) + 1) + -1e16) + 1: the first 1.0 is absorbed
        assert_eq!(pairwise(&a), 1.0);
        assert_eq!(numpy_sum::<f64>(&[]), 0.0);
    }

    #[test]
    fn eight_accumulators_between_8_and_128() {
        // 16 elements: r[j] = a[j] + a[8 + j]; the pairing differs from a sequential sum
        let mut a = [0.0_f64; 16];
        a[0] = 1.0e16;
        a[1] = 1.0;
        a[8] = -1.0e16;
        a[9] = 1.0;
        assert_eq!(pairwise(&a), 2.0);
        let seq: f64 = a.iter().sum();
        assert_eq!(seq, 1.0);
    }

    #[test]
    fn recursion_splits_on_a_multiple_of_eight() {
        // 130 elements: n2 = 65 - 65 % 8 = 64, halves of 64 and 66
        let a: Vec<f64> = (0..130).map(|i| 0.1 * f64::from(i)).collect();
        let expect = pairwise(&a[..64]) + pairwise(&a[64..]);
        assert_eq!(pairwise(&a).to_bits(), expect.to_bits());
    }

    #[test]
    fn negative_zero_sums_to_positive_zero() {
        let a = [-0.0_f64; 9];
        assert!(pairwise(&a).is_sign_negative());
        assert!(numpy_sum(&a).is_sign_positive());
    }

    #[test]
    fn f32_matches_numpy_value() {
        // np.sum(np.arange(1000, dtype=np.float32) * np.float32(0.1)) == 49950.0 (float32)
        let a: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1_f32).collect();
        assert_eq!(numpy_sum(&a), 49950.0_f32);
    }
}
