// SPDX-License-Identifier: Apache-2.0
//! Element kernel of the chunked unit conversion (port spec 6d).

use rayon::prelude::*;

/// A scalar that widens to float64 exactly as numpy's `astype(float64)` does.
pub trait Widen: Copy + Send + Sync {
    fn widen(self) -> f64;
}

impl Widen for f32 {
    #[inline]
    fn widen(self) -> f64 {
        f64::from(self)
    }
}

impl Widen for f64 {
    #[inline]
    fn widen(self) -> f64 {
        self
    }
}

/// `y = float32(float64(x) * factor)` for every element: one multiplication in float64 and
/// one round-to-nearest-even narrowing (no fused or float32 scaling).
pub fn scale_block<T: Widen>(x: &[T], factor: f64) -> Vec<f32> {
    x.par_iter().map(|&v| (v.widen() * factor) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|v| v.to_bits()).collect()
    }

    // -- port spec 6d, test vectors 1-2 (3 is the integer / factor 1 rule, kept in Python) ----

    #[test]
    fn vector_1() {
        let y = scale_block(&[1.0_f32, -2.5, 3e-7, 1e30], 1.225);
        assert_eq!(
            bits(&y),
            [1_067_240_653, 3_225_681_920, 885_345_489, 1_903_649_540]
        );
    }

    #[test]
    fn vector_2() {
        assert_eq!(bits(&scale_block(&[0.1_f32], 3.0)), [1_050_253_722]);
    }

    #[test]
    fn float32_scaling_would_differ() {
        let x = 1e30_f32;
        let fused = x * 1.225_f32;
        assert_ne!(scale_block(&[x], 1.225)[0].to_bits(), fused.to_bits());
    }

    #[test]
    fn float64_input_and_special_values() {
        let y = scale_block(&[1.0e300_f64, -0.0, f64::NAN, f64::INFINITY], 2.0);
        assert!(y[0].is_infinite());
        assert!(y[1] == 0.0 && y[1].is_sign_negative());
        assert!(y[2].is_nan());
        assert!(y[3].is_infinite());
        assert!(scale_block::<f32>(&[], 2.0).is_empty());
    }

    #[test]
    fn identical_for_any_thread_count() {
        let x: Vec<f32> = (0..100_000).map(|i| i as f32 * 1.3e-3 - 40.0).collect();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| bits(&scale_block(&x, 1.225)))
        };
        let reference = run(1);
        assert_eq!(run(4), reference);
    }
}
