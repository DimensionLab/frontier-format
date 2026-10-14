// SPDX-License-Identifier: Apache-2.0
//! The two coordinate scalar types the reference kernels see (`float32` from the dataset
//! arrays, `float64` from callers that built points in memory).

use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Sub};

/// IEEE-754 binary32 or binary64 with the handful of operations the kernels need. Every
/// operation is the plain IEEE one (no fused multiply-add, no reassociation).
pub trait Real:
    Copy
    + Send
    + Sync
    + PartialOrd
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + 'static
{
    const ZERO: Self;
    fn sqrt(self) -> Self;
    fn to_f64(self) -> f64;
    fn to_f32(self) -> f32;
    fn from_f64(x: f64) -> Self;
    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
    fn round_ties_even(self) -> Self;
}

macro_rules! impl_real {
    ($t:ty) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn to_f32(self) -> f32 {
                self as f32
            }
            #[inline]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline]
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
            #[inline]
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
            #[inline]
            fn round_ties_even(self) -> Self {
                <$t>::round_ties_even(self)
            }
        }
    };
}

impl_real!(f32);
impl_real!(f64);

/// numpy's sort order for floats: numeric, `-0.0 == 0.0`, every NaN after every number and
/// all NaNs equal to each other.
#[inline]
pub fn numpy_cmp<T: Real>(a: T, b: T) -> Ordering {
    match a.partial_cmp(&b) {
        Some(o) => o,
        None => match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            _ => Ordering::Less,
        },
    }
}
