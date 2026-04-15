//! 4-point planar homography via DLT (Direct Linear Transform).
//!
//! Solves for H such that `H * src_i ≈ dst_i` for four 2D correspondences,
//! with h_33 fixed to 1 to pin scale. That reduces to an 8×8 linear system
//! in the remaining eight parameters, which nalgebra's LU solves for us.
//!
//! All coordinates here are in normalized space (both source pad-plane and
//! destination image are in [0,1]²), so the matrix stays well-conditioned.

use nalgebra::{DMatrix, DVector};

pub type Mat3 = [[f32; 3]; 3];

/// Solve for the 3×3 homography mapping `src[i]` → `dst[i]` (i = 0..4).
///
/// Returns `None` if the points are degenerate (collinear / duplicate) and
/// the linear system has no unique solution.
pub fn compute_homography(src: [[f32; 2]; 4], dst: [[f32; 2]; 4]) -> Option<Mat3> {
    let mut a = DMatrix::<f64>::zeros(8, 8);
    let mut b = DVector::<f64>::zeros(8);
    for i in 0..4 {
        let (x, y) = (src[i][0] as f64, src[i][1] as f64);
        let (u, v) = (dst[i][0] as f64, dst[i][1] as f64);
        let r0 = 2 * i;
        let r1 = 2 * i + 1;
        a[(r0, 0)] = x;
        a[(r0, 1)] = y;
        a[(r0, 2)] = 1.0;
        a[(r0, 6)] = -u * x;
        a[(r0, 7)] = -u * y;
        b[r0] = u;

        a[(r1, 3)] = x;
        a[(r1, 4)] = y;
        a[(r1, 5)] = 1.0;
        a[(r1, 6)] = -v * x;
        a[(r1, 7)] = -v * y;
        b[r1] = v;
    }
    let sol = a.lu().solve(&b)?;
    Some([
        [sol[0] as f32, sol[1] as f32, sol[2] as f32],
        [sol[3] as f32, sol[4] as f32, sol[5] as f32],
        [sol[6] as f32, sol[7] as f32, 1.0],
    ])
}

/// Apply a 3×3 homography to a 2D point (with perspective divide).
pub fn apply(h: &Mat3, p: [f32; 2]) -> [f32; 2] {
    let (x, y) = (p[0], p[1]);
    let w = h[2][0] * x + h[2][1] * y + h[2][2];
    let u = (h[0][0] * x + h[0][1] * y + h[0][2]) / w;
    let v = (h[1][0] * x + h[1][1] * y + h[1][2]) / w;
    [u, v]
}

/// Inverse of a 3×3 matrix (for taking image coords back to pad-plane).
pub fn invert(h: &Mat3) -> Option<Mat3> {
    let m = nalgebra::Matrix3::new(
        h[0][0] as f64, h[0][1] as f64, h[0][2] as f64,
        h[1][0] as f64, h[1][1] as f64, h[1][2] as f64,
        h[2][0] as f64, h[2][1] as f64, h[2][2] as f64,
    );
    let inv = m.try_inverse()?;
    Some([
        [inv[(0, 0)] as f32, inv[(0, 1)] as f32, inv[(0, 2)] as f32],
        [inv[(1, 0)] as f32, inv[(1, 1)] as f32, inv[(1, 2)] as f32],
        [inv[(2, 0)] as f32, inv[(2, 1)] as f32, inv[(2, 2)] as f32],
    ])
}

/// The four unit-square corners in pad-plane order: TL, TR, BR, BL.
pub const UNIT_SQUARE: [[f32; 2]; 4] =
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
