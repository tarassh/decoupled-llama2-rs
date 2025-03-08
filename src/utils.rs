use rayon::prelude::*;

/// Root Mean Square (RMS) Normalization
/// Normalizes input x using RMS norm and scales with weights
pub fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize) {
    debug_assert_eq!(o.len(), size);
    debug_assert_eq!(x.len(), size);
    debug_assert_eq!(weight.len(), size);

    // Calculate sum of squares
    let ss = x.iter().map(|&xi| xi * xi).sum::<f32>() / size as f32;

    // Add epsilon and take inverse square root
    let scale = 1.0 / (ss + 1e-5f32).sqrt();

    // Normalize and scale
    for j in 0..size {
        o[j] = weight[j] * (scale * x[j]);
    }
}

/// Apply softmax normalization in-place
pub fn softmax(x: &mut [f32]) {
    // Find max value (for numerical stability)
    let max_val = x.iter().fold(x[0], |max, &val| max.max(val));

    // exp and sum
    let mut sum = 0.0f32;
    for xi in x.iter_mut() {
        *xi = (*xi - max_val).exp();
        sum += *xi;
    }

    // normalize
    for xi in x.iter_mut() {
        *xi /= sum;
    }
}

/// Matrix multiplication: W (d,n) @ x (n,) -> xout (d,)
/// Parallelized version using Rayon
#[cfg(feature = "parallel")]
pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    // Verify dimensions
    debug_assert_eq!(xout.len(), d);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(w.len(), d * n);

    // Parallel iteration over rows
    xout.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row_start = i * n;
        *val = w[row_start..row_start + n]
            .iter()
            .zip(x.iter())
            .map(|(&w_ij, &x_j)| w_ij * x_j)
            .sum();
    });
}

/// Matrix multiplication: W (d,n) @ x (n,) -> xout (d,)
/// Sequential version for when parallel feature is not enabled
#[cfg(not(feature = "parallel"))]
pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    // Verify dimensions
    debug_assert_eq!(xout.len(), d);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(w.len(), d * n);

    // Sequential iteration over rows
    for i in 0..d {
        let row_start = i * n;
        xout[i] = w[row_start..row_start + n]
            .iter()
            .zip(x.iter())
            .map(|(&w_ij, &x_j)| w_ij * x_j)
            .sum();
    }
}
