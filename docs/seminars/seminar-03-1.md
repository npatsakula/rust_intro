---
sidebar_position: 5
title: "Seminar 3.1: nalgebra Fundamentals"
---

# Seminar 3.1: nalgebra Fundamentals

**Duration:** 90 minutes
**Prerequisites:** Lecture 3

---

## Objectives

- Implement numerical algorithms with nalgebra
- Solve linear systems and analyze decompositions
- Benchmark Rust vs C++ performance
- Estimate condition numbers

---

## Task 1: Gaussian Elimination (25 min)

### 1.1 Setup Project

```bash
cargo new linalg_seminar
cd linalg_seminar
```

```toml
[dependencies]
nalgebra = "0.33"
rand = "0.8"
```

### 1.2 Implement Gaussian Elimination

```rust
use nalgebra::{DMatrix, DVector};

/// Gaussian elimination with partial pivoting
pub fn gaussian_elimination(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return None;
    }

    // Create augmented matrix
    let mut aug = DMatrix::zeros(n, n + 1);
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = a[(i, j)];
        }
        aug[(i, n)] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[(col, col)].abs();

        for row in (col + 1)..n {
            let val = aug[(row, col)].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Check for singular matrix
        if max_val < 1e-14 {
            return None;
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let temp = aug[(col, j)];
                aug[(col, j)] = aug[(max_row, j)];
                aug[(max_row, j)] = temp;
            }
        }

        // Eliminate below pivot
        for row in (col + 1)..n {
            let factor = aug[(row, col)] / aug[(col, col)];
            for j in col..=n {
                aug[(row, j)] -= factor * aug[(col, j)];
            }
        }
    }

    // Back substitution
    let mut x = DVector::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[(i, n)];
        for j in (i + 1)..n {
            sum -= aug[(i, j)] * x[j];
        }
        x[i] = sum / aug[(i, i)];
    }

    Some(x)
}

fn main() {
    let a = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, -1.0,
        -3.0, -1.0, 2.0,
        -2.0, 1.0, 2.0,
    ]);

    let b = DVector::from_row_slice(&[8.0, -11.0, -3.0]);

    if let Some(x) = gaussian_elimination(&a, &b) {
        println!("Solution x: {}", x);

        // Verify
        let residual = &a * &x - &b;
        println!("Residual: {}", residual.norm());
    }
}
```

### 1.3 Compare with nalgebra's LU

```rust
fn compare_methods() {
    let n = 100;

    // Random matrix
    let a = DMatrix::from_fn(n, n, |_, _| rand::random::<f64>());
    let b = DVector::from_fn(n, |_, _| rand::random::<f64>());

    // Our implementation
    let start = std::time::Instant::now();
    let x_ours = gaussian_elimination(&a, &b).unwrap();
    let our_time = start.elapsed();

    // nalgebra LU
    let start = std::time::Instant::now();
    let x_nalgebra = a.lu().solve(&b).unwrap();
    let lu_time = start.elapsed();

    println!("Our implementation: {:?}", our_time);
    println!("nalgebra LU: {:?}", lu_time);

    // Check solutions match
    let diff = (&x_ours - &x_nalgebra).norm();
    println!("Solution difference: {:.2e}", diff);
}
```

---

## Task 2: Matrix Decompositions (20 min)

### 2.1 LU Decomposition

```rust
use nalgebra::DMatrix;

fn explore_lu() {
    let a = DMatrix::from_row_slice(3, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
    ]);

    let lu = a.clone().lu();

    println!("L:\n{:.4}", lu.l());
    println!("U:\n{:.4}", lu.u());

    // Verify: P * A = L * U
    let p = lu.p();
    let reconstructed = lu.l() * lu.u();
    let pa = p.inverse() * &a;

    println!("P⁻¹ * A:\n{:.4}", pa);
    println!("L * U:\n{:.4}", reconstructed);
    println!("Reconstruction error: {:.2e}", (&pa - &reconstructed).norm());
}
```

### 2.2 QR Decomposition

```rust
fn explore_qr() {
    let a = DMatrix::from_row_slice(4, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 13.0,
    ]);

    let qr = a.clone().qr();
    let q = qr.q();
    let r = qr.r();

    println!("Q:\n{:.4}", q);
    println!("R:\n{:.4}", r);

    // Q should be orthogonal
    let qt_q = q.transpose() * &q;
    println!("Q^T * Q (should be I):\n{:.4}", qt_q);

    // Verify reconstruction
    let reconstructed = &q * &r;
    println!("Reconstruction error: {:.2e}", (&a - &reconstructed).norm());
}
```

### 2.3 SVD

```rust
use nalgebra::SVD;

fn explore_svd() {
    let a = DMatrix::from_row_slice(4, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ]);

    let svd = SVD::new(a.clone(), true, true);

    println!("Singular values: {}", svd.singular_values);

    // Rank (non-zero singular values)
    let tolerance = 1e-10;
    let rank = svd.singular_values.iter()
        .filter(|&&s| s > tolerance)
        .count();
    println!("Numerical rank: {}", rank);

    // Verify reconstruction: A = U * Σ * V^T
    if let (Some(u), Some(v_t)) = (svd.u.as_ref(), svd.v_t.as_ref()) {
        let sigma = DMatrix::from_diagonal(&svd.singular_values);
        let reconstructed = u * sigma * v_t;
        println!("Reconstruction error: {:.2e}", (&a - &reconstructed).norm());
    }
}
```

---

## Task 3: SVD-Based Pseudoinverse (20 min)

### 3.1 Implement Pseudoinverse

```rust
/// Compute Moore-Penrose pseudoinverse using SVD
pub fn pseudoinverse(a: &DMatrix<f64>, tolerance: f64) -> DMatrix<f64> {
    let svd = SVD::new(a.clone(), true, true);

    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();

    // Compute Σ⁺ (pseudoinverse of diagonal matrix)
    let mut sigma_plus = DMatrix::zeros(a.ncols(), a.nrows());
    for (i, &s) in svd.singular_values.iter().enumerate() {
        if s > tolerance {
            sigma_plus[(i, i)] = 1.0 / s;
        }
    }

    // A⁺ = V * Σ⁺ * U^T
    v_t.transpose() * sigma_plus * u.transpose()
}

fn test_pseudoinverse() {
    // Overdetermined system (more equations than unknowns)
    let a = DMatrix::from_row_slice(4, 2, &[
        1.0, 1.0,
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0,
    ]);

    let b = DVector::from_row_slice(&[1.9, 3.1, 3.9, 5.2]);

    let a_pinv = pseudoinverse(&a, 1e-10);
    let x = &a_pinv * &b;

    println!("Least squares solution: {}", x);
    println!("Residual norm: {:.4}", (&a * &x - &b).norm());

    // Compare with nalgebra's SVD solve
    let svd = SVD::new(a.clone(), true, true);
    let x_nalgebra = svd.solve(&b, 1e-10).unwrap();
    println!("nalgebra solution: {}", x_nalgebra);
}
```

### 3.2 Linear Regression via Pseudoinverse

```rust
/// Fit polynomial using pseudoinverse
fn polynomial_fit(x: &[f64], y: &[f64], degree: usize) -> DVector<f64> {
    let n = x.len();

    // Build Vandermonde matrix
    let mut vandermonde = DMatrix::zeros(n, degree + 1);
    for i in 0..n {
        for j in 0..=degree {
            vandermonde[(i, j)] = x[i].powi(j as i32);
        }
    }

    let y_vec = DVector::from_row_slice(y);
    let v_pinv = pseudoinverse(&vandermonde, 1e-10);

    &v_pinv * &y_vec
}

fn test_polynomial_fit() {
    // Noisy quadratic data
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter()
        .map(|&xi| 1.0 + 2.0 * xi + 0.5 * xi * xi + 0.1 * rand::random::<f64>())
        .collect();

    let coeffs = polynomial_fit(&x, &y, 2);
    println!("Fitted coefficients: {}", coeffs);
    println!("Expected: [1, 2, 0.5]");
}
```

---

## Task 4: Condition Number (15 min)

### 4.1 Compute Condition Number

```rust
/// Estimate condition number using SVD
pub fn condition_number(a: &DMatrix<f64>) -> f64 {
    let svd = SVD::new(a.clone(), false, false);
    let singular_values = &svd.singular_values;

    let max_sv = singular_values.iter().cloned().fold(0.0, f64::max);
    let min_sv = singular_values.iter().cloned().fold(f64::INFINITY, f64::min);

    if min_sv < 1e-15 {
        f64::INFINITY
    } else {
        max_sv / min_sv
    }
}

fn analyze_condition_numbers() {
    // Well-conditioned matrix
    let good = DMatrix::from_row_slice(3, 3, &[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);

    // Ill-conditioned Hilbert matrix
    let hilbert = DMatrix::from_fn(5, 5, |i, j| {
        1.0 / ((i + j + 1) as f64)
    });

    println!("Identity condition number: {:.2e}", condition_number(&good));
    println!("Hilbert(5) condition number: {:.2e}", condition_number(&hilbert));
}
```

### 4.2 Effect of Condition Number on Solutions

```rust
fn condition_number_effect() {
    let n = 10;

    // Create Hilbert matrix (notoriously ill-conditioned)
    let hilbert = DMatrix::from_fn(n, n, |i, j| {
        1.0 / ((i + j + 1) as f64)
    });

    // Exact solution
    let x_exact = DVector::from_fn(n, |i, _| (i + 1) as f64);
    let b = &hilbert * &x_exact;

    // Solve
    let x_computed = hilbert.clone().lu().solve(&b).unwrap();

    // Error analysis
    let forward_error = (&x_exact - &x_computed).norm() / x_exact.norm();
    let residual = (&hilbert * &x_computed - &b).norm() / b.norm();
    let cond = condition_number(&hilbert);

    println!("Condition number: {:.2e}", cond);
    println!("Relative forward error: {:.2e}", forward_error);
    println!("Relative residual: {:.2e}", residual);
    println!("Error bound (κ × residual): {:.2e}", cond * residual);
}
```

---

## Task 5: Benchmark (10 min)

### 5.1 Rust vs Eigen (Conceptual)

```rust
use std::time::Instant;

fn benchmark_sizes() {
    println!("{:>6} {:>12} {:>12} {:>12}",
             "Size", "LU (ms)", "QR (ms)", "SVD (ms)");

    for &n in &[10, 50, 100, 200, 500] {
        let a = DMatrix::from_fn(n, n, |_, _| rand::random::<f64>());

        // LU
        let start = Instant::now();
        let _ = a.clone().lu();
        let lu_time = start.elapsed().as_secs_f64() * 1000.0;

        // QR
        let start = Instant::now();
        let _ = a.clone().qr();
        let qr_time = start.elapsed().as_secs_f64() * 1000.0;

        // SVD
        let start = Instant::now();
        let _ = SVD::new(a.clone(), true, true);
        let svd_time = start.elapsed().as_secs_f64() * 1000.0;

        println!("{:>6} {:>12.3} {:>12.3} {:>12.3}",
                 n, lu_time, qr_time, svd_time);
    }
}
```

---

## Homework

### Assignment: Power Iteration

Implement power iteration to find the largest eigenvalue and eigenvector.

```rust
/// Power iteration for dominant eigenvalue
pub fn power_iteration(
    a: &DMatrix<f64>,
    max_iter: usize,
    tolerance: f64,
) -> Option<(f64, DVector<f64>)> {
    // 1. Start with random vector
    // 2. Iterate: x_{k+1} = A * x_k / ||A * x_k||
    // 3. Eigenvalue estimate: λ ≈ x^T * A * x
    // 4. Stop when ||x_{k+1} - x_k|| < tolerance
    todo!()
}
```

**Bonus:** Implement inverse iteration for smallest eigenvalue.

**Submission:** Branch `homework-3-1`

---

## Summary

- ✅ Implemented Gaussian elimination
- ✅ Explored LU, QR, SVD decompositions
- ✅ Computed pseudoinverse for least squares
- ✅ Analyzed condition numbers
- ✅ Benchmarked performance

## Next

Continue to [Seminar 3.2: Tensor Operations for Mechanics](./seminar-03-2.md)
