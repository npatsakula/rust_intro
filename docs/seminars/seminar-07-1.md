---
sidebar_position: 13
title: "Seminar 7.1: Property-Based Testing"
---

# Seminar 7.1: Property-Based Testing and Fuzzing

**Duration:** 90 minutes
**Prerequisites:** Lecture 7

---

## Objectives

- Write property tests for numerical algorithms
- Set up fuzzing for input handling
- Test conservation laws in simulations
- Find bugs in intentionally flawed code

---

## Task 1: Property Tests for Numerical Algorithms (25 min)

```toml
[dev-dependencies]
proptest = "1.5"
approx = "0.5"
```

### 1.1 Basic Properties

```rust
use proptest::prelude::*;
use approx::assert_relative_eq;

proptest! {
    /// Dot product is commutative: a·b = b·a
    #[test]
    fn dot_product_commutative(
        a in prop::collection::vec(-1000.0..1000.0f64, 1..100),
    ) {
        let b: Vec<f64> = a.iter().map(|x| x * 2.0 + 1.0).collect();

        let ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let ba: f64 = b.iter().zip(a.iter()).map(|(x, y)| x * y).sum();

        assert_relative_eq!(ab, ba, epsilon = 1e-10);
    }

    /// Norm is non-negative: ||v|| >= 0
    #[test]
    fn norm_non_negative(
        v in prop::collection::vec(-1000.0..1000.0f64, 1..100),
    ) {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        prop_assert!(norm >= 0.0);
    }

    /// Triangle inequality: ||a + b|| <= ||a|| + ||b||
    #[test]
    fn triangle_inequality(
        a in prop::collection::vec(-100.0..100.0f64, 1..50),
        b in prop::collection::vec(-100.0..100.0f64, 1..50),
    ) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        let sum: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let norm_sum: f64 = sum.iter().map(|x| x * x).sum::<f64>().sqrt();

        prop_assert!(norm_sum <= norm_a + norm_b + 1e-10);
    }
}
```

### 1.2 Matrix Properties

```rust
use nalgebra::{DMatrix, DVector};

proptest! {
    /// Matrix multiplication is associative: (AB)C = A(BC)
    #[test]
    fn matrix_mult_associative(
        m in 2..5usize,
        n in 2..5usize,
        p in 2..5usize,
        q in 2..5usize,
    ) {
        let a = DMatrix::from_fn(m, n, |i, j| (i * n + j) as f64);
        let b = DMatrix::from_fn(n, p, |i, j| (i * p + j) as f64 * 0.1);
        let c = DMatrix::from_fn(p, q, |i, j| (i * q + j) as f64 * 0.01);

        let ab_c = (&a * &b) * &c;
        let a_bc = &a * (&b * &c);

        let error = (&ab_c - &a_bc).norm();
        prop_assert!(error < 1e-8, "Associativity error: {}", error);
    }

    /// Matrix transpose: (AB)^T = B^T A^T
    #[test]
    fn transpose_product(
        m in 2..10usize,
        n in 2..10usize,
        p in 2..10usize,
    ) {
        let a = DMatrix::from_fn(m, n, |i, j| (i + j) as f64);
        let b = DMatrix::from_fn(n, p, |i, j| (i * j) as f64);

        let ab_t = (&a * &b).transpose();
        let bt_at = b.transpose() * a.transpose();

        let error = (&ab_t - &bt_at).norm();
        prop_assert!(error < 1e-10, "Transpose product error: {}", error);
    }

    /// Orthogonal matrix preserves norm: ||Qv|| = ||v||
    #[test]
    fn orthogonal_preserves_norm(n in 2..10usize) {
        // Create orthogonal matrix via QR
        let rand_matrix = DMatrix::from_fn(n, n, |i, j| {
            ((i * 7 + j * 13) % 100) as f64 / 100.0
        });
        let qr = rand_matrix.qr();
        let q = qr.q();

        let v = DVector::from_fn(n, |i, _| (i + 1) as f64);
        let qv = &q * &v;

        let norm_v = v.norm();
        let norm_qv = qv.norm();

        assert_relative_eq!(norm_v, norm_qv, epsilon = 1e-10);
    }
}
```

---

## Task 2: Testing Solvers (20 min)

```rust
proptest! {
    /// Linear solver produces correct solution
    #[test]
    fn linear_solver_correct(
        n in 3..8usize,
        seed in 0..1000u64,
    ) {
        use rand::{SeedableRng, Rng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Generate well-conditioned matrix
        let mut a = DMatrix::from_fn(n, n, |_, _| rng.gen_range(-1.0..1.0));

        // Make diagonally dominant for stability
        for i in 0..n {
            let row_sum: f64 = a.row(i).iter().map(|x| x.abs()).sum();
            a[(i, i)] = row_sum + 1.0;
        }

        let b = DVector::from_fn(n, |_, _| rng.gen_range(-10.0..10.0));

        // Solve
        let x = a.clone().lu().solve(&b).unwrap();

        // Verify
        let residual = &a * &x - &b;
        let rel_error = residual.norm() / b.norm();

        prop_assert!(rel_error < 1e-10, "Relative error: {}", rel_error);
    }

    /// Eigenvalues of symmetric matrix are real
    #[test]
    fn symmetric_eigenvalues_real(n in 2..6usize) {
        // Create symmetric matrix
        let mut m = DMatrix::from_fn(n, n, |i, j| {
            ((i + j) * (i + j + 1)) as f64 / 100.0
        });

        // Make symmetric
        for i in 0..n {
            for j in i+1..n {
                m[(i, j)] = m[(j, i)];
            }
        }

        let eigen = m.symmetric_eigen();

        // All eigenvalues should be real (stored directly, not as complex)
        for &ev in eigen.eigenvalues.iter() {
            prop_assert!(ev.is_finite(), "Non-finite eigenvalue: {}", ev);
        }

        // Verify reconstruction: A = V * D * V^T
        let d = DMatrix::from_diagonal(&eigen.eigenvalues);
        let reconstructed = &eigen.eigenvectors * &d * eigen.eigenvectors.transpose();

        let error = (&m - &reconstructed).norm();
        prop_assert!(error < 1e-10, "Reconstruction error: {}", error);
    }
}
```

---

## Task 3: Conservation Laws (20 min)

```rust
/// Test energy conservation in simple harmonic oscillator
#[test]
fn test_energy_conservation() {
    // x'' = -k*x, energy E = 0.5*v² + 0.5*k*x²
    let k = 1.0;

    let mut x = 1.0;  // Initial position
    let mut v = 0.0;  // Initial velocity
    let dt = 0.001;

    let initial_energy = 0.5 * v * v + 0.5 * k * x * x;

    // Störmer-Verlet integration
    for _ in 0..10000 {
        // Half step velocity
        v -= 0.5 * dt * k * x;
        // Full step position
        x += dt * v;
        // Half step velocity
        v -= 0.5 * dt * k * x;
    }

    let final_energy = 0.5 * v * v + 0.5 * k * x * x;
    let energy_drift = (final_energy - initial_energy).abs() / initial_energy;

    assert!(energy_drift < 1e-10, "Energy drift: {}", energy_drift);
}

proptest! {
    /// Mass conservation in diffusion simulation
    #[test]
    fn mass_conservation(
        n in 10..50usize,
        steps in 10..100usize,
    ) {
        let mut concentration = vec![0.0; n];

        // Initial condition: delta function at center
        concentration[n / 2] = 1.0;

        let initial_mass: f64 = concentration.iter().sum();

        // Diffusion with periodic boundary
        let d = 0.1;
        let dt = 0.001;
        let dx = 1.0 / n as f64;

        let mut new_conc = vec![0.0; n];

        for _ in 0..steps {
            for i in 0..n {
                let left = concentration[(i + n - 1) % n];
                let right = concentration[(i + 1) % n];
                let center = concentration[i];

                new_conc[i] = center + d * dt / (dx * dx) * (left - 2.0 * center + right);
            }
            std::mem::swap(&mut concentration, &mut new_conc);
        }

        let final_mass: f64 = concentration.iter().sum();
        let mass_error = (final_mass - initial_mass).abs();

        prop_assert!(mass_error < 1e-10, "Mass error: {}", mass_error);
    }
}
```

---

## Task 4: Finding Bugs (15 min)

```rust
/// Intentionally buggy implementations - find the bugs!
mod buggy {
    /// Bug 1: Off-by-one error
    pub fn sum_first_n(data: &[f64], n: usize) -> f64 {
        let mut sum = 0.0;
        for i in 0..=n {  // Bug: should be 0..n
            if i < data.len() {
                sum += data[i];
            }
        }
        sum
    }

    /// Bug 2: Integer overflow potential
    pub fn factorial(n: u32) -> u64 {
        let mut result = 1u64;
        for i in 1..=n {
            result *= i as u64;  // Bug: no overflow check
        }
        result
    }

    /// Bug 3: Numerical instability
    pub fn quadratic_roots(a: f64, b: f64, c: f64) -> Option<(f64, f64)> {
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_d = discriminant.sqrt();
        // Bug: catastrophic cancellation when b >> sqrt_d
        let x1 = (-b + sqrt_d) / (2.0 * a);
        let x2 = (-b - sqrt_d) / (2.0 * a);

        Some((x1, x2))
    }
}

#[cfg(test)]
mod bug_tests {
    use super::buggy::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_sum_first_n(
            data in prop::collection::vec(0.0..100.0f64, 1..100),
            n in 0..50usize,
        ) {
            let n = n.min(data.len());
            let result = sum_first_n(&data, n);
            let expected: f64 = data[..n].iter().sum();

            // This will fail when n < data.len()
            prop_assert_eq!(result, expected);
        }

        #[test]
        fn test_factorial(n in 0..25u32) {
            let result = factorial(n);

            // Verify factorial properties
            if n > 0 {
                prop_assert_eq!(result, n as u64 * factorial(n - 1));
            } else {
                prop_assert_eq!(result, 1);
            }
        }

        #[test]
        fn test_quadratic_roots(
            a in 0.1..10.0f64,
            b in -100.0..100.0f64,
            c in -100.0..100.0f64,
        ) {
            if let Some((x1, x2)) = quadratic_roots(a, b, c) {
                // Verify roots satisfy equation
                let f1 = a * x1 * x1 + b * x1 + c;
                let f2 = a * x2 * x2 + b * x2 + c;

                prop_assert!(f1.abs() < 1e-6, "f(x1) = {}", f1);
                prop_assert!(f2.abs() < 1e-6, "f(x2) = {}", f2);
            }
        }
    }
}
```

---

## Task 5: Setup Fuzzing (10 min)

```bash
cargo install cargo-fuzz
cargo fuzz init
```

```rust
// fuzz/fuzz_targets/matrix_ops.rs
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let rows = (data[0] % 10 + 1) as usize;
    let cols = (data[1] % 10 + 1) as usize;

    if data.len() < 2 + rows * cols * 8 {
        return;
    }

    // Parse floats from remaining bytes
    let floats: Vec<f64> = data[2..]
        .chunks(8)
        .take(rows * cols)
        .filter_map(|chunk| {
            if chunk.len() == 8 {
                Some(f64::from_le_bytes(chunk.try_into().unwrap()))
            } else {
                None
            }
        })
        .filter(|f| f.is_finite())
        .collect();

    if floats.len() == rows * cols {
        // Test matrix operations
        let matrix = nalgebra::DMatrix::from_row_slice(rows, cols, &floats);

        let _ = matrix.transpose();
        let _ = matrix.norm();

        if rows == cols {
            let _ = matrix.clone().lu();
            let _ = matrix.clone().qr();
        }
    }
});
```

---

## Homework

### Assignment: Property Tests for FEM

Write property-based tests for FEM assembly:

1. **Stiffness matrix symmetry**: K = K^T
2. **Positive semi-definiteness**: x^T K x >= 0
3. **Rigid body modes**: K * rigid_mode ≈ 0
4. **Patch test**: Uniform strain produces uniform stress

**Submission:** Branch `homework-7-1`

---

## Summary

- ✅ Wrote property tests for numerical algorithms
- ✅ Tested matrix operations and solvers
- ✅ Verified conservation laws
- ✅ Found bugs in flawed code
- ✅ Set up fuzzing

## Next

Continue to [Seminar 7.2: Verification and Final Project](./seminar-07-2.md)
