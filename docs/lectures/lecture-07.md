---
sidebar_position: 7
title: "Lecture 7: Verification and Production"
---

# Lecture 7: Verification, Testing, and Production Code

**Duration:** 90 minutes
**Block:** IV — Advanced Rust

---

## Learning Objectives

By the end of this lecture, you will:
- Apply property-based testing with proptest
- Use fuzzing to find edge cases
- Understand formal verification tools
- Set up CI/CD for scientific code

---

## 1. Why Verification Matters for Scientific Code

### The Stakes

- **Numerical errors** can cascade through simulations
- **Edge cases** may not be obvious (singular matrices, overflow)
- **Reproducibility** is essential for science
- **Silent failures** are worse than crashes

### Testing Pyramid for Scientific Software

```
     ╱╲
    ╱  ╲         Formal Verification
   ╱────╲        (kani, prusti)
  ╱      ╲
 ╱ Property ╲    Property-Based Testing
╱   Tests    ╲   (proptest, fuzzing)
╱──────────────╲
╱  Integration  ╲ Reference Solutions
╱    Tests       ╲
╱──────────────────╲
╱     Unit Tests    ╲ Known Values
╱____________________╲
```

---

## 2. Property-Based Testing with proptest

### Installation

```toml
[dev-dependencies]
proptest = "1.5"
```

### Basic Properties

```rust
use proptest::prelude::*;

proptest! {
    /// Matrix transpose is involutory: (A^T)^T = A
    #[test]
    fn transpose_involution(
        rows in 1..10usize,
        cols in 1..10usize,
    ) {
        let data: Vec<f64> = (0..rows*cols)
            .map(|i| i as f64)
            .collect();

        let matrix = Matrix::from_data(rows, cols, data).unwrap();
        let transposed = matrix.transpose();
        let double_transposed = transposed.transpose();

        // Element-wise comparison
        for i in 0..rows {
            for j in 0..cols {
                prop_assert_eq!(
                    matrix.get(i, j).unwrap(),
                    double_transposed.get(i, j).unwrap()
                );
            }
        }
    }

    /// Matrix inverse identity: A * A^-1 = I
    #[test]
    fn inverse_identity(
        a in prop::array::uniform3(prop::array::uniform3(-100.0f64..100.0))
    ) {
        let mat = Matrix3::from_row_slice(&a.concat());

        if let Some(inv) = mat.try_inverse() {
            let product = mat * inv;
            let identity = Matrix3::identity();

            for i in 0..3 {
                for j in 0..3 {
                    let diff = (product[(i, j)] - identity[(i, j)]).abs();
                    prop_assert!(diff < 1e-10, "diff = {}", diff);
                }
            }
        }
        // If not invertible, that's fine - skip
    }
}
```

### Custom Strategies

```rust
use proptest::prelude::*;

/// Strategy for generating positive definite matrices
fn positive_definite_matrix(size: usize) -> impl Strategy<Value = DMatrix<f64>> {
    // Generate A, then return A * A^T (always positive semi-definite)
    // Add small diagonal for positive definite
    prop::collection::vec(-10.0..10.0f64, size * size)
        .prop_map(move |data| {
            let a = DMatrix::from_row_slice(size, size, &data);
            let aat = &a * a.transpose();

            // Add small positive diagonal
            let mut result = aat;
            for i in 0..size {
                result[(i, i)] += 1.0;
            }
            result
        })
}

proptest! {
    /// Cholesky decomposition works for positive definite matrices
    #[test]
    fn cholesky_works(matrix in positive_definite_matrix(5)) {
        let chol = Cholesky::new(matrix.clone());
        prop_assert!(chol.is_some(), "Cholesky should succeed for PD matrix");

        if let Some(chol) = chol {
            let l = chol.l();
            let reconstructed = &l * l.transpose();

            let error = (&matrix - &reconstructed).norm();
            prop_assert!(error < 1e-10, "Reconstruction error: {}", error);
        }
    }
}

/// Strategy for well-conditioned matrices
fn well_conditioned_matrix(n: usize) -> impl Strategy<Value = DMatrix<f64>> {
    prop::collection::vec(-1.0..1.0f64, n * n)
        .prop_filter_map("singular matrix", move |data| {
            let mat = DMatrix::from_row_slice(n, n, &data);
            if mat.determinant().abs() > 0.1 {
                Some(mat)
            } else {
                None
            }
        })
}
```

### Testing Numerical Algorithms

```rust
proptest! {
    /// LU decomposition satisfies P*A = L*U
    #[test]
    fn lu_decomposition_correct(matrix in well_conditioned_matrix(5)) {
        let lu = matrix.clone().lu();

        let p = lu.p();
        let l = lu.l();
        let u = lu.u();

        let pa = p.inverse() * &matrix;
        let lu_product = &l * &u;

        let error = (&pa - &lu_product).norm();
        prop_assert!(error < 1e-10, "LU error: {}", error);
    }

    /// Linear solver satisfies A*x = b
    #[test]
    fn linear_solver_correct(
        matrix in well_conditioned_matrix(5),
        b in prop::collection::vec(-10.0..10.0f64, 5),
    ) {
        let b_vec = DVector::from_row_slice(&b);

        if let Some(x) = matrix.clone().lu().solve(&b_vec) {
            let residual = &matrix * &x - &b_vec;
            let rel_error = residual.norm() / b_vec.norm();

            prop_assert!(rel_error < 1e-10, "Relative error: {}", rel_error);
        }
    }
}
```

---

## 3. Fuzzing

### Setup cargo-fuzz

```bash
cargo install cargo-fuzz
cargo fuzz init
```

### Fuzz Target Example

```rust
// fuzz/fuzz_targets/matrix_parse.rs
#![no_main]

use libfuzzer_sys::fuzz_target;
use my_library::Matrix;

fuzz_target!(|data: &[u8]| {
    // Try to parse matrix from bytes
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = Matrix::from_csv_string(s);
    }
});
```

```rust
// fuzz/fuzz_targets/matrix_ops.rs
#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct MatrixInput {
    rows: u8,
    cols: u8,
    data: Vec<f64>,
}

fuzz_target!(|input: MatrixInput| {
    let rows = (input.rows % 10 + 1) as usize;
    let cols = (input.cols % 10 + 1) as usize;

    if input.data.len() >= rows * cols {
        let data: Vec<f64> = input.data.iter()
            .take(rows * cols)
            .cloned()
            .collect();

        if let Ok(matrix) = Matrix::from_data(rows, cols, data) {
            // These should never panic
            let _ = matrix.transpose();
            let _ = matrix.norm();

            if rows == cols {
                let _ = matrix.determinant();
                let _ = matrix.try_inverse();
            }
        }
    }
});
```

### Running Fuzzer

```bash
cargo +nightly fuzz run matrix_ops -- -max_total_time=300
```

---

## 4. Formal Verification with Kani

### Installation

```bash
cargo install --locked kani-verifier
kani setup
```

### Basic Verification

```rust
#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_bounds_check() {
        let size: usize = kani::any();
        kani::assume(size > 0 && size <= 100);

        let vec: Vec<i32> = vec![0; size];
        let index: usize = kani::any();
        kani::assume(index < size);

        // This should never panic
        let _ = vec[index];
    }

    #[kani::proof]
    #[kani::unwind(11)]  // Bound loop iterations
    fn verify_matrix_get() {
        let rows: usize = kani::any();
        let cols: usize = kani::any();
        kani::assume(rows > 0 && rows <= 10);
        kani::assume(cols > 0 && cols <= 10);

        let matrix = Matrix::zeros(rows, cols).unwrap();

        let i: usize = kani::any();
        let j: usize = kani::any();

        if i < rows && j < cols {
            // Should succeed
            assert!(matrix.get(i, j).is_ok());
        } else {
            // Should fail
            assert!(matrix.get(i, j).is_err());
        }
    }
}
```

### Running Kani

```bash
cargo kani --tests
```

---

## 5. SMT Solvers with z3

```toml
[dependencies]
z3 = "0.12"
```

```rust
use z3::*;

/// Verify that our linear solver produces correct results
fn verify_solver() {
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let solver = Solver::new(&ctx);

    // Symbolic variables for 2x2 system
    let a11 = Real::new_const(&ctx, "a11");
    let a12 = Real::new_const(&ctx, "a12");
    let a21 = Real::new_const(&ctx, "a21");
    let a22 = Real::new_const(&ctx, "a22");

    let b1 = Real::new_const(&ctx, "b1");
    let b2 = Real::new_const(&ctx, "b2");

    let x1 = Real::new_const(&ctx, "x1");
    let x2 = Real::new_const(&ctx, "x2");

    // Constraint: det(A) != 0
    let det = &a11 * &a22 - &a12 * &a21;
    solver.assert(&det._eq(&Real::from_real(&ctx, 0, 1)).not());

    // Cramer's rule solution
    let x1_cramer = (&b1 * &a22 - &b2 * &a12) / &det;
    let x2_cramer = (&a11 * &b2 - &a21 * &b1) / &det;

    // Verify: A*x = b
    let eq1 = &a11 * &x1_cramer + &a12 * &x2_cramer;
    let eq2 = &a21 * &x1_cramer + &a22 * &x2_cramer;

    // These should always equal b1 and b2
    solver.assert(&eq1._eq(&b1).not().or(&[&eq2._eq(&b2).not()]));

    match solver.check() {
        SatResult::Unsat => println!("Verified: Cramer's rule is correct"),
        SatResult::Sat => println!("Found counterexample!"),
        SatResult::Unknown => println!("Could not verify"),
    }
}
```

---

## 6. CI/CD for Scientific Code

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run tests
        run: cargo test --all-features

      - name: Run clippy
        run: cargo clippy -- -D warnings

      - name: Check formatting
        run: cargo fmt -- --check

  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Property tests
        run: cargo test --release -- --ignored property
        env:
          PROPTEST_CASES: 10000

  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run benchmarks
        run: cargo bench --no-run

  miri:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
        with:
          components: miri

      - name: Run Miri
        run: cargo +nightly miri test
```

---

## Summary

| Tool | Purpose | Confidence Level |
|------|---------|------------------|
| Unit tests | Known values | Basic |
| Property tests | Random inputs | High |
| Fuzzing | Edge cases | Very high |
| Kani | Bounded verification | Mathematical |
| Z3 | Symbolic reasoning | Mathematical |

---

## Next Steps

- **Seminar 7.1**: [Property-Based Testing and Fuzzing](../seminars/seminar-07-1.md)
- **Seminar 7.2**: [Verification and Final Project](../seminars/seminar-07-2.md)
