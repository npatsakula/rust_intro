---
sidebar_position: 3
title: "Lecture 3: Linear Algebra and Tensor Libraries"
---

# Lecture 3: Linear Algebra and Tensor Libraries

**Duration:** 90 minutes
**Block:** II — Mathematical Tooling

---

## Learning Objectives

By the end of this lecture, you will:
- Navigate Rust's numerical computing ecosystem
- Use `nalgebra` for linear algebra operations
- Work with `ndarray` for N-dimensional arrays
- Apply `russell_tensor` for continuum mechanics

---

## 1. Rust Numerical Ecosystem Overview

### Maturity Comparison

| Feature | Python (NumPy/SciPy) | C++ (Eigen) | Rust |
|---------|---------------------|-------------|------|
| Maturity | Very mature | Very mature | Growing |
| Performance | Good (via C) | Excellent | Excellent |
| Safety | Runtime checks | Manual | Compile-time |
| Ecosystem | Huge | Medium | Growing |

### Key Libraries

| Library | Purpose | Python Equivalent |
|---------|---------|-------------------|
| `nalgebra` | Linear algebra | NumPy + parts of SciPy |
| `ndarray` | N-dimensional arrays | NumPy ndarray |
| `faer` | High-perf linear algebra | NumPy + LAPACK |
| `russell` | Scientific computing | SciPy |
| `polars` | DataFrames | Pandas |

---

## 2. nalgebra — Core Linear Algebra

### Installation

```toml
[dependencies]
nalgebra = "0.33"
```

### Static vs Dynamic Matrices

```rust
use nalgebra::{Matrix3, Matrix4, DMatrix, Vector3, DVector};
use nalgebra::{SMatrix, SVector};  // Type aliases

fn main() {
    // Static size — known at compile time
    let m: Matrix3<f64> = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    );

    // Generic static size
    let m: SMatrix<f64, 4, 3> = SMatrix::zeros();

    // Dynamic size — determined at runtime
    let m: DMatrix<f64> = DMatrix::zeros(100, 100);

    // Vectors
    let v: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
    let v: DVector<f64> = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
}
```

### Compile-Time Dimension Checking

```rust
use nalgebra::{Matrix2x3, Matrix3x2, Matrix2, Matrix3};

fn main() {
    let a: Matrix2x3<f64> = Matrix2x3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    );
    let b: Matrix3x2<f64> = Matrix3x2::new(
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    );

    // Valid: (2×3) × (3×2) = (2×2)
    let c: Matrix2<f64> = a * b;

    // Compile error! (2×3) × (2×3) dimension mismatch
    // let d = a * a;
}
```

### Matrix Operations

```rust
use nalgebra::{Matrix3, Vector3};

fn main() {
    let a = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,  // Changed to make invertible
    );

    // Basic operations
    let b = a.transpose();
    let c = a * 2.0;         // Scalar multiplication
    let d = a + b;           // Addition
    let e = a * b;           // Matrix multiplication

    // Properties
    println!("Trace: {}", a.trace());
    println!("Determinant: {}", a.determinant());
    println!("Norm: {}", a.norm());

    // Inverse
    if let Some(inv) = a.try_inverse() {
        println!("Inverse exists");
        let identity = a * inv;
        println!("A * A^-1 ≈ I: {}", identity);
    }

    // Solve linear system Ax = b
    let b = Vector3::new(1.0, 2.0, 3.0);
    if let Some(x) = a.lu().solve(&b) {
        println!("Solution x: {}", x);
    }
}
```

### Decompositions

```rust
use nalgebra::{Matrix3, SymmetricEigen, SVD, Cholesky, LU, QR};

fn main() {
    let a = Matrix3::new(
        4.0, 2.0, 2.0,
        2.0, 5.0, 1.0,
        2.0, 1.0, 6.0,
    );

    // LU decomposition
    let lu = a.lu();
    println!("LU determinant: {}", lu.determinant());

    // QR decomposition
    let qr = a.qr();
    let (q, r) = (qr.q(), qr.r());

    // Symmetric eigendecomposition (for symmetric matrices)
    let eigen = SymmetricEigen::new(a);
    println!("Eigenvalues: {}", eigen.eigenvalues);
    println!("Eigenvectors:\n{}", eigen.eigenvectors);

    // SVD
    let svd = SVD::new(a, true, true);
    println!("Singular values: {}", svd.singular_values);

    // Cholesky (for positive definite matrices)
    if let Some(chol) = Cholesky::new(a) {
        let l = chol.l();
        println!("Cholesky L:\n{}", l);
    }
}
```

### Application: Principal Stress Analysis

```rust
use nalgebra::{Matrix3, SymmetricEigen};

/// Compute principal stresses from stress tensor
fn principal_stresses(stress: &Matrix3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
    let eigen = SymmetricEigen::new(*stress);

    // Sort eigenvalues (principal stresses) in descending order
    let mut indices: Vec<usize> = (0..3).collect();
    indices.sort_by(|&i, &j| {
        eigen.eigenvalues[j].partial_cmp(&eigen.eigenvalues[i]).unwrap()
    });

    let sorted_values = Vector3::new(
        eigen.eigenvalues[indices[0]],
        eigen.eigenvalues[indices[1]],
        eigen.eigenvalues[indices[2]],
    );

    // Principal directions
    let sorted_vectors = Matrix3::from_columns(&[
        eigen.eigenvectors.column(indices[0]).into(),
        eigen.eigenvectors.column(indices[1]).into(),
        eigen.eigenvectors.column(indices[2]).into(),
    ]);

    (sorted_values, sorted_vectors)
}

fn main() {
    // Stress tensor (symmetric)
    let stress = Matrix3::new(
        100.0,  20.0,   0.0,
         20.0,  50.0,   0.0,
          0.0,   0.0,  30.0,
    );

    let (principals, directions) = principal_stresses(&stress);
    println!("Principal stresses: σ₁={:.2}, σ₂={:.2}, σ₃={:.2}",
             principals[0], principals[1], principals[2]);
    println!("Principal directions:\n{}", directions);

    // Von Mises stress
    let s1 = principals[0];
    let s2 = principals[1];
    let s3 = principals[2];
    let von_mises = ((s1-s2).powi(2) + (s2-s3).powi(2) + (s3-s1).powi(2)).sqrt() / 2.0_f64.sqrt();
    println!("Von Mises stress: {:.2}", von_mises);
}
```

---

## 3. ndarray — N-Dimensional Arrays

### When to Use ndarray vs nalgebra

| Use Case | Library |
|----------|---------|
| Fixed-size matrices, transforms | `nalgebra` |
| Linear algebra (decompositions) | `nalgebra` |
| N-dimensional data, broadcasting | `ndarray` |
| NumPy-like operations | `ndarray` |
| Scientific computing pipelines | `ndarray` |

### Basic Usage

```rust
use ndarray::{array, Array1, Array2, Array, Axis};
use ndarray::s;  // Slicing macro

fn main() {
    // Create arrays
    let a: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
    let b: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];

    // From shape and function
    let c = Array::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64);

    // Zeros, ones
    let zeros: Array2<f64> = Array::zeros((10, 10));
    let ones: Array2<f64> = Array::ones((5, 5));

    // Linspace
    let x = Array::linspace(0.0, 1.0, 100);
}
```

### Slicing and Views

```rust
use ndarray::{array, Array2, s};

fn main() {
    let a: Array2<f64> = Array2::from_shape_fn((5, 5), |(i, j)| (i * 5 + j) as f64);

    // Slicing (like NumPy)
    let row = a.row(0);           // First row
    let col = a.column(2);        // Third column
    let slice = a.slice(s![1..3, 2..4]);  // Submatrix

    // Mutable slicing
    let mut b = a.clone();
    b.slice_mut(s![0, ..]).fill(0.0);  // Zero first row

    println!("Slice:\n{}", slice);
}
```

### Broadcasting

```rust
use ndarray::{array, Array2, Axis};

fn main() {
    let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    // Broadcast scalar
    let b = &a + 10.0;

    // Broadcast row vector to matrix
    let row = array![1.0, 2.0, 3.0];
    let c = &a + &row;  // Adds row to each row of a

    // Sum along axis
    let row_sums = a.sum_axis(Axis(1));  // Sum each row
    let col_sums = a.sum_axis(Axis(0));  // Sum each column

    println!("Row sums: {}", row_sums);
    println!("Col sums: {}", col_sums);
}
```

### ndarray-linalg for LAPACK

```toml
[dependencies]
ndarray = "0.16"
ndarray-linalg = { version = "0.17", features = ["openblas-static"] }
```

```rust
use ndarray::array;
use ndarray_linalg::{Solve, Eig, SVD};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = array![[3.0, 1.0], [1.0, 2.0]];
    let b = array![9.0, 8.0];

    // Solve Ax = b
    let x = a.solve(&b)?;
    println!("Solution: {}", x);

    // Eigenvalues
    let (eigvals, eigvecs) = a.eig()?;
    println!("Eigenvalues: {:?}", eigvals);

    Ok(())
}
```

---

## 4. faer — High-Performance Linear Algebra

Modern pure-Rust alternative to BLAS/LAPACK:

```toml
[dependencies]
faer = "0.20"
```

```rust
use faer::prelude::*;
use faer::{mat, Mat};

fn main() {
    // Create matrix
    let a = mat![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
    ];

    // LU decomposition
    let lu = a.partial_piv_lu();

    // Solve system
    let b = mat![[1.0], [2.0], [3.0]];
    let x = lu.solve(&b);

    println!("Solution:\n{:?}", x);

    // SVD
    let svd = a.svd();
    println!("Singular values: {:?}", svd.s_diagonal());
}
```

---

## 5. russell_tensor — Continuum Mechanics

### Installation

```toml
[dependencies]
russell_tensor = "1.0"
```

### Tensor Basics

```rust
use russell_tensor::{Tensor2, Tensor4, Mandel};

fn main() {
    // 2nd order tensor (3×3 symmetric → 6 components in Mandel)
    let mut stress = Tensor2::new(Mandel::Symmetric);

    // Set components
    stress.sym_set(0, 0, 100.0);  // σ_xx
    stress.sym_set(1, 1, 50.0);   // σ_yy
    stress.sym_set(2, 2, 30.0);   // σ_zz
    stress.sym_set(0, 1, 20.0);   // σ_xy

    // Invariants
    let i1 = stress.trace();                    // I₁ = tr(σ)
    let i2 = stress.invariant_jj();             // J₂
    let i3 = stress.determinant();              // I₃ = det(σ)

    println!("I₁ = {:.2}", i1);
    println!("J₂ = {:.2}", i2);
    println!("I₃ = {:.2}", i3);

    // Deviatoric stress
    let mut dev = Tensor2::new(Mandel::Symmetric);
    stress.deviator(&mut dev);

    // Von Mises stress
    let von_mises = (3.0 * dev.invariant_jj()).sqrt();
    println!("Von Mises: {:.2}", von_mises);
}
```

### 4th Order Tensors (Elasticity)

```rust
use russell_tensor::{Tensor2, Tensor4, Mandel};

/// Create isotropic elasticity tensor
fn isotropic_elasticity(e: f64, nu: f64) -> Tensor4 {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));

    let mut c = Tensor4::new(Mandel::Symmetric);

    // C_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)
    // Using Mandel representation

    // Diagonal terms
    for i in 0..3 {
        c.sym_set(i, i, i, i, lambda + 2.0 * mu);
        for j in 0..3 {
            if i != j {
                c.sym_set(i, i, j, j, lambda);
            }
        }
    }

    // Shear terms
    c.sym_set(0, 1, 0, 1, mu);
    c.sym_set(1, 2, 1, 2, mu);
    c.sym_set(0, 2, 0, 2, mu);

    c
}

fn main() {
    // Steel: E = 200 GPa, ν = 0.3
    let c = isotropic_elasticity(200e9, 0.3);

    // Apply to strain
    let mut strain = Tensor2::new(Mandel::Symmetric);
    strain.sym_set(0, 0, 0.001);  // 0.1% strain in x

    let mut stress = Tensor2::new(Mandel::Symmetric);
    c.double_dot_tensor2(&strain, &mut stress);

    println!("Stress from 0.1% uniaxial strain:");
    println!("σ_xx = {:.2} MPa", stress.get(0, 0) / 1e6);
    println!("σ_yy = {:.2} MPa", stress.get(1, 1) / 1e6);
}
```

---

## Summary

| Library | Strength | Use When |
|---------|----------|----------|
| `nalgebra` | Type-safe, decompositions | Small/medium matrices, transforms |
| `ndarray` | N-dimensional, broadcasting | Large arrays, data processing |
| `faer` | Performance, pure Rust | Large matrices, HPC |
| `russell_tensor` | Continuum mechanics | Stress/strain, constitutive models |

---

## Next Steps

- **Seminar 3.1**: [nalgebra Fundamentals](../seminars/seminar-03-1.md)
- **Seminar 3.2**: [Tensor Operations for Mechanics](../seminars/seminar-03-2.md)

## Additional Reading

- [nalgebra User Guide](https://nalgebra.org/docs/)
- [ndarray Documentation](https://docs.rs/ndarray)
- [russell Documentation](https://github.com/cpmech/russell)
