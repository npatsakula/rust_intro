---
sidebar_position: 4
title: "Seminar 2.2: Error Handling and Project Structure"
---

# Seminar 2.2: Error Handling and Project Structure

**Duration:** 90 minutes
**Prerequisites:** Lecture 2, Seminar 2.1

---

## Objectives

- Build a multi-module scientific computing library
- Design custom error types with `thiserror`
- Practice error propagation patterns
- Write comprehensive tests
- Generate documentation

---

## Task 1: Project Setup (15 min)

### 1.1 Create Workspace

```bash
mkdir scilib && cd scilib
```

Create `Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["core", "cli"]

[workspace.package]
version = "0.1.0"
edition = "2024"
authors = ["Your Name <your@email.com>"]
license = "MIT"

[workspace.dependencies]
thiserror = "2.0"
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
```

### 1.2 Create Core Library

```bash
cargo new core --lib
```

Edit `core/Cargo.toml`:
```toml
[package]
name = "scilib-core"
version.workspace = true
edition.workspace = true

[dependencies]
thiserror.workspace = true
```

### 1.3 Create CLI Binary

```bash
cargo new cli
```

Edit `cli/Cargo.toml`:
```toml
[package]
name = "scilib-cli"
version.workspace = true
edition.workspace = true

[dependencies]
scilib-core = { path = "../core" }
anyhow.workspace = true
clap.workspace = true
```

---

## Task 2: Design Error Types (20 min)

### 2.1 Define Error Hierarchy

Create `core/src/error.rs`:

```rust
use thiserror::Error;

/// Errors that can occur in matrix operations
#[derive(Error, Debug, PartialEq)]
pub enum MatrixError {
    #[error("dimension mismatch: expected ({expected_rows}, {expected_cols}), got ({actual_rows}, {actual_cols})")]
    DimensionMismatch {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    #[error("index ({row}, {col}) out of bounds for {rows}x{cols} matrix")]
    IndexOutOfBounds {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },

    #[error("matrix must be square, got {rows}x{cols}")]
    NotSquare { rows: usize, cols: usize },

    #[error("matrix is singular (determinant ≈ 0)")]
    SingularMatrix,

    #[error("empty matrix")]
    EmptyMatrix,
}

/// Errors in linear algebra operations
#[derive(Error, Debug)]
pub enum LinalgError {
    #[error("matrix error: {0}")]
    Matrix(#[from] MatrixError),

    #[error("convergence failed after {iterations} iterations (residual: {residual})")]
    ConvergenceFailed { iterations: usize, residual: f64 },

    #[error("invalid tolerance: {0} (must be positive)")]
    InvalidTolerance(f64),
}

/// Top-level library error
#[derive(Error, Debug)]
pub enum ScilibError {
    #[error("linear algebra error: {0}")]
    Linalg(#[from] LinalgError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse error: {message}")]
    Parse { message: String },
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, ScilibError>;
pub type MatrixResult<T> = std::result::Result<T, MatrixError>;
pub type LinalgResult<T> = std::result::Result<T, LinalgError>;
```

### 2.2 Update Library Root

Edit `core/src/lib.rs`:

```rust
pub mod error;
pub mod matrix;
pub mod linalg;

pub use error::{ScilibError, MatrixError, LinalgError, Result};
pub use matrix::Matrix;
```

---

## Task 3: Implement Matrix Module (25 min)

### 3.1 Create Matrix Implementation

Create `core/src/matrix.rs`:

```rust
use crate::error::{MatrixError, MatrixResult};

/// A dense matrix of floating-point numbers
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Create a new matrix filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> MatrixResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::EmptyMatrix);
        }
        Ok(Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        })
    }

    /// Create an identity matrix
    pub fn identity(n: usize) -> MatrixResult<Self> {
        let mut m = Self::zeros(n, n)?;
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        Ok(m)
    }

    /// Create from row-major data
    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> MatrixResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::EmptyMatrix);
        }
        if data.len() != rows * cols {
            return Err(MatrixError::DimensionMismatch {
                expected_rows: rows,
                expected_cols: cols,
                actual_rows: data.len() / cols.max(1),
                actual_cols: if rows > 0 { data.len() / rows } else { 0 },
            });
        }
        Ok(Matrix { data, rows, cols })
    }

    /// Create from 2D vector
    pub fn from_rows(data: Vec<Vec<f64>>) -> MatrixResult<Self> {
        if data.is_empty() || data[0].is_empty() {
            return Err(MatrixError::EmptyMatrix);
        }

        let rows = data.len();
        let cols = data[0].len();

        // Check all rows have same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(MatrixError::DimensionMismatch {
                    expected_rows: rows,
                    expected_cols: cols,
                    actual_rows: rows,
                    actual_cols: row.len(),
                });
            }
        }

        let flat: Vec<f64> = data.into_iter().flatten().collect();
        Ok(Matrix { data: flat, rows, cols })
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn is_square(&self) -> bool { self.rows == self.cols }

    /// Get element at (i, j)
    pub fn get(&self, row: usize, col: usize) -> MatrixResult<f64> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::IndexOutOfBounds {
                row,
                col,
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(self.data[row * self.cols + col])
    }

    /// Set element at (i, j)
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> MatrixResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::IndexOutOfBounds {
                row,
                col,
                rows: self.rows,
                cols: self.cols,
            });
        }
        self.data[row * self.cols + col] = value;
        Ok(())
    }

    /// Get a row as a slice
    pub fn row(&self, i: usize) -> MatrixResult<&[f64]> {
        if i >= self.rows {
            return Err(MatrixError::IndexOutOfBounds {
                row: i,
                col: 0,
                rows: self.rows,
                cols: self.cols,
            });
        }
        let start = i * self.cols;
        Ok(&self.data[start..start + self.cols])
    }

    /// Matrix addition
    pub fn add(&self, other: &Self) -> MatrixResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                expected_rows: self.rows,
                expected_cols: self.cols,
                actual_rows: other.rows,
                actual_cols: other.cols,
            });
        }

        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Matrix multiplication
    pub fn mul(&self, other: &Self) -> MatrixResult<Self> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch {
                expected_rows: self.cols,
                expected_cols: other.rows,
                actual_rows: other.rows,
                actual_cols: other.cols,
            });
        }

        let mut result = Self::zeros(self.rows, other.cols)?;

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
            }
        }

        Ok(result)
    }

    /// Transpose
    pub fn transpose(&self) -> Self {
        let mut result = Matrix {
            data: vec![0.0; self.rows * self.cols],
            rows: self.cols,
            cols: self.rows,
        };

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        result
    }

    /// Frobenius norm
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

// Display implementation
impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:8.4}", self.data[i * self.cols + j])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}
```

---

## Task 4: Implement Linear Algebra Module (15 min)

Create `core/src/linalg.rs`:

```rust
use crate::error::{LinalgError, LinalgResult, MatrixError};
use crate::matrix::Matrix;

/// Solve linear system Ax = b using Gauss-Jordan elimination
pub fn solve(a: &Matrix, b: &Matrix) -> LinalgResult<Matrix> {
    if !a.is_square() {
        return Err(MatrixError::NotSquare {
            rows: a.rows(),
            cols: a.cols(),
        }.into());
    }

    if a.rows() != b.rows() {
        return Err(MatrixError::DimensionMismatch {
            expected_rows: a.rows(),
            expected_cols: 1,
            actual_rows: b.rows(),
            actual_cols: b.cols(),
        }.into());
    }

    let n = a.rows();

    // Create augmented matrix [A | b]
    let mut aug = Matrix::zeros(n, n + b.cols())?;
    for i in 0..n {
        for j in 0..n {
            aug.set(i, j, a.get(i, j)?)?;
        }
        for j in 0..b.cols() {
            aug.set(i, n + j, b.get(i, j)?)?;
        }
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug.get(col, col)?.abs();
        for row in (col + 1)..n {
            let val = aug.get(row, col)?.abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(MatrixError::SingularMatrix.into());
        }

        // Swap rows
        if max_row != col {
            for j in 0..(n + b.cols()) {
                let temp = aug.get(col, j)?;
                aug.set(col, j, aug.get(max_row, j)?)?;
                aug.set(max_row, j, temp)?;
            }
        }

        // Eliminate column
        let pivot = aug.get(col, col)?;
        for row in 0..n {
            if row != col {
                let factor = aug.get(row, col)? / pivot;
                for j in col..(n + b.cols()) {
                    let new_val = aug.get(row, j)? - factor * aug.get(col, j)?;
                    aug.set(row, j, new_val)?;
                }
            }
        }

        // Normalize pivot row
        for j in col..(n + b.cols()) {
            let new_val = aug.get(col, j)? / pivot;
            aug.set(col, j, new_val)?;
        }
    }

    // Extract solution
    let mut result = Matrix::zeros(n, b.cols())?;
    for i in 0..n {
        for j in 0..b.cols() {
            result.set(i, j, aug.get(i, n + j)?)?;
        }
    }

    Ok(result)
}

/// Iterative solver using Jacobi method
pub fn solve_jacobi(
    a: &Matrix,
    b: &Matrix,
    max_iterations: usize,
    tolerance: f64,
) -> LinalgResult<Matrix> {
    if tolerance <= 0.0 {
        return Err(LinalgError::InvalidTolerance(tolerance));
    }

    if !a.is_square() {
        return Err(MatrixError::NotSquare {
            rows: a.rows(),
            cols: a.cols(),
        }.into());
    }

    let n = a.rows();
    let mut x = Matrix::zeros(n, 1)?;
    let mut x_new = Matrix::zeros(n, 1)?;

    for iteration in 0..max_iterations {
        for i in 0..n {
            let mut sigma = 0.0;
            for j in 0..n {
                if i != j {
                    sigma += a.get(i, j)? * x.get(j, 0)?;
                }
            }

            let diag = a.get(i, i)?;
            if diag.abs() < 1e-14 {
                return Err(MatrixError::SingularMatrix.into());
            }

            x_new.set(i, 0, (b.get(i, 0)? - sigma) / diag)?;
        }

        // Check convergence
        let diff = x_new.add(&Matrix::from_data(n, 1,
            x.row(0).unwrap().iter().map(|v| -v).collect()
        )?)?;
        let residual = diff.norm();

        if residual < tolerance {
            return Ok(x_new);
        }

        std::mem::swap(&mut x, &mut x_new);
    }

    Err(LinalgError::ConvergenceFailed {
        iterations: max_iterations,
        residual: x_new.add(&Matrix::from_data(n, 1,
            x.row(0).unwrap().iter().map(|v| -v).collect()
        )?)?.norm(),
    })
}
```

---

## Task 5: Write Tests (10 min)

Create `core/src/matrix.rs` tests section:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m = Matrix::zeros(3, 4).unwrap();
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.get(0, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_zeros_empty() {
        assert!(Matrix::zeros(0, 5).is_err());
        assert!(Matrix::zeros(5, 0).is_err());
    }

    #[test]
    fn test_identity() {
        let m = Matrix::identity(3).unwrap();
        assert_eq!(m.get(0, 0).unwrap(), 1.0);
        assert_eq!(m.get(0, 1).unwrap(), 0.0);
        assert_eq!(m.get(1, 1).unwrap(), 1.0);
    }

    #[test]
    fn test_from_rows() {
        let m = Matrix::from_rows(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]).unwrap();
        assert_eq!(m.get(0, 0).unwrap(), 1.0);
        assert_eq!(m.get(1, 1).unwrap(), 4.0);
    }

    #[test]
    fn test_multiplication() {
        let a = Matrix::from_rows(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]).unwrap();
        let b = Matrix::from_rows(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]).unwrap();
        let c = a.mul(&b).unwrap();

        assert_eq!(c.get(0, 0).unwrap(), 19.0);
        assert_eq!(c.get(0, 1).unwrap(), 22.0);
        assert_eq!(c.get(1, 0).unwrap(), 43.0);
        assert_eq!(c.get(1, 1).unwrap(), 50.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = Matrix::zeros(2, 3).unwrap();
        let b = Matrix::zeros(4, 5).unwrap();

        let result = a.mul(&b);
        assert!(matches!(result, Err(MatrixError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_index_out_of_bounds() {
        let m = Matrix::zeros(3, 3).unwrap();
        assert!(matches!(
            m.get(5, 0),
            Err(MatrixError::IndexOutOfBounds { .. })
        ));
    }
}
```

---

## Task 6: Generate Documentation (5 min)

Add documentation comments to your code:

```rust
/// A dense matrix of floating-point numbers stored in row-major order.
///
/// # Examples
///
/// ```
/// use scilib_core::Matrix;
///
/// let m = Matrix::zeros(3, 3).unwrap();
/// assert_eq!(m.rows(), 3);
///
/// let id = Matrix::identity(2).unwrap();
/// assert_eq!(id.get(0, 0).unwrap(), 1.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    // ...
}
```

Generate documentation:

```bash
cargo doc --open
```

---

## Homework

### Assignment: Create Complete Workspace

Extend the workspace with:

1. **Add `io` module** — read/write matrices from CSV files
2. **Implement CLI** — command-line tool that:
   - Reads matrix from file
   - Performs operation (multiply, transpose, solve)
   - Writes result to file
3. **Add integration tests** in `tests/` directory
4. **Achieve 80%+ code coverage** (use `cargo-tarpaulin`)

**Example CLI usage:**
```bash
scilib-cli solve --matrix a.csv --vector b.csv --output x.csv
scilib-cli multiply --left a.csv --right b.csv --output c.csv
```

**Submission:** Branch `homework-2-2`

---

## Summary

- ✅ Created multi-crate workspace
- ✅ Designed error hierarchy with `thiserror`
- ✅ Implemented Matrix with proper error handling
- ✅ Wrote unit tests
- ✅ Generated documentation

## Next

Continue to [Lecture 3: Linear Algebra and Tensor Libraries](../lectures/lecture-03.md)
