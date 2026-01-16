---
sidebar_position: 12
title: "Seminar 6.2: C/C++ Integration"
---

# Seminar 6.2: C/C++ Integration

**Duration:** 90 minutes
**Prerequisites:** Lecture 6, Seminar 6.1

---

## Objectives

- Use bindgen to wrap C libraries
- Call Rust from C
- Integrate C++ with cxx
- Benchmark Rust vs C matrix operations

---

## Task 1: Wrapping C Library with bindgen (25 min)

### 1.1 Example: Wrapping a Simple C Library

Create `c_math/math_lib.h`:
```c
#ifndef MATH_LIB_H
#define MATH_LIB_H

typedef struct {
    double* data;
    int rows;
    int cols;
} Matrix;

Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix* m);
void matrix_set(Matrix* m, int i, int j, double value);
double matrix_get(const Matrix* m, int i, int j);
Matrix* matrix_multiply(const Matrix* a, const Matrix* b);
double matrix_norm(const Matrix* m);

#endif
```

Create `c_math/math_lib.c`:
```c
#include "math_lib.h"
#include <stdlib.h>
#include <math.h>

Matrix* matrix_create(int rows, int cols) {
    Matrix* m = malloc(sizeof(Matrix));
    m->data = calloc(rows * cols, sizeof(double));
    m->rows = rows;
    m->cols = cols;
    return m;
}

void matrix_free(Matrix* m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

void matrix_set(Matrix* m, int i, int j, double value) {
    m->data[i * m->cols + j] = value;
}

double matrix_get(const Matrix* m, int i, int j) {
    return m->data[i * m->cols + j];
}

Matrix* matrix_multiply(const Matrix* a, const Matrix* b) {
    if (a->cols != b->rows) return NULL;

    Matrix* c = matrix_create(a->rows, b->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += matrix_get(a, i, k) * matrix_get(b, k, j);
            }
            matrix_set(c, i, j, sum);
        }
    }

    return c;
}

double matrix_norm(const Matrix* m) {
    double sum = 0.0;
    for (int i = 0; i < m->rows * m->cols; i++) {
        sum += m->data[i] * m->data[i];
    }
    return sqrt(sum);
}
```

### 1.2 Setup Cargo Project

```toml
# Cargo.toml
[package]
name = "c_wrapper"
version = "0.1.0"
edition = "2024"

[build-dependencies]
bindgen = "0.70"
cc = "1.0"
```

### 1.3 Build Script

```rust
// build.rs
use std::env;
use std::path::PathBuf;

fn main() {
    // Compile C library
    cc::Build::new()
        .file("c_math/math_lib.c")
        .compile("math_lib");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("c_math/math_lib.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
```

### 1.4 Safe Rust Wrapper

```rust
// src/lib.rs
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Safe wrapper around C Matrix
pub struct SafeMatrix {
    ptr: *mut Matrix,
}

impl SafeMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let ptr = unsafe { matrix_create(rows as i32, cols as i32) };
        assert!(!ptr.is_null());
        SafeMatrix { ptr }
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        unsafe {
            matrix_set(self.ptr, i as i32, j as i32, value);
        }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        unsafe { matrix_get(self.ptr, i as i32, j as i32) }
    }

    pub fn multiply(&self, other: &SafeMatrix) -> Option<SafeMatrix> {
        let result = unsafe { matrix_multiply(self.ptr, other.ptr) };
        if result.is_null() {
            None
        } else {
            Some(SafeMatrix { ptr: result })
        }
    }

    pub fn norm(&self) -> f64 {
        unsafe { matrix_norm(self.ptr) }
    }

    pub fn rows(&self) -> usize {
        unsafe { (*self.ptr).rows as usize }
    }

    pub fn cols(&self) -> usize {
        unsafe { (*self.ptr).cols as usize }
    }
}

impl Drop for SafeMatrix {
    fn drop(&mut self) {
        unsafe {
            matrix_free(self.ptr);
        }
    }
}

// Ensure SafeMatrix can't be sent across threads
// (C library might not be thread-safe)
impl !Send for SafeMatrix {}
impl !Sync for SafeMatrix {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_operations() {
        let mut m = SafeMatrix::new(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 4.0);

        let norm = m.norm();
        assert!((norm - (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt()).abs() < 1e-10);
    }
}
```

---

## Task 2: Calling Rust from C (20 min)

### 2.1 Rust Library

```rust
// src/lib.rs
use std::slice;

/// Exposed to C: compute sum of array
#[no_mangle]
pub extern "C" fn rust_sum(data: *const f64, len: usize) -> f64 {
    if data.is_null() {
        return 0.0;
    }

    // SAFETY: Caller must ensure data is valid for len elements
    let slice = unsafe { slice::from_raw_parts(data, len) };
    slice.iter().sum()
}

/// Exposed to C: compute dot product
#[no_mangle]
pub extern "C" fn rust_dot_product(
    a: *const f64,
    b: *const f64,
    len: usize,
) -> f64 {
    if a.is_null() || b.is_null() {
        return 0.0;
    }

    // SAFETY: Caller must ensure validity
    let slice_a = unsafe { slice::from_raw_parts(a, len) };
    let slice_b = unsafe { slice::from_raw_parts(b, len) };

    slice_a.iter()
        .zip(slice_b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Matrix multiply: C = A * B
#[no_mangle]
pub extern "C" fn rust_matrix_multiply(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    n: usize,
    k: usize,
) {
    if a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    // SAFETY: Caller guarantees dimensions
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += *a.add(i * k + l) * *b.add(l * n + j);
                }
                *c.add(i * n + j) = sum;
            }
        }
    }
}
```

### 2.2 Generate C Header with cbindgen

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib", "staticlib"]

[build-dependencies]
cbindgen = "0.27"
```

```rust
// build.rs (add to existing)
let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

cbindgen::Builder::new()
    .with_crate(crate_dir)
    .with_language(cbindgen::Language::C)
    .generate()
    .expect("Unable to generate C bindings")
    .write_to_file("include/rust_math.h");
```

### 2.3 Use from C

```c
// main.c
#include <stdio.h>
#include "rust_math.h"

int main() {
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double sum = rust_sum(data, 5);
    printf("Sum: %f\n", sum);

    double a[] = {1.0, 2.0, 3.0};
    double b[] = {4.0, 5.0, 6.0};
    double dot = rust_dot_product(a, b, 3);
    printf("Dot product: %f\n", dot);

    return 0;
}
```

---

## Task 3: C++ Integration with cxx (25 min)

```toml
# Cargo.toml
[dependencies]
cxx = "1.0"

[build-dependencies]
cxx-build = "1.0"
```

### 3.1 Define Bridge

```rust
// src/lib.rs
#[cxx::bridge]
mod ffi {
    // Shared types
    struct Point {
        x: f64,
        y: f64,
        z: f64,
    }

    // C++ functions callable from Rust
    unsafe extern "C++" {
        include!("my_project/cpp_solver.h");

        type CppSolver;

        fn new_solver() -> UniquePtr<CppSolver>;
        fn solve(self: Pin<&mut CppSolver>, data: &[f64]) -> Vec<f64>;
        fn get_iterations(self: &CppSolver) -> i32;
    }

    // Rust functions callable from C++
    extern "Rust" {
        fn rust_process(data: &[f64]) -> Vec<f64>;
        fn rust_compute_norm(data: &[f64]) -> f64;
    }
}

fn rust_process(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x * 2.0).collect()
}

fn rust_compute_norm(data: &[f64]) -> f64 {
    data.iter().map(|x| x * x).sum::<f64>().sqrt()
}
```

### 3.2 C++ Implementation

```cpp
// include/my_project/cpp_solver.h
#pragma once
#include <vector>
#include <memory>

class CppSolver {
public:
    CppSolver();
    std::vector<double> solve(rust::Slice<const double> data);
    int get_iterations() const;

private:
    int iterations_;
};

std::unique_ptr<CppSolver> new_solver();
```

```cpp
// src/cpp_solver.cpp
#include "my_project/cpp_solver.h"

CppSolver::CppSolver() : iterations_(0) {}

std::vector<double> CppSolver::solve(rust::Slice<const double> data) {
    std::vector<double> result;
    result.reserve(data.size());

    for (auto x : data) {
        result.push_back(x * x);
    }

    iterations_ = 1;
    return result;
}

int CppSolver::get_iterations() const {
    return iterations_;
}

std::unique_ptr<CppSolver> new_solver() {
    return std::make_unique<CppSolver>();
}
```

---

## Task 4: Benchmark Comparison (15 min)

```rust
use std::time::Instant;

fn benchmark_matrix_multiply() {
    let n = 500;

    // Create test matrices
    let a: Vec<f64> = (0..n*n).map(|i| (i % 100) as f64).collect();
    let b: Vec<f64> = (0..n*n).map(|i| ((i + 50) % 100) as f64).collect();

    // Rust implementation
    let start = Instant::now();
    let mut c_rust = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c_rust[i * n + j] = sum;
        }
    }
    let rust_time = start.elapsed();

    // C implementation (via wrapper)
    let start = Instant::now();
    let mut m_a = SafeMatrix::new(n, n);
    let mut m_b = SafeMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            m_a.set(i, j, a[i * n + j]);
            m_b.set(i, j, b[i * n + j]);
        }
    }
    let m_c = m_a.multiply(&m_b).unwrap();
    let c_time = start.elapsed();

    println!("Matrix multiply {}x{}:", n, n);
    println!("  Rust: {:?}", rust_time);
    println!("  C:    {:?}", c_time);
    println!("  Speedup: {:.2}x",
             c_time.as_nanos() as f64 / rust_time.as_nanos() as f64);

    // Verify results match
    let mut max_diff = 0.0;
    for i in 0..n {
        for j in 0..n {
            let diff = (c_rust[i * n + j] - m_c.get(i, j)).abs();
            max_diff = max_diff.max(diff);
        }
    }
    println!("  Max difference: {:.2e}", max_diff);
}
```

---

## Homework

### Assignment: Wrap Domain-Specific Library

Choose one:

1. **Wrap SuiteSparse** (sparse matrix solver)
2. **Wrap FFTW** (FFT library)
3. **Wrap your own C/C++ code**

Requirements:
- Use bindgen for C or cxx for C++
- Create safe Rust wrapper
- Write tests
- Document safety requirements

**Submission:** Branch `homework-6-2`

---

## Summary

- ✅ Used bindgen to wrap C library
- ✅ Exposed Rust functions to C
- ✅ Integrated C++ with cxx
- ✅ Benchmarked performance

## Next

Continue to [Lecture 7: Verification, Testing, and Production](../lectures/lecture-07.md)
