---
slug: /
sidebar_position: 1
title: Introduction
---

# Rust for Applied Mathematics and Scientific Computing

Welcome to the university course on Rust programming language, designed specifically for students of applied mathematics and computer science.

## Course Overview

| Block | Lectures | Focus |
|-------|----------|-------|
| I. Foundations | 1-2 | C++ → Rust transition |
| II. Scientific Computing | 3-4 | Math libraries & tooling |
| III. Domain Applications | 5 | Continuum mechanics & numerical methods |
| IV. Advanced Topics | 6-7 | Unsafe Rust, FFI, verification |

## Prerequisites

This course assumes you have:

- **C/C++ programming experience** — you understand pointers, memory management, RAII
- **Computational mathematics background** — linear algebra, numerical methods, ODEs/PDEs
- **Statistics fundamentals** — probability distributions, hypothesis testing
- **Future application** — this course prepares you for continuum mechanics work

## Why Rust for Scientific Computing?

### Memory Safety Without Garbage Collection

```rust
// Rust prevents use-after-free at compile time
fn safe_reference() {
    let data = vec![1, 2, 3];
    let reference = &data;
    // data.clear(); // ERROR: cannot borrow as mutable
    println!("{:?}", reference);
}
```

### Zero-Cost Abstractions

Rust's iterators compile to the same machine code as hand-written loops:

```rust
// High-level, readable code
let sum: f64 = matrix.iter()
    .map(|row| row.iter().sum::<f64>())
    .sum();

// Compiles to efficient assembly — no runtime overhead
```

### Fearless Concurrency

The type system prevents data races at compile time:

```rust
use std::thread;

fn parallel_computation(data: &[f64]) -> f64 {
    // Rust ensures thread-safe access
    data.par_iter()  // rayon parallel iterator
        .map(|x| x * x)
        .sum()
}
```

### Excellent C/C++ Interoperability

Call existing BLAS, LAPACK, or your legacy Fortran code:

```rust
extern "C" {
    fn dgemm_(/* ... */);
}
```

## Course Structure

### Lectures (7 × 90 minutes)

Each lecture introduces concepts with comparisons to C++ equivalents you already know.

### Seminars (14 × 90 minutes)

Hands-on coding sessions where you'll:
- Implement numerical algorithms
- Work with scientific libraries
- Build progressively complex projects

### Assessment

| Component | Weight |
|-----------|--------|
| Homework | 30% |
| Midterm (practical) | 20% |
| Final Project | 40% |
| Participation | 10% |

## Getting Started

Before the first lecture, please:

1. **Install Rust**: Visit [rustup.rs](https://rustup.rs)
2. **Set up IDE**: Install [VS Code](https://code.visualstudio.com/) with [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
3. **Verify installation**:

```bash
rustc --version
cargo --version
```

## Key Libraries We'll Use

| Library | Purpose |
|---------|---------|
| `nalgebra` | Linear algebra, matrices |
| `ndarray` | N-dimensional arrays |
| `russell` | Tensors, ODE solvers, continuum mechanics |
| `fenris` | Finite element methods |
| `linfa` | Machine learning |
| `polars` | Data manipulation |
| `statrs` | Statistical distributions |

## Navigation

Use the sidebar to navigate through lectures and seminars. Each lecture has two associated seminars that follow immediately after.

---

Ready to begin? Start with [Lecture 1: Ecosystem and Ownership Model](./lectures/lecture-01.md).
