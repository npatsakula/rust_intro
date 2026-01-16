---
sidebar_position: 14
title: "Seminar 7.2: Final Project"
---

# Seminar 7.2: Verification and Final Project Setup

**Duration:** 90 minutes
**Prerequisites:** All previous lectures and seminars

---

## Objectives

- Introduction to Kani verification
- Set up CI/CD pipeline
- Discuss final project options
- Code review practices

---

## Task 1: Kani Verification (25 min)

### 1.1 Install Kani

```bash
cargo install --locked kani-verifier
kani setup
```

### 1.2 Simple Verification

```rust
// src/lib.rs
pub fn safe_div(a: i32, b: i32) -> Option<i32> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

pub fn array_access(arr: &[i32], index: usize) -> Option<i32> {
    if index < arr.len() {
        Some(arr[index])
    } else {
        None
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_safe_div_no_panic() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();

        // This should never panic
        let result = safe_div(a, b);

        if b == 0 {
            assert!(result.is_none());
        } else {
            assert!(result.is_some());
        }
    }

    #[kani::proof]
    #[kani::unwind(11)]
    fn verify_array_access() {
        let size: usize = kani::any();
        kani::assume(size > 0 && size <= 10);

        let arr: Vec<i32> = vec![42; size];
        let index: usize = kani::any();

        let result = array_access(&arr, index);

        if index < size {
            assert!(result == Some(42));
        } else {
            assert!(result.is_none());
        }
    }
}
```

### 1.3 Verify Numerical Algorithm

```rust
#[cfg(kani)]
mod numerical_verification {
    /// Verify binary search always finds correct position
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_binary_search() {
        let size: usize = kani::any();
        kani::assume(size > 0 && size <= 4);

        // Create sorted array
        let arr: Vec<i32> = (0..size as i32).collect();
        let target: i32 = kani::any();

        let result = arr.binary_search(&target);

        match result {
            Ok(idx) => {
                assert!(idx < size);
                assert!(arr[idx] == target);
            }
            Err(idx) => {
                assert!(idx <= size);
                // target not in array
                assert!(!arr.contains(&target));
            }
        }
    }
}
```

### 1.4 Run Kani

```bash
cargo kani --tests
```

---

## Task 2: CI/CD Setup (20 min)

### 2.1 GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always

jobs:
  check:
    name: Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo check --all-features

  test:
    name: Test Suite
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all-features

  fmt:
    name: Rustfmt
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - run: cargo fmt --all -- --check

  clippy:
    name: Clippy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - run: cargo clippy --all-features -- -D warnings

  docs:
    name: Documentation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo doc --no-deps --all-features
        env:
          RUSTDOCFLAGS: -D warnings

  property-tests:
    name: Property Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --release -- property
        env:
          PROPTEST_CASES: 5000

  miri:
    name: Miri
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
        with:
          components: miri
      - run: cargo +nightly miri test --lib
```

### 2.2 Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: cargo-fmt
        name: cargo fmt
        entry: cargo fmt --
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-clippy
        name: cargo clippy
        entry: cargo clippy -- -D warnings
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-test
        name: cargo test
        entry: cargo test
        language: system
        types: [rust]
        pass_filenames: false
```

---

## Task 3: Code Review Practices (15 min)

### 3.1 Review Checklist for Scientific Code

```markdown
## Code Review Checklist

### Correctness
- [ ] Algorithm matches mathematical specification
- [ ] Edge cases handled (empty input, singular matrices, etc.)
- [ ] Numerical stability considered
- [ ] Error bounds documented

### Safety
- [ ] No unsafe blocks (or properly documented)
- [ ] Error handling is appropriate
- [ ] No panics in library code
- [ ] Resources properly cleaned up

### Testing
- [ ] Unit tests for new functions
- [ ] Property tests for mathematical properties
- [ ] Integration tests for workflows
- [ ] Test coverage adequate

### Performance
- [ ] No unnecessary allocations in hot paths
- [ ] Appropriate algorithm complexity
- [ ] Benchmarks for performance-critical code

### Documentation
- [ ] Public API documented
- [ ] Mathematical notation explained
- [ ] Examples provided
- [ ] References to papers/algorithms
```

### 3.2 Example Review Comments

```rust
// REVIEW: This loop can be vectorized using iterators
// Before:
for i in 0..n {
    sum += data[i] * weights[i];
}

// After:
let sum: f64 = data.iter()
    .zip(weights.iter())
    .map(|(d, w)| d * w)
    .sum();

// REVIEW: Consider numerical stability
// Before (catastrophic cancellation):
let result = (a + b) - b;  // May lose precision

// After:
let result = a;  // Direct assignment if mathematically equivalent
```

---

## Task 4: Final Project Discussion (30 min)

### Option 1: 2D Elasticity Solver

**Scope:**
- Triangular finite elements
- Plane stress/strain formulation
- Sparse matrix assembly
- Visualization of results

**Requirements:**
- Property tests for stiffness matrix
- Benchmark against reference solution
- Documentation with mathematical derivation

### Option 2: ODE Integrator Library

**Scope:**
- Multiple Runge-Kutta methods
- Adaptive time stepping
- Error estimation
- Stiff system support

**Requirements:**
- Property tests for order of accuracy
- Comparison with analytical solutions
- Energy conservation verification

### Option 3: Data Analysis Pipeline

**Scope:**
- CSV/Parquet data loading
- Statistical analysis functions
- Curve fitting capabilities
- Report generation

**Requirements:**
- Property tests for statistical functions
- Fuzzing for input parsing
- CI/CD with automated testing

### Option 4: Port C++ Scientific Code

**Scope:**
- Choose existing C++ library
- Create safe Rust wrapper
- Idiomatic Rust API
- Comprehensive testing

**Requirements:**
- FFI safety documentation
- Performance comparison
- Memory safety verification (miri)

### Project Template

```
final_project/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   └── ...
├── tests/
│   ├── integration_tests.rs
│   └── property_tests.rs
├── benches/
│   └── benchmarks.rs
├── examples/
│   └── basic_usage.rs
└── .github/
    └── workflows/
        └── ci.yml
```

### Grading Rubric

| Component | Weight | Criteria |
|-----------|--------|----------|
| Functionality | 30% | Correct implementation |
| Testing | 25% | Unit, property, integration |
| Code Quality | 20% | Style, documentation |
| Performance | 15% | Benchmarks, optimization |
| Presentation | 10% | README, examples |

---

## Project Setup

```bash
# Create project
cargo new final_project --lib
cd final_project

# Add dependencies
cargo add nalgebra ndarray thiserror

# Add dev dependencies
cargo add --dev proptest criterion

# Initialize git
git init
git add .
git commit -m "Initial project structure"

# Create GitHub repo and push
gh repo create final_project --public
git push -u origin main
```

---

## Summary

- ✅ Learned Kani verification basics
- ✅ Set up CI/CD pipeline
- ✅ Discussed code review practices
- ✅ Chose final project topic

## Course Completion

Congratulations on completing the Rust for Scientific Computing course!

### What You've Learned

1. **Rust Fundamentals**: Ownership, borrowing, lifetimes
2. **Scientific Libraries**: nalgebra, ndarray, russell
3. **Numerical Methods**: ODE solvers, FEM basics
4. **Advanced Topics**: Unsafe, FFI, verification
5. **Best Practices**: Testing, CI/CD, documentation

### Next Steps

- Complete final project
- Contribute to open-source Rust scientific libraries
- Apply skills to continuum mechanics research
- Share knowledge with colleagues

### Resources

- [Rust Scientific Computing](https://rust-sci.github.io/)
- [Are We Learning Yet](https://www.arewelearningyet.com/)
- [This Week in Rust](https://this-week-in-rust.org/)
