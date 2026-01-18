---
sidebar_position: 3
title: "Lecture 3: Linear Algebra and Tensor Libraries"
---

# Lecture 3: Linear Algebra and Tensor Libraries

**Duration:** 90 minutes
**Block:** II -- Mathematical Tooling

---

## From Ownership to Abstraction

In [Lecture 1](./lecture-01.md), we built Rust's foundation: ownership, borrowing, and memory safety. In [Lecture 2](./lecture-02.md), we learned traits and types--how to make code flexible and reusable. Now we have the tools to write **correct** Rust code. But scientific computing isn't about memory management--it's about **mathematics**.

You need to solve linear systems, compute eigenvalues, work with tensors, process massive arrays. In C++, you'd reach for Eigen or LAPACK. They're powerful, but... well, you know the story. Memory management, template errors, linking issues. And when things go wrong? Segfaults that are nightmares to debug.

Rust brings the same mathematical power, but with the safety guarantees you've come to expect. The type system from Lecture 2 applies here: linear algebra libraries use traits to provide compile-time dimension checking while maintaining zero-cost performance.

Today, you'll master the scientific computing ecosystem and write code that's both **fast** and **impossible to misuse**.

## What You'll Build Today

By the end of this lecture, you'll be able to:
- **Choose the right library** for each scientific computing task
- **Write type-safe linear algebra** with compile-time dimension checking
- **Process N-dimensional data** using Rust's powerful array ecosystem
- **Implement continuum mechanics** with tensor operations
- **Transition from C++'s Eigen** to Rust's native libraries

---

## 1. Rust Numerical Ecosystem Overview

### The Scientific Computing Landscape

Quick quiz: Which language has the best scientific computing ecosystem? It's a trick question--the answer depends on what you're doing:

- **Python**: NumPy is unbeatable for prototyping and data science
- **C++**: Eigen and LAPACK give you raw performance and flexibility
- **Rust**: You get C++'s performance **plus** safety, with growing ecosystem

Here's how they compare:

| Feature | Python (NumPy/SciPy) | C++ (Eigen) | Rust |
|---------|---------------------|-------------|------|
| Maturity | Very mature | Very mature | Growing fast |
| Performance | Good (via C) | Excellent | Excellent (often beats C++) |
| Safety | Runtime checks | Manual | Compile-time guaranteed |
| Development Speed | Fast (prototyping) | Slow (boilerplate) | Medium |
| Ecosystem | Huge | Medium | Growing & curated |

### Why Rust's Ecosystem Matters

Remember the ownership system from [Lecture 1](./lecture-01.md)? And traits from [Lecture 2](./lecture-02.md)? They combine perfectly for scientific computing:

**Ownership** ensures you can't use freed memory--no more mysterious segfaults after Eigen calls.

**Traits** let libraries express mathematical constraints: a matrix multiplication function can only accept compatible dimensions at compile time.

**Zero-cost abstractions** mean you get NumPy's convenience without runtime overhead.

### Key Libraries Overview

| Library | Purpose | Python Equivalent | When to Use |
|---------|---------|-------------------|--------------|
| `nalgebra` | Linear algebra | NumPy + parts of SciPy | Small/medium matrices, transformations, graphics |
| `ndarray` | N-dimensional arrays | NumPy ndarray | Large arrays, broadcasting, data processing |
| `faer` | High-perf linear algebra | NumPy + LAPACK | Large matrices, HPC, performance-critical code |
| `russell` | Scientific computing | SciPy | Continuum mechanics, tensors, FEM |
| `polars` | DataFrames | Pandas | Data analysis, CSV processing |

---

## 2. nalgebra -- Core Linear Algebra

### From Eigen to nalgebra

If you're coming from C++, you know **Eigen**. It's excellent--fast, flexible, widely used. But it has quirks:

```cpp
// Eigen: Dynamic sizes only
Eigen::MatrixXd a(3, 3);  // OK, size determined at runtime
Eigen::Matrix3d b;         // Fixed size 3x3 (compile-time known)

// But: Template errors are cryptic
auto result = a * b;  // What type is this? Good luck debugging!
```

And here's the killer: **dimension mismatches are runtime errors**:

```cpp
Eigen::Matrix3d a(3, 3);
Eigen::Matrix3d b(3, 3);
auto c = a * a;  // Compiles! But crashes at runtime if you misuse dimensions
```

Eigen uses **expression templates**--powerful for performance, but painful for debugging.

**Enter nalgebra:**

```rust
use nalgebra::{Matrix3, Matrix4, DMatrix};

fn main() {
    // Static size -- known at compile time
    let m: Matrix3<f64> = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    );

    // Dynamic size -- determined at runtime
    let m: DMatrix<f64> = DMatrix::zeros(100, 100);
}
```

### Installation

```toml
[dependencies]
nalgebra = "0.33"
```

### Why This Matters: Compile-Time Dimension Checking

Remember traits from [Lecture 2](./lecture-02.md)? nalgebra uses them to make matrices **type-safe**:

### Static vs Dynamic Matrices

```rust
use nalgebra::{Matrix3, Matrix4, DMatrix, Vector3, DVector};
use nalgebra::{SMatrix, SVector};  // Type aliases

fn main() {
    // Static size -- known at compile time
    let m: Matrix3<f64> = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    );

    // Generic static size
    let m: SMatrix<f64, 4, 3> = SMatrix::zeros();

    // Dynamic size -- determined at runtime
    let m: DMatrix<f64> = DMatrix::zeros(100, 100);

    // Vectors
    let v: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
    let v: DVector<f64> = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
}
```

**Eigen Comparison:**

```cpp
// Eigen: You choose between fixed and dynamic at runtime
Eigen::Matrix3d a;  // Fixed size
Eigen::MatrixXd b(3, 3);  // Dynamic size

// Mix them? Runtime checks and possible crashes!
```

In Rust, you choose at **compile time**--ownership and types work together to prevent misuse.

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

    // Valid: (2x3) x (3x2) = (2x2)
    let c: Matrix2<f64> = a * b;

    // Compile error! (2x3) x (2x3) dimension mismatch
    // let d = a * a;
}
```

### Matrix Operations

In scientific computing, you constantly multiply matrices, solve linear systems, compute eigenvalues. Here's how nalgebra makes it **readable** and **safe**:

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
        println!("A * A^-1 ~ I: {}", identity);
    }

    // Solve linear system Ax = b
    let b = Vector3::new(1.0, 2.0, 3.0);
    if let Some(x) = a.lu().solve(&b) {
        println!("Solution x: {}", x);
    }
}
```

**Eigen Comparison:**

```cpp
// Eigen: Operator overloading makes code compact, but errors are vague
Eigen::Matrix3d a = /* ... */;
auto b = a.transpose();  // What type is b?
auto c = a * 2.0;       // What type is c?
auto e = a * b;         // What type is e?

// Inverse: Runtime check required
auto inv = a.inverse();  // Crash if singular!
```

In nalgebra:
- **Types are explicit**--no guessing about expression templates
- **Errors are clear**--compiler tells you exactly what's wrong
- **Safety first**--`try_inverse()` returns `Option` instead of crashing

**Connection to Lecture 2:** This is traits in action! Matrix operations use the `Mul`, `Add`, `Transpose` traits we learned about. nalgebra implements them, and the compiler ensures you use them correctly.

**Connection to Lecture 1:** Notice how we never worry about memory? When `a * b` creates a new matrix, ownership is transferred automatically. When `a.transpose()` returns a view (borrowed), it can't outlive the original.

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

Real-world example: You're implementing finite element analysis (FEM) for a structural simulation. After computing the stress tensor at each element, you need to find the principal stresses--the eigenvalues. This is standard procedure in solid mechanics and fracture mechanics.

Here's how nalgebra makes this straightforward:

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
    println!("Principal stresses: sigma_1={:.2}, sigma_2={:.2}, sigma_3={:.2}",
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

## 3. ndarray -- N-Dimensional Arrays

### When to Use ndarray vs nalgebra

You might be wondering: *I already have nalgebra, why do I need another library?* Great question! They serve different purposes:

| Use Case | Library | Why |
|----------|---------|-------|
| Fixed-size matrices, transforms | `nalgebra` | Compile-time dimensions, graphics |
| Linear algebra (decompositions) | `nalgebra` | Small/medium matrices, clear API |
| N-dimensional data, broadcasting | `ndarray` | Flexible shapes, NumPy-like |
| Large arrays (>1000 elements) | `ndarray` | Better memory layout |
| NumPy-like operations | `ndarray` | Familiar syntax |
| Scientific computing pipelines | `ndarray` | Data processing workflows |

**Think of it this way:** nalgebra is for **algebra**, ndarray is for **data**.

### Basic Usage

NumPy users feel right at home with ndarray:

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

**Connection to Lecture 1 (Borrowing):** Notice `&a` vs `a`? In ndarray, operations can be borrows:
- `&a + 10.0` - creates new array (borrows, doesn't consume)
- `a + &b` - borrows both arrays
- `a.into_iter()` - consumes ownership (array is moved)

Just like we learned with vectors in Lecture 1, you choose borrowing vs moving based on what you need.

**Connection to Lecture 2 (Generics):** `Array<T, D>` is a generic type:
- `T` can be `f64`, `f32`, or even your own numeric type
- `D` is dimension (usize, IxDyn for dynamic)
- Compiler monomorphizes for each concrete type (zero-cost!)

### Slicing and Views

This is where ndarray shines--views that borrow data without copying:

```rust
use ndarray::{array, Array2, s};

fn main() {
    let a: Array2<f64> = Array2::from_shape_fn((5, 5), |(i, j)| (i * 5 + j) as f64);

    // Slicing (like NumPy)
    let row = a.row(0);           // First row (borrowed view)
    let col = a.column(2);        // Third column (borrowed view)
    let slice = a.slice(s![1..3, 2..4]);  // Submatrix (borrowed view)

    // Mutable slicing
    let mut b = a.clone();
    b.slice_mut(s![0, ..]).fill(0.0);  // Zero first row

    println!("Slice:\n{}", slice);
}
```

**Connection to Lecture 1 (Ownership):** These views are **borrows**!
- `row`, `column`, `slice` don't copy data--they point into `a`
- Can't outlive the original array (compiler enforces this!)
- Zero-cost abstraction--same as pointer arithmetic in C

**C++ Comparison:**

```cpp
// Eigen: Views are powerful but error-prone
auto row = a.row(0);  // What lifetime does this have?
// If 'a' gets deleted... UB!

// Or worse:
auto slice = a.block(1, 1, 2, 2);  // Copy? View? Who knows!
```

In Rust, ownership rules make it explicit: if you return a view, the compiler ensures it can't outlive the source.

### Broadcasting

NumPy's killer feature, brought to Rust:

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

**Why This Matters:**

In C++ scientific code, you've probably written this:

```cpp
// Manual broadcasting loop (error-prone!)
Eigen::MatrixXd result = a;
for (int i = 0; i < a.rows(); ++i) {
    result.row(i) += row_vector;  // Did you forget to += vs =?
}

// Or copy into temp arrays
Eigen::MatrixXd temp = a.replicate(1, a.cols()) + row_vector;  // Unnecessary allocation!
```

In Rust with ndarray, broadcasting is:
- **Explicit**--clear what's happening
- **Zero-copy** when possible (uses views)
- **Type-safe**--compiler checks dimensions

**Real-world scenario:** You're processing experimental data from 100 sensors, applying calibration vector to all readings. Broadcasting makes this one line instead of error-prone loop.

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

## 4. faer -- High-Performance Linear Algebra

### The Performance Critical Path

You've probably used LAPACK (or Eigen's LAPACK bindings) for large-scale computations. They're fast, but come with pain:

```cpp
// LAPACK: C bindings, manual memory management
extern "C" {
    void dgesv_(int* n, int* lda, double* a, ...);
}

// You must:
// 1. Manually allocate and free memory
// 2. Correctly call with right parameters
// 3. Handle errors with integer return codes
// 4. Debug segfaults with gdb (nightmare!)
```

**Enter faer:**

Modern pure-Rust alternative that matches LAPACK performance without the C baggage:

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

**Connection to Lecture 1 (Ownership):** Notice how clean this is?
- No manual `malloc`/`free`
- No `extern "C"` function calls
- Memory managed by Rust--ownership system we learned in Lecture 1

**Performance Reality:**

faer beats LAPACK on many benchmarks due to:
- Cache-friendly memory layouts
- SIMD optimizations Rust compiler can't apply to C code
- No C-Rust boundary overhead

**When to Use faer:**

- Large matrices (>1000x1000)
- Performance-critical loops (FEM assembly, CFD solvers)
- When you'd normally link LAPACK or use MKL

**When to Use nalgebra instead:**

- Small/medium matrices (\<1000x1000)
- When compile-time dimensions matter (transforms, graphics)
- When type safety is more important than raw speed

---

## 5. russell_tensor -- Continuum Mechanics

### The Mechanics Problem

In solid mechanics, continuum mechanics, and computational physics, you constantly work with tensors:
- Stress tensors (second-order, 3x3)
- Strain tensors (second-order, 3x3)
- Elasticity tensors (fourth-order, 3x3x3x3)

In C++, you'd implement these as:
- Raw arrays (`std::array<double, 9>` for stress)
- Hand-coded operations (error-prone)
- External libraries (difficult to integrate)

**Enter russell_tensor:** Rust library designed specifically for continuum mechanics

### Installation

```toml
[dependencies]
russell_tensor = "1.0"
```

### Why This Matters: Tensor Operations Made Easy

Let me show you what I mean with a concrete example: computing Von Mises stress from a stress tensor.

### Tensor Basics

```rust
use russell_tensor::{Tensor2, Tensor4, Mandel};

fn main() {
    // 2nd order tensor (3x3 symmetric -> 6 components in Mandel)
    let mut stress = Tensor2::new(Mandel::Symmetric);

    // Set components
    stress.sym_set(0, 0, 100.0);  // sigma_xx
    stress.sym_set(1, 1, 50.0);   // sigma_yy
    stress.sym_set(2, 2, 30.0);   // sigma_zz
    stress.sym_set(0, 1, 20.0);   // sigma_xy

    // Invariants
    let i1 = stress.trace();                    // I_1 = tr(sigma)
    let i2 = stress.invariant_jj();             // J_2
    let i3 = stress.determinant();              // I_3 = det(sigma)

    println!("I_1 = {:.2}", i1);
    println!("J_2 = {:.2}", i2);
    println!("I_3 = {:.2}", i3);

    // Deviatoric stress
    let mut dev = Tensor2::new(Mandel::Symmetric);
    stress.deviator(&mut dev);

    // Von Mises stress
    let von_mises = (3.0 * dev.invariant_jj()).sqrt();
    println!("Von Mises: {:.2}", von_mises);
}
```

### 4th Order Tensors (Elasticity)

Now we get to the complex stuff--fourth-order elasticity tensors. These relate strain to stress: `sigma = C : epsilon` (C is a 3x3x3x3 tensor).

In C++, you've probably hand-coded this:

```cpp
// C++: Raw array representation
std::array<double, 81> C;  // 3x3x3x3

// Or worse:
std::vector<std::vector<std::vector<std::vector<double>>>> C;
```

**russell_tensor makes this readable and safe:**

```rust
use russell_tensor::{Tensor2, Tensor4, Mandel};

/// Create isotropic elasticity tensor
fn isotropic_elasticity(e: f64, nu: f64) -> Tensor4 {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));

    let mut c = Tensor4::new(Mandel::Symmetric);

    // C_ijkl = lambda delta_ij delta_kl + mu (delta_ik delta_jl + delta_il delta_jk)
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
    // Steel: E = 200 GPa, nu = 0.3
    let c = isotropic_elasticity(200e9, 0.3);

    // Apply to strain
    let mut strain = Tensor2::new(Mandel::Symmetric);
    strain.sym_set(0, 0, 0.001);  // 0.1% strain in x

    let mut stress = Tensor2::new(Mandel::Symmetric);
    c.double_dot_tensor2(&strain, &mut stress);

    println!("Stress from 0.1% uniaxial strain:");
    println!("sigma_xx = {:.2} MPa", stress.get(0, 0) / 1e6);
    println!("sigma_yy = {:.2} MPa", stress.get(1, 1) / 1e6);
}
```

**Connection to Lecture 2 (Traits):** russell_tensor defines traits like `Tensor2` and `Tensor4`. These aren't just structs--they're **interfaces** with methods like `double_dot_tensor2()`. This is the trait system in action!

**Connection to Lecture 1 (Ownership):** Notice `&mut stress`? The borrow checker ensures:
- You have exclusive mutable access while computing
- Can't accidentally modify stress concurrently
- Reference can't outlive the borrowed data

**Why This Matters for Scientific Computing:**

- **Constitutive models**: Different materials (isotropic, orthotropic, anisotropic)
- **Finite element analysis**: Computing element stiffness matrices
- **Composite materials**: Layered structures, fiber-reinforced materials

All of these require correct tensor mathematics. russell_tensor handles the bookkeeping so you can focus on the physics.

---

## Summary

So, you now have Rust's mathematical tooling at your fingertips. Let's recap what makes it powerful:

| Library | Why Rust Wins | Perfect For... |
|---------|----------------|------------------|
| `nalgebra` | Type-safe dimensions, no template hell | Small/medium matrices, transforms, real-time apps |
| `ndarray` | Borrowing-based views, generics | Large arrays, data processing pipelines |
| `faer` | Ownership safety, beats LAPACK | Large-scale computations, HPC, performance-critical code |
| `russell_tensor` | Traits for tensor operations | Continuum mechanics, FEM, constitutive models |

**The Big Picture:**

From [Lecture 1](./lecture-01.md)'s ownership - [Lecture 2](./lecture-02.md)'s traits -> Lecture 3's libraries. These pieces combine into something greater than their parts:

- **Ownership** prevents memory errors (L1) in matrix operations
- **Traits** provide flexible interfaces (L2) for linear algebra
- **Zero-cost abstractions** give you NumPy's convenience without runtime overhead
- **Type safety** catches dimension errors at compile time (not during 3am debugging session!)

You're not just switching languages--you're upgrading your entire scientific computing workflow.

---

## Next Steps

You've learned the tools. Now let's put them to work in hands-on seminars:

- **Seminar 3.1**: [nalgebra Fundamentals](../seminars/seminar-03-1.md)
- **Seminar 3.2**: [Tensor Operations for Mechanics](../seminars/seminar-03-2.md)

## Additional Reading

- [nalgebra User Guide](https://nalgebra.org/docs/) -- Your go-to for linear algebra
- [ndarray Documentation](https://docs.rs/ndarray) -- NumPy alternative in Rust
- [russell Documentation](https://github.com/cpmech/russell) -- Continuum mechanics toolkit

---

**Ready for more?** In [Lecture 4](./lecture-04.md), we'll explore parallel computing and async programming--where Rust's type system really shines for scientific workloads.
