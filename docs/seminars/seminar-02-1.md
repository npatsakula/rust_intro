---
sidebar_position: 3
title: "Seminar 2.1: Traits and Generics"
---

# Seminar 2.1: Traits and Generics

**Duration:** 90 minutes
**Prerequisites:** Lecture 2

---

## Objectives

- Implement custom traits for numeric types
- Create generic data structures with trait bounds
- Implement operator overloading
- Compare static vs dynamic dispatch performance

---

## Task 1: Numeric Trait (25 min)

### 1.1 Define the Trait

Create a trait for types that support basic arithmetic:

```rust
pub trait Numeric: Clone + PartialEq {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Option<Self>;
    fn abs(&self) -> Self;
}
```

### 1.2 Implement for Primitive Types

```rust
impl Numeric for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn add(&self, other: &Self) -> Self { self + other }
    fn sub(&self, other: &Self) -> Self { self - other }
    fn mul(&self, other: &Self) -> Self { self * other }

    fn div(&self, other: &Self) -> Option<Self> {
        if *other == 0.0 {
            None
        } else {
            Some(self / other)
        }
    }

    fn abs(&self) -> Self { f64::abs(*self) }
}

impl Numeric for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn add(&self, other: &Self) -> Self { self + other }
    fn sub(&self, other: &Self) -> Self { self - other }
    fn mul(&self, other: &Self) -> Self { self * other }

    fn div(&self, other: &Self) -> Option<Self> {
        if *other == 0.0 {
            None
        } else {
            Some(self / other)
        }
    }

    fn abs(&self) -> Self { f32::abs(*self) }
}

impl Numeric for i32 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn add(&self, other: &Self) -> Self { self + other }
    fn sub(&self, other: &Self) -> Self { self - other }
    fn mul(&self, other: &Self) -> Self { self * other }

    fn div(&self, other: &Self) -> Option<Self> {
        if *other == 0 {
            None
        } else {
            Some(self / other)
        }
    }

    fn abs(&self) -> Self { i32::abs(*self) }
}
```

### 1.3 Use the Trait in Generic Functions

```rust
/// Compute the sum of a slice using our Numeric trait
fn sum<T: Numeric>(values: &[T]) -> T {
    let mut result = T::zero();
    for value in values {
        result = result.add(value);
    }
    result
}

/// Compute the dot product of two vectors
fn dot_product<T: Numeric>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let mut result = T::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        result = result.add(&x.mul(y));
    }
    result
}

fn main() {
    let floats = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("Sum of floats: {}", sum(&floats));

    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    println!("Dot product: {}", dot_product(&a, &b));  // 1*4 + 2*5 + 3*6 = 32
}
```

---

## Task 2: Generic Matrix (25 min)

### 2.1 Define the Structure

```rust
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Numeric + Debug> Matrix<T> {
    /// Create a new matrix filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![T::zero(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create an identity matrix
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = T::one();
        }
        m
    }

    /// Create from a 2D vector
    pub fn from_2d(data: Vec<Vec<T>>) -> Option<Self> {
        if data.is_empty() {
            return Some(Matrix {
                data: vec![],
                rows: 0,
                cols: 0,
            });
        }

        let rows = data.len();
        let cols = data[0].len();

        // Check all rows have same length
        if !data.iter().all(|row| row.len() == cols) {
            return None;
        }

        let flat: Vec<T> = data.into_iter().flatten().collect();

        Some(Matrix {
            data: flat,
            rows,
            cols,
        })
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    /// Get element at (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i < self.rows && j < self.cols {
            Some(&self.data[i * self.cols + j])
        } else {
            None
        }
    }

    /// Get mutable element at (i, j)
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if i < self.rows && j < self.cols {
            Some(&mut self.data[i * self.cols + j])
        } else {
            None
        }
    }
}
```

### 2.2 Implement Matrix Operations

```rust
impl<T: Numeric + Debug> Matrix<T> {
    /// Matrix addition
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.add(b))
            .collect();

        Some(Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Matrix-scalar multiplication
    pub fn scale(&self, scalar: &T) -> Self {
        let data: Vec<T> = self.data.iter()
            .map(|x| x.mul(scalar))
            .collect();

        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Matrix multiplication
    pub fn mul(&self, other: &Self) -> Option<Self> {
        if self.cols != other.rows {
            return None;
        }

        let mut result = Self::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::zero();
                for k in 0..self.cols {
                    let a = &self.data[i * self.cols + k];
                    let b = &other.data[k * other.cols + j];
                    sum = sum.add(&a.mul(b));
                }
                result.data[i * other.cols + j] = sum;
            }
        }

        Some(result)
    }

    /// Transpose
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j].clone();
            }
        }

        result
    }
}
```

### 2.3 Test the Implementation

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m: Matrix<f64> = Matrix::zeros(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
    }

    #[test]
    fn test_identity() {
        let id: Matrix<f64> = Matrix::identity(3);
        assert_eq!(*id.get(0, 0).unwrap(), 1.0);
        assert_eq!(*id.get(0, 1).unwrap(), 0.0);
        assert_eq!(*id.get(1, 1).unwrap(), 1.0);
    }

    #[test]
    fn test_multiplication() {
        let a = Matrix::from_2d(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]).unwrap();

        let b = Matrix::from_2d(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]).unwrap();

        let c = a.mul(&b).unwrap();

        // [1 2] * [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        assert_eq!(*c.get(0, 0).unwrap(), 19.0);
        assert_eq!(*c.get(0, 1).unwrap(), 22.0);
        assert_eq!(*c.get(1, 0).unwrap(), 43.0);
        assert_eq!(*c.get(1, 1).unwrap(), 50.0);
    }
}
```

---

## Task 3: Operator Overloading (20 min)

### 3.1 Implement Traits for Matrix

```rust
use std::ops::{Add, Mul, Index, IndexMut};

impl<T: Numeric + Debug> Add for Matrix<T> {
    type Output = Option<Matrix<T>>;

    fn add(self, other: Self) -> Self::Output {
        Matrix::add(&self, &other)
    }
}

// Better: implement for references to avoid moves
impl<T: Numeric + Debug> Add for &Matrix<T> {
    type Output = Option<Matrix<T>>;

    fn add(self, other: Self) -> Self::Output {
        Matrix::add(self, other)
    }
}

impl<T: Numeric + Debug> Mul for &Matrix<T> {
    type Output = Option<Matrix<T>>;

    fn mul(self, other: Self) -> Self::Output {
        Matrix::mul(self, other)
    }
}

// Index with tuple (i, j)
impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[i * self.cols + j]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[i * self.cols + j]
    }
}
```

### 3.2 Usage Example

```rust
fn main() {
    let a = Matrix::from_2d(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ]).unwrap();

    let b = Matrix::from_2d(vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ]).unwrap();

    // Using operator overloading
    if let Some(c) = &a * &b {
        println!("a * b = {:?}", c);
    }

    // Using indexing
    let mut m = Matrix::zeros(2, 2);
    m[(0, 0)] = 1.0;
    m[(1, 1)] = 1.0;
    println!("m[0,0] = {}", m[(0, 0)]);
}
```

---

## Task 4: Static vs Dynamic Dispatch (20 min)

### 4.1 Define a Shape Trait

```rust
use std::f64::consts::PI;

trait Shape {
    fn area(&self) -> f64;
    fn perimeter(&self) -> f64;
}

struct Circle {
    radius: f64,
}

struct Rectangle {
    width: f64,
    height: f64,
}

struct Triangle {
    a: f64,
    b: f64,
    c: f64,
}

impl Shape for Circle {
    fn area(&self) -> f64 { PI * self.radius * self.radius }
    fn perimeter(&self) -> f64 { 2.0 * PI * self.radius }
}

impl Shape for Rectangle {
    fn area(&self) -> f64 { self.width * self.height }
    fn perimeter(&self) -> f64 { 2.0 * (self.width + self.height) }
}

impl Shape for Triangle {
    fn area(&self) -> f64 {
        // Heron's formula
        let s = (self.a + self.b + self.c) / 2.0;
        (s * (s - self.a) * (s - self.b) * (s - self.c)).sqrt()
    }
    fn perimeter(&self) -> f64 { self.a + self.b + self.c }
}
```

### 4.2 Static Dispatch (Monomorphization)

```rust
fn total_area_static<T: Shape>(shapes: &[T]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// The compiler generates separate functions for each type:
// total_area_static::<Circle>
// total_area_static::<Rectangle>
// etc.
```

### 4.3 Dynamic Dispatch (Trait Objects)

```rust
fn total_area_dynamic(shapes: &[&dyn Shape]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// Or with Box:
fn total_area_boxed(shapes: &[Box<dyn Shape>]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}
```

### 4.4 Benchmark Comparison

```rust
use std::time::Instant;

fn benchmark() {
    const N: usize = 1_000_000;

    // Create test data
    let circles: Vec<Circle> = (0..N)
        .map(|i| Circle { radius: (i as f64) * 0.001 })
        .collect();

    let shapes_dyn: Vec<&dyn Shape> = circles.iter()
        .map(|c| c as &dyn Shape)
        .collect();

    // Benchmark static dispatch
    let start = Instant::now();
    let sum_static = total_area_static(&circles);
    let static_time = start.elapsed();

    // Benchmark dynamic dispatch
    let start = Instant::now();
    let sum_dynamic = total_area_dynamic(&shapes_dyn);
    let dynamic_time = start.elapsed();

    println!("Static dispatch:  {:?} (sum = {})", static_time, sum_static);
    println!("Dynamic dispatch: {:?} (sum = {})", dynamic_time, sum_dynamic);
    println!("Ratio: {:.2}x", dynamic_time.as_nanos() as f64 / static_time.as_nanos() as f64);
}
```

**Expected results:**
- Static dispatch is typically 2-10x faster
- Dynamic dispatch adds vtable lookup overhead
- Use static when types are known, dynamic when flexibility needed

---

## Task 5: Iterator Implementation (Bonus)

### 5.1 Implement Iterator for Matrix Rows

```rust
pub struct RowIter<'a, T> {
    matrix: &'a Matrix<T>,
    current_row: usize,
}

impl<'a, T> Iterator for RowIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row < self.matrix.rows {
            let start = self.current_row * self.matrix.cols;
            let end = start + self.matrix.cols;
            self.current_row += 1;
            Some(&self.matrix.data[start..end])
        } else {
            None
        }
    }
}

impl<T> Matrix<T> {
    pub fn rows_iter(&self) -> RowIter<T> {
        RowIter {
            matrix: self,
            current_row: 0,
        }
    }
}

// Usage
fn main() {
    let m = Matrix::from_2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]).unwrap();

    for (i, row) in m.rows_iter().enumerate() {
        println!("Row {}: {:?}", i, row);
    }
}
```

---

## Homework

### Assignment: Generic Polynomial Type

Implement a `Polynomial<T>` type with arithmetic operations.

**Requirements:**

```rust
pub struct Polynomial<T> {
    coefficients: Vec<T>,  // coefficients[i] is coefficient of x^i
}

impl<T: Numeric> Polynomial<T> {
    fn new(coefficients: Vec<T>) -> Self;
    fn degree(&self) -> usize;
    fn evaluate(&self, x: &T) -> T;  // Horner's method
}

// Implement Add, Sub, Mul traits
// Implement Display trait for pretty printing

// Example:
// p(x) = 1 + 2x + 3x² would be Polynomial::new(vec![1, 2, 3])
// p(2) = 1 + 4 + 12 = 17
```

**Submission:** Branch `homework-2-1`

---

## Summary

- ✅ Created custom `Numeric` trait
- ✅ Built generic `Matrix<T>` with trait bounds
- ✅ Implemented operator overloading
- ✅ Compared static vs dynamic dispatch
- ✅ Implemented custom iterators

## Next

Continue to [Seminar 2.2: Error Handling and Project Structure](./seminar-02-2.md)
