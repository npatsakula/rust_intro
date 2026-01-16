---
sidebar_position: 2
title: "Lecture 2: Type System, Traits, and Modules"
---

# Lecture 2: Type System, Traits, and Modules

**Duration:** 90 minutes
**Block:** I — From C++ to Rust

---

## Learning Objectives

By the end of this lecture, you will:
- Understand Rust's algebraic data types and pattern matching
- Master traits as Rust's approach to polymorphism
- Implement error handling with `Result` and `Option`
- Organize code with modules and workspaces

---

## 1. Type System Comparison

### Algebraic Data Types: enum

Rust's `enum` is far more powerful than C++ `enum`:

```rust
// C++ style enum (also works in Rust)
enum Color {
    Red,
    Green,
    Blue,
}

// Rust: enum with data (like C++ std::variant)
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(u8, u8, u8),
}

fn process_message(msg: Message) {
    match msg {
        Message::Quit => println!("Quit"),
        Message::Move { x, y } => println!("Move to ({}, {})", x, y),
        Message::Write(text) => println!("Text: {}", text),
        Message::ChangeColor(r, g, b) => println!("RGB({}, {}, {})", r, g, b),
    }
}
```

### Compare with C++ std::variant

```cpp
// C++ approach (verbose)
using Message = std::variant<
    std::monostate,  // Quit
    std::pair<int, int>,  // Move
    std::string,  // Write
    std::tuple<uint8_t, uint8_t, uint8_t>  // ChangeColor
>;

void process_message(const Message& msg) {
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            // ...
        }
        // etc.
    }, msg);
}
```

### Pattern Matching

More powerful than `switch`:

```rust
fn describe_number(n: i32) -> &'static str {
    match n {
        0 => "zero",
        1 | 2 | 3 => "small",
        4..=10 => "medium",
        _ if n < 0 => "negative",
        _ => "large",
    }
}

// Destructuring
struct Point { x: i32, y: i32 }

fn classify_point(p: Point) -> &'static str {
    match p {
        Point { x: 0, y: 0 } => "origin",
        Point { x: 0, .. } => "on y-axis",
        Point { y: 0, .. } => "on x-axis",
        Point { x, y } if x == y => "on diagonal",
        _ => "somewhere else",
    }
}
```

### Generics vs C++ Templates

```rust
// Rust generics
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// C++20 concepts (similar idea)
// template<typename T>
// requires std::totally_ordered<T>
// T max(T a, T b) { return a > b ? a : b; }
```

Key differences:
- Rust checks generic constraints at definition site
- C++ templates checked at instantiation (better error messages in Rust)
- Rust: monomorphization (like C++), but also supports trait objects

---

## 2. Traits — Rust's Polymorphism

### Defining and Implementing Traits

```rust
trait Numeric {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
}

impl Numeric for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn add(&self, other: &Self) -> Self { self + other }
    fn mul(&self, other: &Self) -> Self { self * other }
}

impl Numeric for i32 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn add(&self, other: &Self) -> Self { self + other }
    fn mul(&self, other: &Self) -> Self { self * other }
}
```

### Standard Library Traits

```rust
// Clone: explicit duplication
#[derive(Clone)]
struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}

// Copy: implicit bitwise copy (must also be Clone)
#[derive(Copy, Clone)]
struct Point2D {
    x: f64,
    y: f64,
}

// Debug: {:?} formatting
#[derive(Debug)]
struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

// Default: T::default()
#[derive(Default)]
struct Config {
    iterations: usize,  // defaults to 0
    tolerance: f64,     // defaults to 0.0
}

// PartialEq, Eq: equality comparison
#[derive(PartialEq)]
struct Complex {
    re: f64,
    im: f64,
}
```

### Operator Overloading via Traits

```rust
use std::ops::{Add, Mul, Index};

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, scalar: f64) -> Vec3 {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

// Dot product: Vec3 * Vec3
impl Mul for Vec3 {
    type Output = f64;

    fn mul(self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}
```

### Trait Objects vs Static Dispatch

```rust
trait Shape {
    fn area(&self) -> f64;
}

struct Circle { radius: f64 }
struct Rectangle { width: f64, height: f64 }

impl Shape for Circle {
    fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius }
}

impl Shape for Rectangle {
    fn area(&self) -> f64 { self.width * self.height }
}

// Static dispatch (monomorphization) — like C++ templates
fn print_area<T: Shape>(shape: &T) {
    println!("Area: {}", shape.area());
}

// Dynamic dispatch (trait object) — like C++ virtual functions
fn print_area_dyn(shape: &dyn Shape) {
    println!("Area: {}", shape.area());
}

// Storing mixed types
fn total_area(shapes: &[&dyn Shape]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}
```

---

## 3. Error Handling

### `Result<T, E>` and `Option<T>`

```rust
// Option: value may or may not exist
fn find_element(data: &[i32], target: i32) -> Option<usize> {
    data.iter().position(|&x| x == target)
}

// Result: operation may fail with an error
fn parse_matrix(input: &str) -> Result<Vec<f64>, ParseError> {
    // ...
}
```

### The ? Operator

```rust
use std::fs::File;
use std::io::{self, Read};

// Without ?
fn read_file_verbose(path: &str) -> Result<String, io::Error> {
    let file = File::open(path);
    let mut file = match file {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    let mut contents = String::new();
    match file.read_to_string(&mut contents) {
        Ok(_) => Ok(contents),
        Err(e) => Err(e),
    }
}

// With ? (equivalent, but cleaner)
fn read_file(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}
```

### Custom Error Types

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MatrixError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("index out of bounds: ({row}, {col}) in {rows}x{cols} matrix")]
    IndexOutOfBounds { row: usize, col: usize, rows: usize, cols: usize },

    #[error("matrix is singular")]
    SingularMatrix,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

fn solve_linear_system(a: &Matrix, b: &Vector) -> Result<Vector, MatrixError> {
    if a.rows != b.len() {
        return Err(MatrixError::DimensionMismatch {
            expected: a.rows,
            actual: b.len(),
        });
    }
    // ...
}
```

### Why No Exceptions?

| Aspect | Exceptions (C++) | Result (Rust) |
|--------|------------------|---------------|
| Control flow | Hidden, non-local | Explicit, local |
| Performance | Stack unwinding overhead | Zero-cost when no error |
| Checking | Easy to forget try/catch | Compiler forces handling |
| Error info | Runtime polymorphism | Compile-time known |

---

## 4. Module System

### Basic Modules

```rust
// In src/lib.rs or src/main.rs
mod math {
    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }

    fn helper() {  // Private by default
        // ...
    }

    pub mod linalg {
        pub struct Matrix { /* ... */ }
    }
}

use math::linalg::Matrix;
```

### File-Based Modules

```
src/
├── main.rs
├── math.rs          // or math/mod.rs
└── math/
    └── linalg.rs
```

**src/main.rs:**
```rust
mod math;

use math::linalg::Matrix;

fn main() {
    let m = Matrix::new(3, 3);
}
```

**src/math.rs:**
```rust
pub mod linalg;

pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

**src/math/linalg.rs:**
```rust
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }
}
```

### Visibility Rules

```rust
pub mod outer {
    pub struct Public;           // Visible everywhere
    struct Private;              // Only in this module
    pub(crate) struct CrateOnly; // Only in this crate
    pub(super) struct ParentOnly; // Only in parent module

    pub mod inner {
        pub(in crate::outer) struct OuterOnly; // Only in outer
    }
}
```

### Workspaces for Multi-Crate Projects

```
my_project/
├── Cargo.toml          # Workspace manifest
├── core/
│   ├── Cargo.toml      # Library crate
│   └── src/lib.rs
├── cli/
│   ├── Cargo.toml      # Binary crate
│   └── src/main.rs
└── tests/
    ├── Cargo.toml      # Integration tests
    └── src/lib.rs
```

**Workspace Cargo.toml:**
```toml
[workspace]
members = ["core", "cli", "tests"]

[workspace.dependencies]
nalgebra = "0.33"
```

**cli/Cargo.toml:**
```toml
[package]
name = "my_project_cli"
version = "0.1.0"
edition = "2024"

[dependencies]
my_project_core = { path = "../core" }
nalgebra = { workspace = true }
```

---

## 5. Standard Library Highlights

### Collections

```rust
use std::collections::{HashMap, HashSet, BTreeMap, VecDeque};

// Vec: dynamic array
let mut v = vec![1, 2, 3];
v.push(4);

// HashMap: hash table
let mut scores: HashMap<String, i32> = HashMap::new();
scores.insert("Alice".to_string(), 95);

// BTreeMap: sorted map (useful for ordered iteration)
let mut sorted: BTreeMap<i32, String> = BTreeMap::new();

// HashSet: unique values
let primes: HashSet<i32> = [2, 3, 5, 7, 11].into_iter().collect();
```

### Iterators

```rust
let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// Functional style (often faster than loops due to optimizations)
let sum_of_squares: i32 = data.iter()
    .filter(|&&x| x % 2 == 0)
    .map(|&x| x * x)
    .sum();

// Chaining transformations
let result: Vec<_> = data.iter()
    .enumerate()
    .filter(|(i, _)| i % 2 == 0)
    .map(|(_, &v)| v * 2)
    .take(3)
    .collect();
```

### Smart Pointers

```rust
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

// Box: heap allocation with single owner
let boxed: Box<[f64; 1000000]> = Box::new([0.0; 1000000]);

// Rc: reference counting (single-threaded)
let shared = Rc::new(vec![1, 2, 3]);
let clone = Rc::clone(&shared);

// RefCell: interior mutability (runtime borrow checking)
let data = RefCell::new(vec![1, 2, 3]);
data.borrow_mut().push(4);

// Arc + Mutex: thread-safe shared state
let counter = Arc::new(Mutex::new(0));
```

---

## Summary

| Concept | C++ | Rust |
|---------|-----|------|
| Sum types | `std::variant` | `enum` |
| Pattern matching | Limited `switch` | Powerful `match` |
| Polymorphism | Virtual functions | Traits |
| Generics | Templates (SFINAE) | Traits (bounds) |
| Errors | Exceptions | `Result<T, E>` |
| Namespaces | `namespace` | `mod` |

---

## Next Steps

- **Seminar 2.1**: [Traits and Generics](../seminars/seminar-02-1.md)
- **Seminar 2.2**: [Error Handling and Project Structure](../seminars/seminar-02-2.md)

## Additional Reading

- [The Rust Programming Language, Chapter 10 (Generics, Traits, Lifetimes)](https://doc.rust-lang.org/book/ch10-00-generics.html)
- [Rust by Example: Traits](https://doc.rust-lang.org/rust-by-example/trait.html)
