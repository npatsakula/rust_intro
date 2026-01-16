---
sidebar_position: 6
title: "Lecture 6: Unsafe Rust and FFI"
---

# Lecture 6: Unsafe Rust and FFI

**Duration:** 90 minutes
**Block:** IV — Advanced Rust

---

## Learning Objectives

By the end of this lecture, you will:
- Understand when and how to use unsafe Rust
- Write sound unsafe code with proper invariants
- Interface with C/C++ libraries via FFI
- Optimize performance-critical code

---

## 1. When Safe Rust Isn't Enough

### Valid Use Cases

1. **FFI** — Calling C/C++/Fortran code
2. **Hardware access** — Memory-mapped I/O
3. **Performance** — Bypassing bounds checks in hot loops
4. **Data structures** — Intrusive containers, self-referential types

### The Five Unsafe Superpowers

```rust
unsafe {
    // 1. Dereference raw pointers
    let ptr: *const i32 = &42;
    let value = *ptr;

    // 2. Call unsafe functions
    dangerous_function();

    // 3. Access/modify mutable statics
    COUNTER += 1;

    // 4. Implement unsafe traits
    // impl UnsafeTrait for MyType { }

    // 5. Access fields of unions
    let u = MyUnion { i: 42 };
    let f = u.f;
}
```

---

## 2. Writing Sound Unsafe Code

### Safety Invariants

```rust
/// A simple unsafe abstraction with documented invariants
pub struct RawArray<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

impl<T> RawArray<T> {
    /// Creates a new array with given capacity
    ///
    /// # Safety Invariants (internal)
    /// - `ptr` is valid for reads/writes of `len` elements
    /// - `ptr` is valid for reads of `cap` elements
    /// - `ptr` is aligned for T
    /// - Elements 0..len are initialized
    pub fn with_capacity(cap: usize) -> Self {
        let layout = std::alloc::Layout::array::<T>(cap).unwrap();

        // SAFETY: layout is non-zero size (T has size), properly aligned
        let ptr = unsafe { std::alloc::alloc(layout) as *mut T };

        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        RawArray { ptr, len: 0, cap }
    }

    /// Push an element
    pub fn push(&mut self, value: T) {
        assert!(self.len < self.cap, "capacity exceeded");

        // SAFETY: self.len < self.cap, so ptr.add(len) is valid
        unsafe {
            self.ptr.add(self.len).write(value);
        }
        self.len += 1;
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY: index < len, element is initialized
            Some(unsafe { &*self.ptr.add(index) })
        } else {
            None
        }
    }
}

impl<T> Drop for RawArray<T> {
    fn drop(&mut self) {
        // SAFETY: Drop all initialized elements
        unsafe {
            std::ptr::drop_in_place(std::slice::from_raw_parts_mut(self.ptr, self.len));

            let layout = std::alloc::Layout::array::<T>(self.cap).unwrap();
            std::alloc::dealloc(self.ptr as *mut u8, layout);
        }
    }
}
```

### MaybeUninit for Delayed Initialization

```rust
use std::mem::MaybeUninit;

/// Initialize array without zeroing first
fn create_array() -> [f64; 1000] {
    let mut arr: [MaybeUninit<f64>; 1000] = unsafe {
        MaybeUninit::uninit().assume_init()
    };

    for (i, elem) in arr.iter_mut().enumerate() {
        elem.write(i as f64);
    }

    // SAFETY: All elements are now initialized
    unsafe {
        std::mem::transmute::<_, [f64; 1000]>(arr)
    }
}
```

### Common Pitfalls

```rust
// ❌ BAD: Aliasing mutable references
fn bad_aliasing() {
    let mut data = vec![1, 2, 3];
    let ptr1 = data.as_mut_ptr();
    let ptr2 = data.as_mut_ptr();

    unsafe {
        *ptr1 = 10;
        *ptr2 = 20;  // UB: two mutable references!
    }
}

// ❌ BAD: Use after free
fn bad_use_after_free() {
    let ptr = {
        let data = vec![1, 2, 3];
        data.as_ptr()
    };
    // data is dropped here!

    unsafe {
        println!("{}", *ptr);  // UB: dangling pointer!
    }
}

// ❌ BAD: Uninitialized memory
fn bad_uninit() {
    let mut x: i32;
    unsafe {
        // println!("{}", x);  // UB: reading uninitialized!
    }
}
```

---

## 3. FFI with C

### Basic FFI

```rust
// Declare external C function
extern "C" {
    fn abs(x: i32) -> i32;
    fn sqrt(x: f64) -> f64;
    fn printf(format: *const i8, ...) -> i32;
}

fn call_c_functions() {
    unsafe {
        let a = abs(-42);
        println!("abs(-42) = {}", a);

        let s = sqrt(2.0);
        println!("sqrt(2) = {}", s);
    }
}
```

### Structs with C Layout

```rust
/// Struct with C-compatible memory layout
#[repr(C)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[repr(C)]
pub struct Matrix3x3 {
    pub data: [[f64; 3]; 3],
}

extern "C" {
    fn process_point(p: *const Point) -> f64;
    fn transform_matrix(m: *mut Matrix3x3);
}
```

### Strings

```rust
use std::ffi::{CString, CStr};
use std::os::raw::c_char;

extern "C" {
    fn strlen(s: *const c_char) -> usize;
}

fn c_string_example() {
    // Rust String → C string
    let rust_string = "Hello, C!";
    let c_string = CString::new(rust_string).unwrap();

    unsafe {
        let len = strlen(c_string.as_ptr());
        println!("strlen = {}", len);
    }
}

// C string → Rust string
fn from_c_string(ptr: *const c_char) -> String {
    unsafe {
        CStr::from_ptr(ptr)
            .to_string_lossy()
            .into_owned()
    }
}
```

### Callbacks

```rust
type Callback = extern "C" fn(f64) -> f64;

extern "C" {
    fn integrate(f: Callback, a: f64, b: f64, n: i32) -> f64;
}

extern "C" fn my_function(x: f64) -> f64 {
    x * x
}

fn use_callback() {
    unsafe {
        let result = integrate(my_function, 0.0, 1.0, 1000);
        println!("∫x² dx from 0 to 1 = {}", result);
    }
}
```

---

## 4. Using bindgen

### Generate Bindings Automatically

```toml
# Cargo.toml
[build-dependencies]
bindgen = "0.70"
```

```rust
// build.rs
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-lib=mylib");
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
```

```rust
// src/lib.rs
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
```

---

## 5. Calling BLAS/LAPACK

### Direct FFI

```rust
extern "C" {
    // BLAS dgemm: C = α*A*B + β*C
    fn dgemm_(
        transa: *const i8,
        transb: *const i8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
}

fn matrix_multiply_blas(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) {
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k_i32 = k as i32;
    let alpha = 1.0;
    let beta = 0.0;

    unsafe {
        dgemm_(
            b"N\0".as_ptr() as *const i8,  // No transpose A
            b"N\0".as_ptr() as *const i8,  // No transpose B
            &m_i32,
            &n_i32,
            &k_i32,
            &alpha,
            a.as_ptr(),
            &m_i32,
            b.as_ptr(),
            &k_i32,
            &beta,
            c.as_mut_ptr(),
            &m_i32,
        );
    }
}
```

---

## 6. Performance Optimization

### SIMD

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Sum array using AVX
#[cfg(target_arch = "x86_64")]
unsafe fn sum_avx(data: &[f64]) -> f64 {
    let mut sum = _mm256_setzero_pd();
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let v = _mm256_loadu_pd(chunk.as_ptr());
        sum = _mm256_add_pd(sum, v);
    }

    // Horizontal sum
    let mut result = [0.0f64; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), sum);
    let total: f64 = result.iter().sum();

    // Add remainder
    total + remainder.iter().sum::<f64>()
}
```

### Unsafe for Performance

```rust
/// Fast matrix access without bounds checking
pub struct FastMatrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl FastMatrix {
    /// Get element without bounds checking
    ///
    /// # Safety
    /// - `row` must be < `self.rows`
    /// - `col` must be < `self.cols`
    #[inline]
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> f64 {
        *self.data.get_unchecked(row * self.cols + col)
    }

    /// Matrix multiply (optimized)
    pub fn multiply(&self, other: &FastMatrix) -> FastMatrix {
        assert_eq!(self.cols, other.rows);

        let mut result = FastMatrix {
            data: vec![0.0; self.rows * other.cols],
            rows: self.rows,
            cols: other.cols,
        };

        // SAFETY: All indices are within bounds due to loop structure
        unsafe {
            for i in 0..self.rows {
                for k in 0..self.cols {
                    let a_ik = self.get_unchecked(i, k);
                    for j in 0..other.cols {
                        let idx = i * other.cols + j;
                        *result.data.get_unchecked_mut(idx) +=
                            a_ik * other.get_unchecked(k, j);
                    }
                }
            }
        }

        result
    }
}
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| Unsafe blocks | Five superpowers, use sparingly |
| Invariants | Document, maintain, verify |
| FFI | `extern "C"`, `#[repr(C)]`, bindgen |
| BLAS/LAPACK | Fortran calling convention |
| SIMD | `std::arch`, platform-specific |

---

## Next Steps

- **Seminar 6.1**: [Unsafe Code Patterns](../seminars/seminar-06-1.md)
- **Seminar 6.2**: [C/C++ Integration](../seminars/seminar-06-2.md)
