---
sidebar_position: 11
title: "Seminar 6.1: Unsafe Code Patterns"
---

# Seminar 6.1: Unsafe Code Patterns

**Duration:** 90 minutes
**Prerequisites:** Lecture 6

---

## Objectives

- Implement unsafe Vec from scratch
- Use miri to detect undefined behavior
- Benchmark safe vs unsafe code
- Implement arena allocator

---

## Task 1: Implementing Vec (30 min)

```rust
use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};
use std::mem;

pub struct MyVec<T> {
    ptr: NonNull<T>,
    cap: usize,
    len: usize,
}

impl<T> MyVec<T> {
    pub fn new() -> Self {
        // Don't allocate until first push
        MyVec {
            ptr: NonNull::dangling(),
            cap: 0,
            len: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        if cap == 0 {
            return Self::new();
        }

        let layout = Layout::array::<T>(cap).unwrap();

        // SAFETY: layout has non-zero size
        let ptr = unsafe { alloc::alloc(layout) as *mut T };

        let ptr = match NonNull::new(ptr) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout),
        };

        MyVec { ptr, cap, len: 0 }
    }

    fn grow(&mut self) {
        let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };

        let new_layout = Layout::array::<T>(new_cap).unwrap();

        let new_ptr = if self.cap == 0 {
            // SAFETY: new_layout has non-zero size
            unsafe { alloc::alloc(new_layout) }
        } else {
            let old_layout = Layout::array::<T>(self.cap).unwrap();
            // SAFETY: ptr was allocated with old_layout
            unsafe {
                alloc::realloc(
                    self.ptr.as_ptr() as *mut u8,
                    old_layout,
                    new_layout.size(),
                )
            }
        };

        self.ptr = match NonNull::new(new_ptr as *mut T) {
            Some(p) => p,
            None => alloc::handle_alloc_error(new_layout),
        };
        self.cap = new_cap;
    }

    pub fn push(&mut self, value: T) {
        if self.len == self.cap {
            self.grow();
        }

        // SAFETY: len < cap after grow, ptr + len is valid
        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.len), value);
        }
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            // SAFETY: len was > 0, element at len is initialized
            unsafe {
                Some(ptr::read(self.ptr.as_ptr().add(self.len)))
            }
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY: index < len, element is initialized
            Some(unsafe { &*self.ptr.as_ptr().add(index) })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            // SAFETY: index < len, element is initialized
            Some(unsafe { &mut *self.ptr.as_ptr().add(index) })
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }
}

impl<T> Drop for MyVec<T> {
    fn drop(&mut self) {
        if self.cap != 0 {
            // SAFETY: Drop all initialized elements
            unsafe {
                for i in 0..self.len {
                    ptr::drop_in_place(self.ptr.as_ptr().add(i));
                }

                let layout = Layout::array::<T>(self.cap).unwrap();
                alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let mut v = MyVec::new();
        v.push(1);
        v.push(2);
        v.push(3);

        assert_eq!(v.len(), 3);
        assert_eq!(v.pop(), Some(3));
        assert_eq!(v.pop(), Some(2));
        assert_eq!(v.pop(), Some(1));
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_get() {
        let mut v = MyVec::new();
        v.push("hello");
        v.push("world");

        assert_eq!(v.get(0), Some(&"hello"));
        assert_eq!(v.get(1), Some(&"world"));
        assert_eq!(v.get(2), None);
    }

    #[test]
    fn test_drops() {
        use std::rc::Rc;
        use std::cell::Cell;

        let counter = Rc::new(Cell::new(0));

        struct DropCounter(Rc<Cell<i32>>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        {
            let mut v = MyVec::new();
            v.push(DropCounter(counter.clone()));
            v.push(DropCounter(counter.clone()));
            v.push(DropCounter(counter.clone()));
        }

        assert_eq!(counter.get(), 3);
    }
}
```

---

## Task 2: Finding UB with Miri (20 min)

### Install and Run Miri

```bash
rustup +nightly component add miri
cargo +nightly miri test
cargo +nightly miri run
```

### Buggy Code Examples

```rust
/// Contains undefined behavior - find it!
mod buggy {
    pub fn use_after_free() -> i32 {
        let ptr = {
            let x = Box::new(42);
            &*x as *const i32
        };
        // Box dropped, ptr is dangling!
        unsafe { *ptr }
    }

    pub fn out_of_bounds() {
        let arr = [1, 2, 3];
        let ptr = arr.as_ptr();
        unsafe {
            let _ = *ptr.add(5);  // Out of bounds!
        }
    }

    pub fn unaligned_access() {
        let data: [u8; 8] = [0; 8];
        let ptr = data.as_ptr().wrapping_add(1) as *const u32;
        unsafe {
            let _ = *ptr;  // Unaligned read!
        }
    }

    pub fn aliasing_violation() {
        let mut x = 42;
        let r1 = &mut x as *mut i32;
        let r2 = &mut x as *mut i32;
        unsafe {
            *r1 = 1;
            *r2 = 2;  // Two mutable refs!
        }
    }
}

#[test]
fn test_buggy_code() {
    // Uncomment to test with miri
    // buggy::use_after_free();
    // buggy::out_of_bounds();
    // buggy::unaligned_access();
    // buggy::aliasing_violation();
}
```

---

## Task 3: Safe vs Unsafe Benchmarks (20 min)

```rust
use std::time::Instant;

fn sum_safe(data: &[f64]) -> f64 {
    data.iter().sum()
}

fn sum_unsafe(data: &[f64]) -> f64 {
    let mut total = 0.0;
    let ptr = data.as_ptr();
    let len = data.len();

    // SAFETY: We iterate exactly within bounds
    unsafe {
        for i in 0..len {
            total += *ptr.add(i);
        }
    }
    total
}

fn matrix_multiply_safe(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn matrix_multiply_unsafe(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    // SAFETY: All indices are computed to be within bounds
    unsafe {
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += *a_ptr.add(i * n + k) * *b_ptr.add(k * n + j);
                }
                *c_ptr.add(i * n + j) = sum;
            }
        }
    }
    c
}

fn benchmark() {
    let n = 1_000_000;
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // Sum benchmark
    let start = Instant::now();
    let _ = sum_safe(&data);
    let safe_time = start.elapsed();

    let start = Instant::now();
    let _ = sum_unsafe(&data);
    let unsafe_time = start.elapsed();

    println!("Sum ({} elements):", n);
    println!("  Safe:   {:?}", safe_time);
    println!("  Unsafe: {:?}", unsafe_time);
    println!("  Ratio:  {:.2}x", safe_time.as_nanos() as f64 / unsafe_time.as_nanos() as f64);

    // Matrix multiply benchmark
    let n = 200;
    let a: Vec<f64> = (0..n*n).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..n*n).map(|i| (i * 2) as f64).collect();

    let start = Instant::now();
    let _ = matrix_multiply_safe(&a, &b, n);
    let safe_time = start.elapsed();

    let start = Instant::now();
    let _ = matrix_multiply_unsafe(&a, &b, n);
    let unsafe_time = start.elapsed();

    println!("\nMatrix multiply ({}x{}):", n, n);
    println!("  Safe:   {:?}", safe_time);
    println!("  Unsafe: {:?}", unsafe_time);
    println!("  Ratio:  {:.2}x", safe_time.as_nanos() as f64 / unsafe_time.as_nanos() as f64);
}
```

---

## Task 4: Arena Allocator (20 min)

```rust
use std::cell::Cell;
use std::mem;

/// Simple bump allocator for homogeneous types
pub struct Arena<T> {
    chunks: Vec<Vec<T>>,
    current: Cell<usize>,
}

impl<T> Arena<T> {
    const CHUNK_SIZE: usize = 1024;

    pub fn new() -> Self {
        Arena {
            chunks: vec![Vec::with_capacity(Self::CHUNK_SIZE)],
            current: Cell::new(0),
        }
    }

    pub fn alloc(&self, value: T) -> &T {
        let current = self.current.get();
        let chunk = self.chunks.last().unwrap();

        if current < chunk.capacity() {
            // SAFETY: We have exclusive access through &self (interior mutability)
            // and current < capacity
            unsafe {
                let ptr = chunk.as_ptr().add(current) as *mut T;
                ptr.write(value);
                self.current.set(current + 1);
                &*ptr
            }
        } else {
            // Need to allocate new chunk
            // This requires &mut self in real implementation
            panic!("Arena full - would need new chunk");
        }
    }
}

/// Thread-safe arena using atomic operations
pub struct SyncArena<T> {
    chunks: Vec<Vec<T>>,
    current: std::sync::atomic::AtomicUsize,
}

// Usage for FEM nodes
#[derive(Debug)]
struct FEMNode {
    id: usize,
    x: f64,
    y: f64,
    z: f64,
}

fn arena_fem_example() {
    let arena = Arena::new();

    // Allocate nodes - all live for arena's lifetime
    let nodes: Vec<&FEMNode> = (0..100)
        .map(|i| {
            arena.alloc(FEMNode {
                id: i,
                x: i as f64 * 0.1,
                y: 0.0,
                z: 0.0,
            })
        })
        .collect();

    // All references valid
    for node in &nodes[..5] {
        println!("Node {}: ({}, {}, {})", node.id, node.x, node.y, node.z);
    }

    // No individual deallocation - all freed when arena drops
}
```

---

## Homework

### Assignment: Intrusive Linked List

Implement an intrusive doubly-linked list for FEM element connectivity:

```rust
pub struct IntrusiveList<T> {
    head: *mut Node<T>,
    tail: *mut Node<T>,
    len: usize,
}

pub struct Node<T> {
    prev: *mut Node<T>,
    next: *mut Node<T>,
    value: T,
}

impl<T> IntrusiveList<T> {
    pub fn new() -> Self;
    pub fn push_front(&mut self, node: Box<Node<T>>);
    pub fn push_back(&mut self, node: Box<Node<T>>);
    pub fn remove(&mut self, node: *mut Node<T>) -> Option<Box<Node<T>>>;
    pub fn iter(&self) -> Iter<T>;
}
```

**Requirements:**
- No memory leaks (test with miri)
- Support removal from middle in O(1)
- Safe public API

**Submission:** Branch `homework-6-1`

---

## Summary

- ✅ Implemented Vec from scratch with unsafe
- ✅ Used miri to detect undefined behavior
- ✅ Benchmarked safe vs unsafe performance
- ✅ Built arena allocator

## Next

Continue to [Seminar 6.2: C/C++ Integration](./seminar-06-2.md)
