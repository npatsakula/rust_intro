---
sidebar_position: 2
title: "Seminar 1.2: Ownership Deep Dive"
---

# Seminar 1.2: Ownership Deep Dive

**Duration:** 90 minutes
**Prerequisites:** Lecture 1, Seminar 1.1

---

## Objectives

- Master ownership transfer and borrowing
- Understand why linked lists are hard in Rust
- Practice refactoring C++ pointer code to Rust
- Work with `String` vs `&str`, `Vec<T>` vs `&[T]`
- Practice lifetime annotations

---

## Task 1: Ownership Puzzles (20 min)

Predict whether each code snippet compiles. If not, explain why.

### Puzzle 1
```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;
    println!("{}", s1);
}
```

<details>
<summary>Solution</summary>

**Does not compile.** `s1` was moved to `s2`. After a move, the original variable is no longer valid.

```rust
// Fix: Clone the string
let s2 = s1.clone();
println!("{}", s1);  // Now OK
```
</details>

### Puzzle 2
```rust
fn main() {
    let x = 5;
    let y = x;
    println!("{}, {}", x, y);
}
```

<details>
<summary>Solution</summary>

**Compiles!** Integers implement `Copy` trait, so they are copied rather than moved.

Types that implement `Copy`: integers, floats, bools, chars, tuples of `Copy` types.
</details>

### Puzzle 3
```rust
fn take_string(s: String) {
    println!("{}", s);
}

fn main() {
    let s = String::from("hello");
    take_string(s);
    take_string(s);
}
```

<details>
<summary>Solution</summary>

**Does not compile.** First `take_string(s)` takes ownership. Second call tries to use moved value.

```rust
// Fix 1: Clone
take_string(s.clone());
take_string(s);

// Fix 2: Borrow
fn take_string(s: &String) {
    println!("{}", s);
}
take_string(&s);
take_string(&s);

// Fix 3: Return ownership
fn take_and_return(s: String) -> String {
    println!("{}", s);
    s
}
let s = take_and_return(s);
take_and_return(s);
```
</details>

### Puzzle 4
```rust
fn main() {
    let mut v = vec![1, 2, 3];
    let first = &v[0];
    v.push(4);
    println!("{}", first);
}
```

<details>
<summary>Solution</summary>

**Does not compile.** `first` is an immutable borrow. `v.push(4)` requires mutable borrow. Can't have both simultaneously.

Also: `push` might reallocate the vector, invalidating `first`.

```rust
// Fix: Don't use first after mutation
let mut v = vec![1, 2, 3];
let first = v[0];  // Copy the value, not borrow
v.push(4);
println!("{}", first);

// Or: Use first before mutating
let first = &v[0];
println!("{}", first);
v.push(4);  // Now OK, first is no longer used
```
</details>

### Puzzle 5
```rust
fn main() {
    let mut data = vec![1, 2, 3, 4, 5];

    for item in &data {
        if *item == 3 {
            data.push(6);
        }
    }
}
```

<details>
<summary>Solution</summary>

**Does not compile.** Cannot mutate `data` while iterating over it with immutable borrow.

```rust
// Fix: Collect indices first, then mutate
let indices: Vec<_> = data.iter()
    .enumerate()
    .filter(|(_, &item)| item == 3)
    .map(|(i, _)| i)
    .collect();

for _ in indices {
    data.push(6);
}
```
</details>

---

## Task 2: Why Linked Lists Are Hard (20 min)

### 2.1 The Naive Approach

Try to implement a simple linked list:

```rust
// This won't work!
struct Node {
    value: i32,
    next: Node,  // ERROR: recursive type has infinite size
}
```

### 2.2 Using Box

```rust
struct Node {
    value: i32,
    next: Option<Box<Node>>,  // Box provides heap allocation
}

struct LinkedList {
    head: Option<Box<Node>>,
}

impl LinkedList {
    fn new() -> Self {
        LinkedList { head: None }
    }

    fn push_front(&mut self, value: i32) {
        let new_node = Box::new(Node {
            value,
            next: self.head.take(),  // take() replaces with None and returns old value
        });
        self.head = Some(new_node);
    }

    fn pop_front(&mut self) -> Option<i32> {
        self.head.take().map(|node| {
            self.head = node.next;
            node.value
        })
    }
}

fn main() {
    let mut list = LinkedList::new();
    list.push_front(1);
    list.push_front(2);
    list.push_front(3);

    while let Some(value) = list.pop_front() {
        println!("{}", value);
    }
}
```

### 2.3 The Challenge: Multiple References

```rust
// What if we want both head and tail pointers?
struct LinkedList {
    head: Option<Box<Node>>,
    tail: ???,  // How do we point to the last node?
}
```

This is where Rust's ownership model becomes challenging:
- `Box` gives exclusive ownership
- We can't have `tail` also "own" the last node
- We need `Rc` (reference counting) or unsafe code

### 2.4 Discussion

Why is this important for scientific computing?
- Mesh data structures often need complex pointer relationships
- Solution: Use index-based approaches or arena allocation
- Libraries like `petgraph` handle this properly

---

## Task 3: Refactoring C++ Pointers to Rust (20 min)

### 3.1 C++ Code with Raw Pointers

```cpp
#include <iostream>

double* find_max(double* arr, int n) {
    if (n == 0) return nullptr;

    double* max_ptr = &arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > *max_ptr) {
            max_ptr = &arr[i];
        }
    }
    return max_ptr;
}

void double_values(double* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] *= 2.0;
    }
}

int main() {
    double data[] = {1.0, 5.0, 3.0, 9.0, 2.0};
    int n = 5;

    double* max = find_max(data, n);
    if (max) {
        std::cout << "Max: " << *max << std::endl;
    }

    double_values(data, n);

    for (int i = 0; i < n; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### 3.2 Rust Translation

```rust
fn find_max(arr: &[f64]) -> Option<&f64> {
    if arr.is_empty() {
        return None;
    }

    let mut max_ref = &arr[0];
    for value in arr.iter().skip(1) {
        if value > max_ref {
            max_ref = value;
        }
    }
    Some(max_ref)
}

fn double_values(arr: &mut [f64]) {
    for value in arr.iter_mut() {
        *value *= 2.0;
    }
}

fn main() {
    let mut data = [1.0, 5.0, 3.0, 9.0, 2.0];

    if let Some(max) = find_max(&data) {
        println!("Max: {}", max);
    }

    double_values(&mut data);

    for value in &data {
        print!("{} ", value);
    }
    println!();
}
```

### 3.3 Key Differences

| C++ | Rust |
|-----|------|
| `double* arr, int n` | `&[f64]` (slice includes length) |
| `nullptr` | `Option::None` |
| Raw pointer dereference | Automatic with references |
| No bounds checking | Bounds checked (can use `.get()` for unchecked) |
| Mutable by default | `&mut` explicitly required |

---

## Task 4: String Types (15 min)

### 4.1 String vs &str

```rust
fn main() {
    // String: owned, heap-allocated, growable
    let mut owned = String::from("Hello");
    owned.push_str(", world!");

    // &str: borrowed string slice
    let borrowed: &str = "Hello, world!";  // Points to static memory
    let slice: &str = &owned[0..5];        // Points into owned String

    // Converting
    let s: String = borrowed.to_string();
    let s: String = String::from(borrowed);
    let s: &str = &owned;  // Deref coercion
}
```

### 4.2 Function Parameters

```rust
// Prefer &str for input parameters (more flexible)
fn greet(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    let owned = String::from("Alice");
    let borrowed = "Bob";

    greet(&owned);    // &String coerces to &str
    greet(borrowed);  // &str works directly
}

// Return String when creating new data
fn create_greeting(name: &str) -> String {
    format!("Hello, {}!", name)
}
```

### 4.3 `Vec<T>` vs `&[T]`

Same pattern applies to vectors:

```rust
// Prefer slices for input parameters
fn sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

fn main() {
    let vec = vec![1.0, 2.0, 3.0];
    let array = [4.0, 5.0, 6.0];

    println!("{}", sum(&vec));    // &Vec<f64> coerces to &[f64]
    println!("{}", sum(&array));  // &[f64; 3] coerces to &[f64]
}
```

---

## Task 5: Lifetime Annotations (15 min)

### 5.1 When Lifetimes Are Needed

```rust
// The compiler can infer lifetimes here (elision rules)
fn first_word(s: &str) -> &str {
    match s.find(' ') {
        Some(i) => &s[0..i],
        None => s,
    }
}

// But not here: two input references, which one does output relate to?
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### 5.2 Struct with References

```rust
// Struct containing a reference needs lifetime annotation
struct Excerpt<'a> {
    text: &'a str,
}

impl<'a> Excerpt<'a> {
    fn new(text: &'a str) -> Self {
        Excerpt { text }
    }

    fn len(&self) -> usize {
        self.text.len()
    }
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().unwrap();

    let excerpt = Excerpt::new(first_sentence);
    println!("Excerpt: '{}' ({} chars)", excerpt.text, excerpt.len());
}
```

### 5.3 Practical Example: Matrix View

```rust
/// A view into a matrix row
struct RowView<'a> {
    data: &'a [f64],
}

impl<'a> RowView<'a> {
    fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    fn get(&self, index: usize) -> Option<f64> {
        self.data.get(index).copied()
    }
}

struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    fn row(&self, i: usize) -> RowView {
        let start = i * self.cols;
        let end = start + self.cols;
        RowView {
            data: &self.data[start..end],
        }
    }
}

fn main() {
    let mut matrix = Matrix::new(3, 4);
    // Fill with some values
    for (i, val) in matrix.data.iter_mut().enumerate() {
        *val = i as f64;
    }

    let row = matrix.row(1);
    println!("Row 1 sum: {}", row.sum());
}
```

---

## Homework

### Assignment: Matrix Type with Proper Borrowing

Create a `Matrix<T>` type that demonstrates ownership and borrowing concepts.

**Requirements:**

1. `Matrix<T>` struct with owned data (`Vec<T>`)
2. Implement:
   - `new(rows, cols)` — create zero-initialized matrix
   - `from_vec(rows, cols, data)` — create from existing data
   - `get(row, col) -> Option<&T>` — immutable access
   - `get_mut(row, col) -> Option<&mut T>` — mutable access
   - `row(i) -> &[T]` — slice of row
   - `iter()` — iterator over all elements
3. Implement `Index` and `IndexMut` traits for `matrix[(i, j)]` syntax
4. Write unit tests

**Starter code:**

```rust
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Default + Clone> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        todo!()
    }
}

impl<T> Matrix<T> {
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Option<Self> {
        todo!()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        todo!()
    }

    // ... more methods
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let m: Matrix<f64> = Matrix::new(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
    }

    // ... more tests
}
```

**Submission:** Push to course repository, branch `homework-1-2`.

---

## Summary

Today you:
- ✅ Solved ownership puzzles
- ✅ Understood why linked lists are challenging in Rust
- ✅ Refactored C++ pointer code to Rust
- ✅ Learned `String`/`&str` and `Vec<T>`/`&[T]` patterns
- ✅ Practiced lifetime annotations

## Next

Continue to [Lecture 2: Type System, Traits, and Modules](../lectures/lecture-02.md)
