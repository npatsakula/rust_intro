---
sidebar_position: 1
title: "Lecture 1: Ecosystem and Ownership Model"
---

# Lecture 1: Why Rust? (And What's Wrong with C++)

**Duration:** 90 minutes
**Block:** I — From C++ to Rust

---

You know C++. You've written numerical code, dealt with templates, fought with CMake. You understand RAII and smart pointers. So why should you care about yet another systems language?

Let me be direct: Rust isn't just "C++ but safer." It's a fundamentally different approach to systems programming — one that eliminates entire categories of bugs at compile time while giving you the same (sometimes better) performance.

But before we dive into Rust, let's talk honestly about C++. Not to bash it — we all use it and will continue to use it — but to understand _why_ Rust exists and what problems it actually solves.

---

## The C++ Complexity Tax

### The Constructor Madness

Quick quiz: How many constructors can a single C++ class have?

```cpp
class Matrix {
public:
    // 1. Default constructor
    Matrix();

    // 2. Parameterized constructor
    Matrix(size_t rows, size_t cols);

    // 3. Copy constructor
    Matrix(const Matrix& other);

    // 4. Move constructor
    Matrix(Matrix&& other) noexcept;

    // 5. Initializer list constructor
    Matrix(std::initializer_list<std::initializer_list<double>> init);

    // 6. Converting constructor (implicit!)
    Matrix(const std::vector<std::vector<double>>& data);

    // 7. Another converting constructor
    Matrix(double scalar);  // 1x1 matrix? Or fill value?

    // Copy assignment
    Matrix& operator=(const Matrix& other);

    // Move assignment
    Matrix& operator=(Matrix&& other) noexcept;

    // Destructor
    ~Matrix();
};
```

That's **7 constructors**, **2 assignment operators**, and a **destructor**. The Rule of Zero, Rule of Three, Rule of Five — we need _rules_ just to remember what we need to write.

And it gets worse. Each of those constructors can be:

- `explicit` or implicit
- `constexpr` or not
- `noexcept` or throwing
- Deleted or defaulted

Now multiply this by inheritance. By templates. By SFINAE to disable certain overloads. Suddenly you're writing 200 lines of boilerplate for a simple matrix class.

**In Rust?**

```rust
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
}
```

That's it. No copy constructor — you implement `Clone` trait if you want copying. No move constructor — _everything_ moves by default. No destructor — `Drop` trait is there if you need custom cleanup, but usually you don't.

### The `auto` Return Type Sin

C++14 gave us `auto` return type deduction. It seemed like a good idea at the time:

```cpp
auto computeSomething(const Matrix& m) {
    if (m.rows() == 1) {
        return m.row(0);  // Returns... what exactly?
    }
    return m;  // Wait, is this even the same type?
}
```

This doesn't compile, of course. But the problem is deeper. Consider this real code:

```cpp
auto getElements() {
    std::vector<int> v = {1, 2, 3};
    return v | std::views::transform([](int x) { return x * 2; });
}

int main() {
    auto result = getElements();
    for (int x : result) {  // UNDEFINED BEHAVIOR
        std::cout << x << "\n";
    }
}
```

What's the return type of `getElements()`? It's something like:

```cpp
std::ranges::transform_view<std::ranges::ref_view<std::vector<int>>, lambda>
```

But here's the killer: **this is a view into a local variable**. The vector is destroyed when the function returns. The code compiles, runs, and might even produce correct output... sometimes. Welcome to undefined behavior.

The compiler didn't warn us. The type system didn't save us. We just created a dangling reference with modern, "safe" C++20 features.

**In Rust**, this is a compile-time error:

```rust
fn get_elements() -> impl Iterator<Item = i32> {
    let v = vec![1, 2, 3];
    v.into_iter().map(|x| x * 2)  // into_iter() takes ownership — no dangling!
}

// If you tried to return a reference to local data:
fn broken() -> &[i32] {
    let v = vec![1, 2, 3];
    &v  // ERROR: cannot return reference to local variable
}
```

The ownership system makes it _impossible_ to write the buggy version.

### The Diamond of Death

You probably learned to avoid diamond inheritance. But did you know C++ still doesn't really solve it?

```cpp
class Sensor {
public:
    virtual void calibrate() = 0;
    virtual double read() = 0;
};

class TemperatureSensor : public virtual Sensor {
public:
    void calibrate() override { /* ... */ }
    double read() override { return temperature_; }
private:
    double temperature_;
};

class PressureSensor : public virtual Sensor {
public:
    void calibrate() override { /* ... */ }
    double read() override { return pressure_; }
private:
    double pressure_;
};

// The dreaded diamond
class CombinedSensor : public TemperatureSensor, public PressureSensor {
public:
    // Which calibrate()? Which read()?
    void calibrate() override { /* must override */ }
    double read() override { /* must override */ }
};
```

Virtual inheritance "solves" this with runtime overhead and subtle gotchas. But the fundamental problem remains: **C++ conflates interface and implementation inheritance**.

**Rust separates them completely:**

```rust
trait Sensor {
    fn calibrate(&mut self);
    fn read(&self) -> f64;
}

struct TemperatureSensor { /* ... */ }
struct PressureSensor { /* ... */ }

impl Sensor for TemperatureSensor { /* ... */ }
impl Sensor for PressureSensor { /* ... */ }

// Need both? Compose, don't inherit:
struct CombinedSensor {
    temp: TemperatureSensor,
    pressure: PressureSensor,
}

impl CombinedSensor {
    fn read_temperature(&self) -> f64 {
        self.temp.read()
    }

    fn read_pressure(&self) -> f64 {
        self.pressure.read()
    }
}
```

No diamonds. No virtual inheritance overhead. No ambiguity. Just composition.

### Ranges: Promise vs Reality

C++20 ranges were supposed to revolutionize how we write code. The reality is... complicated.

```cpp
std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// This looks nice:
auto result = numbers
    | std::views::filter([](int n) { return n % 2 == 0; })
    | std::views::transform([](int n) { return n * n; });

// But what's the performance? Let's check the assembly...
```

The generated code often contains virtual function calls through `std::function`-like type erasure, branch mispredictions from the lazy evaluation, and suboptimal memory access patterns.

And then there's the UB potential we saw earlier — views don't own their data, and C++ won't stop you from letting that data die.

Even worse, error messages are incomprehensible:

```cpp
auto result = numbers | std::views::filter([](std::string s) { return s.empty(); });
// Error: no match for 'operator|' (operand types are 'std::vector<int>' and ...)
// ... followed by 200 lines of template instantiation errors
```

**Rust iterators** are lazy too, but:

- They're zero-cost (no type erasure)
- The ownership system prevents dangling
- Error messages are readable

```rust
let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

let result: Vec<i32> = numbers.iter()
    .filter(|&&n| n % 2 == 0)
    .map(|&n| n * n)
    .collect();

// Type error:
let bad = numbers.iter().filter(|s: &String| s.is_empty());
// Error: expected `&&i32`, found `&String`
// Clear. Actionable. Human-readable.
```

---

## What C++ Still Doesn't Have (in 2025)

### Proper Strings

After 40+ years, C++ still doesn't have a good string story:

```cpp
std::string s = "Hello, World!";

// Is this O(1) or O(n)?
size_t len = s.length();  // O(1), counts bytes

// What about Unicode?
std::string emoji = "🦀";
std::cout << emoji.length();  // Prints 4 (bytes), not 1 (character)

// String concatenation allocates:
std::string result = s + " " + emoji;  // Two allocations!

// No string interpolation:
int x = 42;
std::string formatted = "Value: " + std::to_string(x);  // Awkward
// Or: std::format("Value: {}", x);  // C++20, finally
```

C++ strings are "just bytes." Unicode support is an afterthought. String views help with performance but add lifetime complexity.

**Rust strings are UTF-8 by design:**

```rust
let s = "Hello, World!";
let emoji = "🦀";

// Clear distinction:
println!("{}", s.len());           // 13 bytes
println!("{}", s.chars().count()); // 13 characters
println!("{}", emoji.len());       // 4 bytes
println!("{}", emoji.chars().count()); // 1 character

// String interpolation built-in:
let x = 42;
let formatted = format!("Value: {x}");  // Clean and efficient

// Concatenation that's obvious about allocation:
let result = format!("{s} {emoji}");
```

### Sum Types (Algebraic Data Types)

In many languages, you can express "this value is either A or B":

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}

enum Option<T> {
    Some(T),
    None,
}

// Custom sum types:
enum Measurement {
    Temperature(f64),
    Pressure(f64),
    Humidity(f64),
    Missing { sensor_id: u32, reason: String },
}
```

C++17 gave us `std::variant`:

```cpp
std::variant<double, std::string, std::monostate> value;

// Setting a value:
value = 3.14;

// Getting it back:
if (std::holds_alternative<double>(value)) {
    double d = std::get<double>(value);
}

// Or with visit (verbose):
std::visit([](auto&& arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, double>) {
        std::cout << "double: " << arg << "\n";
    } else if constexpr (std::is_same_v<T, std::string>) {
        std::cout << "string: " << arg << "\n";
    } else {
        std::cout << "empty\n";
    }
}, value);
```

Compare to Rust's pattern matching:

```rust
match measurement {
    Measurement::Temperature(t) => println!("Temp: {t}°C"),
    Measurement::Pressure(p) => println!("Pressure: {p} Pa"),
    Measurement::Humidity(h) => println!("Humidity: {h}%"),
    Measurement::Missing { sensor_id, reason } => {
        println!("Sensor {sensor_id} failed: {reason}");
    }
}
```

The compiler ensures you handle all cases. Forget one? Compile error. C++ `std::visit` doesn't have this guarantee.

### Structured Concurrency

What if you want to parallelize a loop? In scientific computing, this is bread and butter — process each matrix row in parallel, run simulations concurrently, parallelize Monte Carlo sampling.

In C++, you reach for OpenMP:

```cpp
#include <omp.h>

void parallel_process(std::vector<double>& data) {
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = expensive_computation(data[i]);
    }
}
```

This works, but:

- It's a compiler extension, not part of the language
- Data race protection is your responsibility
- Composability is limited (nested parallelism is tricky)
- Error handling? What error handling?

C++17 added parallel algorithms:

```cpp
std::transform(std::execution::par, data.begin(), data.end(),
               data.begin(), expensive_computation);
```

Better, but still no structured concurrency. What's structured concurrency? The idea that **parallel work is scoped** — all spawned tasks complete before the scope exits, and resources are automatically managed.

**Rust has Rayon**, which makes parallelism trivial:

```rust
use rayon::prelude::*;

fn parallel_process(data: &mut [f64]) {
    data.par_iter_mut()
        .for_each(|x| *x = expensive_computation(*x));
}

// Or with map:
fn parallel_squares(data: &[f64]) -> Vec<f64> {
    data.par_iter()
        .map(|&x| x * x)
        .collect()
}

// Filter + map + reduce, all parallel:
fn parallel_pipeline(data: &[f64]) -> f64 {
    data.par_iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x.sqrt())
        .sum()
}
```

Change `.iter()` to `.par_iter()` — that's it. The work is automatically distributed across CPU cores, and **the parallel scope completes before the function returns**. No manual thread management, no join handles to forget, no data races possible (the type system prevents them).

Need more control? Rayon supports explicit scopes:

```rust
use rayon::scope;

fn custom_parallel_work(matrices: &[Matrix]) -> Vec<Result> {
    let mut results = vec![None; matrices.len()];

    rayon::scope(|s| {
        for (i, matrix) in matrices.iter().enumerate() {
            let result_slot = &mut results[i];
            s.spawn(move |_| {
                *result_slot = Some(process_matrix(matrix));
            });
        }
    });  // All spawned tasks complete here

    results.into_iter().map(|r| r.unwrap()).collect()
}
```

#### What About Thread Safety Bugs?

C++ lets you create data races easily:

```cpp
void data_race() {
    std::vector<int> data = {1, 2, 3};

    std::thread t1([&data]() { data.push_back(4); });
    std::thread t2([&data]() { data.push_back(5); });

    t1.join();
    t2.join();
    // Undefined behavior — concurrent mutation
}
```

**Rust catches this at compile time:**

```rust
fn wont_compile() {
    let mut data = vec![1, 2, 3];

    std::thread::scope(|s| {
        s.spawn(|| data.push(4));  // ERROR: cannot borrow `data` as mutable
        s.spawn(|| data.push(5));  // more than once at a time
    });
}
```

If you need shared mutable state, you must be explicit:

```rust
use std::sync::Mutex;

fn explicit_sharing() {
    let data = Mutex::new(vec![1, 2, 3]);

    std::thread::scope(|s| {
        s.spawn(|| data.lock().unwrap().push(4));
        s.spawn(|| data.lock().unwrap().push(5));
    });
    // Works — Mutex provides synchronized access
}
```

The type system makes concurrency bugs compile-time errors, not runtime mysteries.

---

## The Rust Ecosystem

Enough complaining about C++. Let's see what Rust actually offers.

### Cargo: Build System That Just Works

Remember spending days configuring CMake? Fighting with vcpkg? Manually downloading dependencies?

```bash
# Create a new project
cargo new my_simulation
cd my_simulation

# Add dependencies (that's it, really)
cargo add nalgebra ndarray rayon

# Build
cargo build

# Run
cargo run

# Test
cargo test

# Build optimized
cargo build --release
```

Your `Cargo.toml`:

```toml
[package]
name = "my_simulation"
version = "0.1.0"
edition = "2021"

[dependencies]
nalgebra = "0.33"
ndarray = "0.16"
rayon = "1.10"

[dev-dependencies]
criterion = "0.5"
proptest = "1.5"
```

Compare that to a typical CMake setup:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_simulation)

set(CMAKE_CXX_STANDARD 20)

# Hope you have these installed correctly...
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Or maybe you're using FetchContent?
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

add_executable(my_simulation src/main.cpp)
target_link_libraries(my_simulation PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_options(my_simulation PRIVATE -Wall -Wextra -O3)
# ... 50 more lines for tests, benchmarks, install rules...
```

### The Toolchain

| Tool            | What it does                  |
| --------------- | ----------------------------- |
| `cargo build`   | Compiles your project         |
| `cargo test`    | Runs tests                    |
| `cargo doc`     | Generates documentation       |
| `cargo fmt`     | Formats code (one true style) |
| `cargo clippy`  | Lints for common mistakes     |
| `cargo bench`   | Runs benchmarks               |
| `cargo publish` | Publishes to crates.io        |

All built-in. All work together. No configuration needed.

### Package Ecosystem

- **[crates.io](https://crates.io)** — 150,000+ packages
- **[docs.rs](https://docs.rs)** — Documentation for every package, auto-generated
- **[lib.rs](https://lib.rs)** — Better search and categorization

For scientific computing specifically:

- `nalgebra` — Linear algebra (like Eigen)
- `ndarray` — N-dimensional arrays (like NumPy)
- `russell` — Tensors and ODE solvers
- `faer` — High-performance linear algebra
- `polars` — DataFrames (like Pandas, but faster)
- `linfa` — Machine learning

---

## The Ownership Model

Now let's understand how Rust actually prevents the bugs we discussed.

### Three Simple Rules

1. **Each value has exactly one owner**
2. **When the owner goes out of scope, the value is dropped**
3. **Ownership can be transferred (moved) or borrowed**

That's it. These three rules eliminate use-after-free, double-free, data races, and null pointer bugs.

### Rule 1: One Owner

```rust
fn main() {
    let data = vec![1, 2, 3];  // data owns the vector

    let also_data = data;  // ownership MOVED to also_data

    // println!("{:?}", data);  // ERROR: data no longer valid
    println!("{:?}", also_data);  // OK
}
```

This is different from C++:

```cpp
std::vector<int> data = {1, 2, 3};
std::vector<int> also_data = std::move(data);
// data is now in "valid but unspecified state"
// C++ lets you use it anyway — Rust doesn't
```

### Rule 2: Automatic Cleanup

```rust
fn process() {
    let data = vec![1, 2, 3];  // memory allocated
    // ... use data ...
}  // data dropped here, memory freed

fn main() {
    process();
    // No memory leak possible
}
```

### Rule 3: Borrowing

Moving isn't always what you want. Sometimes you just need temporary access:

```rust
fn sum(data: &[i32]) -> i32 {  // borrows a slice
    data.iter().sum()
}

fn main() {
    let numbers = vec![1, 2, 3, 4, 5];

    let total = sum(&numbers);  // borrow the data

    println!("Numbers: {:?}", numbers);  // still valid!
    println!("Sum: {}", total);
}
```

### Mutable Borrowing

Need to modify borrowed data? Use `&mut`:

```rust
fn add_element(data: &mut Vec<i32>, value: i32) {
    data.push(value);
}

fn main() {
    let mut numbers = vec![1, 2, 3];
    add_element(&mut numbers, 4);
    println!("{:?}", numbers);  // [1, 2, 3, 4]
}
```

### The Key Invariant

At any time, you can have:

- **Many** immutable references (`&T`), OR
- **One** mutable reference (`&mut T`)

Never both. This prevents data races _at compile time_.

```rust
fn main() {
    let mut data = vec![1, 2, 3];

    let r1 = &data;     // OK
    let r2 = &data;     // OK — multiple readers
    println!("{:?} {:?}", r1, r2);

    let r3 = &mut data; // OK — r1, r2 no longer used
    r3.push(4);

    // let r4 = &data;  // ERROR if used while r3 exists
}
```

---

## Lifetimes: Making References Safe

C++ trusts you not to create dangling pointers. Rust verifies it.

### The Problem

```cpp
int* bad_pointer() {
    int x = 42;
    return &x;  // Compiles! Undefined behavior.
}
```

### Rust's Solution

```rust
fn bad_reference() -> &i32 {
    let x = 42;
    &x  // ERROR: cannot return reference to local variable
}

// The fix: return owned data
fn good_value() -> i32 {
    let x = 42;
    x  // Return the value itself
}
```

### Lifetime Annotations

Sometimes the compiler needs help understanding how long references live:

```rust
// Says: the returned reference lives as long as BOTH inputs
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

The `'a` is a _lifetime parameter_. It's not something you write often — the compiler infers lifetimes in most cases. But when it can't, you make the relationships explicit.

---

## Summary

C++ is powerful, but its complexity has grown beyond what any single person can master. Every new feature adds edge cases, every edge case adds bugs.

Rust takes a different approach:

- **Ownership** instead of manual memory management
- **Borrowing** with compile-time checking instead of hoping you got it right
- **Sum types** that force you to handle all cases
- **Traits** (covered in [Lecture 2](./lecture-02.md)) instead of inheritance hierarchies
- **One build system** that just works

The learning curve is real. The compiler will fight you at first. But once you internalize the ownership model, you'll find yourself writing correct code faster than you did in C++.

In [Lecture 2](./lecture-02.md), we'll build on this foundation to explore Rust's powerful type system, traits for polymorphism, and how to organize large scientific codebases. The ownership you learned here is the foundation—traits and generics are the flexible abstractions that make it practical for real-world applications.

---

## Next Steps

- **Seminar 1.1**: [Environment Setup and First Programs](../seminars/seminar-01-1.md)
- **Seminar 1.2**: [Ownership Deep Dive](../seminars/seminar-01-2.md)

## Reading

- [The Rust Programming Language, Chapter 4: Understanding Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
- [Rust by Example: Ownership and Moves](https://doc.rust-lang.org/rust-by-example/scope/move.html)
