---
sidebar_position: 1
title: "Seminar 1.1: Environment Setup"
---

# Seminar 1.1: Environment Setup and First Programs

**Duration:** 90 minutes
**Prerequisites:** Lecture 1

---

## Objectives

- Install and configure the Rust toolchain
- Set up IDE with rust-analyzer
- Create your first Rust project with Cargo
- Write a CLI tool with external dependencies
- Practice debugging Rust code

---

## Task 1: Install Rust Toolchain (15 min)

### 1.1 Install rustup

**Linux/macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows:**
Download and run [rustup-init.exe](https://rustup.rs)

### 1.2 Verify Installation

```bash
rustc --version
# rustc 1.XX.0 (...)

cargo --version
# cargo 1.XX.0 (...)

rustup --version
# rustup 1.XX.0 (...)
```

### 1.3 Install Additional Components

```bash
# Formatter
rustup component add rustfmt

# Linter
rustup component add clippy

# Documentation
rustup component add rust-docs
```

### 1.4 Explore rustup

```bash
# Show installed toolchains
rustup show

# Update Rust
rustup update

# Install nightly (for some tools)
rustup install nightly
```

---

## Task 2: Configure IDE (15 min)

### 2.1 VS Code Setup

1. Install [VS Code](https://code.visualstudio.com/)
2. Install extensions:
   - **rust-analyzer** (official Rust language server)
   - **CodeLLDB** (for debugging)
   - **Even Better TOML** (for Cargo.toml)

### 2.2 Verify rust-analyzer

Create a test file and check that:
- Syntax highlighting works
- Inline type hints appear
- Autocomplete works
- Errors are highlighted

### 2.3 Configure Settings (Optional)

`.vscode/settings.json`:
```json
{
    "rust-analyzer.check.command": "clippy",
    "rust-analyzer.inlayHints.typeHints.enable": true,
    "rust-analyzer.inlayHints.parameterHints.enable": true,
    "editor.formatOnSave": true
}
```

---

## Task 3: Create First Project (20 min)

### 3.1 Initialize Project

```bash
cargo new hello_rust
cd hello_rust
```

### 3.2 Explore Project Structure

```
hello_rust/
├── Cargo.toml
└── src/
    └── main.rs
```

**Cargo.toml:**
```toml
[package]
name = "hello_rust"
version = "0.1.0"
edition = "2024"

[dependencies]
```

**src/main.rs:**
```rust
fn main() {
    println!("Hello, world!");
}
```

### 3.3 Build and Run

```bash
# Development build
cargo build

# Run
cargo run

# Build optimized release
cargo build --release

# Check without building (faster)
cargo check
```

### 3.4 Experiment

Modify `main.rs`:

```rust
fn main() {
    // Variables are immutable by default
    let x = 5;
    println!("x = {}", x);

    // Mutable variable
    let mut y = 10;
    println!("y = {}", y);
    y = 20;
    println!("y = {}", y);

    // Type annotation
    let z: f64 = 3.14;
    println!("z = {}", z);

    // Shadowing (new variable with same name)
    let x = x + 1;
    println!("x = {}", x);
}
```

---

## Task 4: CLI Tool with Dependencies (25 min)

### 4.1 Create New Project

```bash
cargo new csv_analyzer
cd csv_analyzer
```

### 4.2 Add Dependencies

Edit `Cargo.toml`:
```toml
[package]
name = "csv_analyzer"
version = "0.1.0"
edition = "2024"

[dependencies]
csv = "1.3"
clap = { version = "4.5", features = ["derive"] }
```

### 4.3 Implement CSV Reader

Replace `src/main.rs`:

```rust
use clap::Parser;
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

/// Simple CSV analyzer
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to CSV file
    #[arg(short, long)]
    file: PathBuf,

    /// Delimiter character
    #[arg(short, long, default_value = ",")]
    delimiter: char,

    /// Show first N rows
    #[arg(short, long, default_value = "5")]
    rows: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let file = File::open(&args.file)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(args.delimiter as u8)
        .from_reader(file);

    // Print headers
    if let Some(headers) = reader.headers().ok() {
        println!("Columns: {:?}", headers);
        println!("Number of columns: {}", headers.len());
        println!();
    }

    // Print first N rows
    println!("First {} rows:", args.rows);
    for (i, result) in reader.records().enumerate() {
        if i >= args.rows {
            break;
        }
        let record = result?;
        println!("Row {}: {:?}", i + 1, record);
    }

    Ok(())
}
```

### 4.4 Create Test Data

Create `test.csv`:
```csv
name,age,score
Alice,25,95.5
Bob,30,87.3
Charlie,22,91.0
Diana,28,88.7
Eve,26,93.2
```

### 4.5 Run and Test

```bash
# Build and run
cargo run -- --file test.csv

# Show help
cargo run -- --help

# Custom options
cargo run -- --file test.csv --rows 3
```

### 4.6 Extend (Optional)

Add statistics calculation:

```rust
// Add to main() before Ok(())
fn calculate_stats(file_path: &PathBuf, delimiter: char) -> Result<(), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter as u8)
        .from_reader(file);

    let mut row_count = 0;
    for result in reader.records() {
        let _ = result?;
        row_count += 1;
    }

    println!("\nStatistics:");
    println!("Total rows: {}", row_count);

    Ok(())
}
```

---

## Task 5: Debugging (15 min)

### 5.1 Add Debug Code

Create a buggy function:

```rust
fn find_max(numbers: &[i32]) -> Option<i32> {
    if numbers.is_empty() {
        return None;
    }

    let mut max = numbers[0];
    for &num in numbers.iter().skip(1) {
        if num > max {
            max = num;
        }
    }
    Some(max)
}

fn main() {
    let data = vec![3, 1, 4, 1, 5, 9, 2, 6];

    // Set a breakpoint on the next line
    let max = find_max(&data);

    match max {
        Some(m) => println!("Maximum: {}", m),
        None => println!("Empty array"),
    }
}
```

### 5.2 VS Code Debugging

1. Set breakpoint (click left of line number)
2. Press F5 or "Run > Start Debugging"
3. Step through code with F10 (step over) or F11 (step into)
4. Inspect variables in the sidebar

### 5.3 Command-Line Debugging

```bash
# Build with debug info (default for cargo build)
cargo build

# Start GDB
rust-gdb target/debug/csv_analyzer

# Basic GDB commands:
# (gdb) break main
# (gdb) run
# (gdb) next
# (gdb) print variable_name
# (gdb) continue
# (gdb) quit
```

### 5.4 Using println! for Debugging

```rust
fn complex_calculation(x: f64) -> f64 {
    let step1 = x * 2.0;
    println!("DEBUG: step1 = {}", step1);  // Debug output

    let step2 = step1 + 10.0;
    println!("DEBUG: step2 = {}", step2);

    let result = step2.sqrt();
    println!("DEBUG: result = {}", result);

    result
}
```

Better approach with `dbg!` macro:
```rust
fn complex_calculation(x: f64) -> f64 {
    let step1 = dbg!(x * 2.0);      // Prints: [src/main.rs:2] x * 2.0 = 20.0
    let step2 = dbg!(step1 + 10.0); // Prints: [src/main.rs:3] step1 + 10.0 = 30.0
    dbg!(step2.sqrt())              // Prints: [src/main.rs:4] step2.sqrt() = 5.477...
}
```

---

## Homework

### Assignment: Port Newton's Method to Rust

Implement Newton's method for finding roots of `f(x) = x² - 2` (i.e., finding √2).

**Requirements:**
1. Create new Cargo project: `newton_method`
2. Implement iterative Newton-Raphson algorithm
3. Accept initial guess and tolerance via command line (use `clap`)
4. Print each iteration's values
5. Handle edge cases (division by near-zero derivative)

**Reference C++ implementation:**
```cpp
double newton_sqrt2(double x0, double tol, int max_iter) {
    double x = x0;
    for (int i = 0; i < max_iter; i++) {
        double fx = x * x - 2.0;
        double fpx = 2.0 * x;
        if (std::abs(fpx) < 1e-10) break;

        double x_new = x - fx / fpx;
        if (std::abs(x_new - x) < tol) return x_new;
        x = x_new;
    }
    return x;
}
```

**Expected output:**
```
$ cargo run -- --initial 1.0 --tolerance 1e-10
Iteration 1: x = 1.5, f(x) = 0.25
Iteration 2: x = 1.4166666666666667, f(x) = 0.006944444444444642
Iteration 3: x = 1.4142156862745099, f(x) = 0.000006007304882871267
Iteration 4: x = 1.4142135623746899, f(x) = 4.510614104447086e-12
Converged to √2 ≈ 1.4142135623746899
```

**Submission:** Push to your course repository.

---

## Summary

Today you:
- ✅ Installed and configured Rust toolchain
- ✅ Set up IDE with rust-analyzer
- ✅ Created projects with Cargo
- ✅ Used external crates (csv, clap)
- ✅ Practiced debugging techniques

## Next

Continue to [Seminar 1.2: Ownership Deep Dive](./seminar-01-2.md)
