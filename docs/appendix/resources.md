---
sidebar_position: 1
title: Resources and References
---

# Resources and References

## Official Documentation

### Rust Language
- [The Rust Programming Language](https://doc.rust-lang.org/book/) — The official book
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) — Learn through examples
- [Rustonomicon](https://doc.rust-lang.org/nomicon/) — The dark arts of unsafe Rust
- [Rust Reference](https://doc.rust-lang.org/reference/) — Language specification
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) — Best practices

### Standard Library
- [std Documentation](https://doc.rust-lang.org/std/)
- [The Rust Standard Library Cookbook](https://rust-lang-nursery.github.io/rust-cookbook/)

## Scientific Computing Libraries

### Linear Algebra
| Library | Documentation | Description |
|---------|--------------|-------------|
| nalgebra | [nalgebra.org](https://nalgebra.org/) | General-purpose linear algebra |
| ndarray | [docs.rs/ndarray](https://docs.rs/ndarray) | N-dimensional arrays |
| faer | [docs.rs/faer](https://docs.rs/faer) | High-performance linear algebra |
| ndarray-linalg | [docs.rs/ndarray-linalg](https://docs.rs/ndarray-linalg) | LAPACK bindings for ndarray |

### Scientific Computing
| Library | Documentation | Description |
|---------|--------------|-------------|
| russell | [github.com/cpmech/russell](https://github.com/cpmech/russell) | Scientific computing suite |
| russell_tensor | Part of russell | Tensor analysis for mechanics |
| russell_ode | Part of russell | ODE/DAE solvers |
| russell_sparse | Part of russell | Sparse matrix solvers |

### Statistics and ML
| Library | Documentation | Description |
|---------|--------------|-------------|
| statrs | [docs.rs/statrs](https://docs.rs/statrs) | Statistical distributions |
| linfa | [rust-ml.github.io/linfa](https://rust-ml.github.io/linfa/) | ML framework |
| polars | [pola.rs](https://pola.rs/) | DataFrames |
| argmin | [docs.rs/argmin](https://docs.rs/argmin) | Optimization |

### Visualization
| Library | Documentation | Description |
|---------|--------------|-------------|
| plotters | [docs.rs/plotters](https://docs.rs/plotters) | Plotting library |
| gnuplot | [docs.rs/gnuplot](https://docs.rs/gnuplot) | Gnuplot bindings |

### FEM and Mechanics
| Library | Documentation | Description |
|---------|--------------|-------------|
| fenris | [github.com/InteractiveComputerGraphics/fenris](https://github.com/InteractiveComputerGraphics/fenris) | FEM library |
| parry | [parry.rs](https://www.parry.rs/) | Computational geometry |

## Books

### Rust Programming
1. **"The Rust Programming Language"** by Klabnik & Nichols
   - Official book, excellent for beginners
   - [Free online](https://doc.rust-lang.org/book/)

2. **"Programming Rust"** by Blandy, Orendorff, Tindall
   - Comprehensive, systems programming focus
   - O'Reilly, 2nd Edition 2021

3. **"Rust for Rustaceans"** by Jon Gjengset
   - Advanced topics, best practices
   - No Starch Press, 2021

4. **"Rust in Action"** by Tim McNamara
   - Systems programming projects
   - Manning, 2021

### Numerical Methods
1. **"Numerical Recipes"** by Press et al.
   - Classic algorithms reference
   - Adapt C/Fortran code to Rust

2. **"Matrix Computations"** by Golub & Van Loan
   - Definitive linear algebra reference

3. **"Finite Element Method"** by Zienkiewicz & Taylor
   - FEM fundamentals

## Online Resources

### Learning Rust
- [Rust Playground](https://play.rust-lang.org/) — Try code online
- [Exercism Rust Track](https://exercism.org/tracks/rust) — Practice exercises
- [Rustlings](https://github.com/rust-lang/rustlings) — Small exercises

### Scientific Computing in Rust
- [Scientific Computing in Rust](https://scientificcomputing.rs/) — Monthly newsletter
- [Are We Learning Yet](https://www.arewelearningyet.com/) — ML ecosystem status
- [rust-sci.github.io](https://rust-sci.github.io/) — Scientific computing hub

### Community
- [Rust Users Forum](https://users.rust-lang.org/)
- [Rust Discord](https://discord.gg/rust-lang)
- [r/rust](https://www.reddit.com/r/rust/)

## Tools

### Development
| Tool | Purpose |
|------|---------|
| rust-analyzer | IDE support |
| rustfmt | Code formatting |
| clippy | Linting |
| cargo-edit | Dependency management |
| cargo-watch | Auto-rebuild |

### Testing & Verification
| Tool | Purpose |
|------|---------|
| proptest | Property-based testing |
| cargo-fuzz | Fuzzing |
| miri | UB detection |
| kani | Bounded model checking |
| criterion | Benchmarking |

### Profiling
| Tool | Purpose |
|------|---------|
| perf | Linux profiling |
| flamegraph | Visualization |
| cargo-flamegraph | Rust integration |

## Research Papers

### Rust for Scientific Computing
- "Safe Systems Programming in Rust" — CACM 2021
- Various papers using fenris for graphics research

### Numerical Methods
- See specific algorithm papers referenced in lecture notes

## Video Resources

### Rust
- [Jon Gjengset's YouTube](https://www.youtube.com/c/JonGjengset) — Advanced Rust
- [Rust Conf Videos](https://www.youtube.com/c/RustVideos)

### Scientific Computing
- Various conference talks at SciPy, JuliaCon (concepts transfer)
