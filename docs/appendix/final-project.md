---
sidebar_position: 3
title: Final Project Guidelines
---

# Final Project Guidelines

## Overview

The final project demonstrates your ability to apply Rust to a substantial scientific computing problem. You will design, implement, test, and document a software project relevant to applied mathematics.

**Weight:** 40% of final grade
**Duration:** 3 weeks

---

## Project Options

### Option 1: 2D Elasticity Solver

**Description:** Implement a finite element solver for 2D linear elasticity problems.

**Requirements:**
- Triangular elements (CST or LST)
- Plane stress and plane strain modes
- Sparse matrix assembly and solution
- Post-processing (stress, strain, displacement)
- Visualization of results

**Suggested Libraries:**
- `nalgebra` for dense linear algebra
- `russell_sparse` for sparse solving
- `plotters` for visualization

**Milestones:**
1. Element stiffness matrix implementation
2. Global assembly with boundary conditions
3. Solver integration
4. Post-processing and visualization

---

### Option 2: ODE Integration Library

**Description:** Create a library of ODE integrators for initial value problems.

**Requirements:**
- At least 3 methods (e.g., RK4, DoPri5, implicit Euler)
- Adaptive time stepping with error control
- Event detection capability
- Comprehensive test suite

**Suggested Structure:**
```rust
pub trait ODESolver {
    fn step(&mut self, f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
            t: f64, y: &mut [f64], dt: f64) -> Result<f64, ODEError>;
}

pub struct AdaptiveSolver<S: ODESolver> {
    solver: S,
    tol: f64,
    // ...
}
```

**Milestones:**
1. Basic fixed-step methods
2. Adaptive stepping
3. Stiff solver (implicit method)
4. Benchmark suite

---

### Option 3: Scientific Data Analysis Pipeline

**Description:** Build a data analysis tool for experimental/simulation data.

**Requirements:**
- Multiple input formats (CSV, JSON, binary)
- Statistical analysis functions
- Curve fitting (linear, polynomial, custom)
- Report generation (markdown, plots)

**Suggested Libraries:**
- `polars` for data manipulation
- `linfa` for ML/fitting
- `statrs` for statistics
- `plotters` for visualization

**Milestones:**
1. Data loading and validation
2. Statistical analysis module
3. Curve fitting implementation
4. Report generator

---

### Option 4: C/C++ Library Wrapper

**Description:** Create safe Rust bindings for an existing C/C++ scientific library.

**Suggested Libraries to Wrap:**
- SuiteSparse (sparse solvers)
- FFTW (FFT)
- PETSc subset
- Your research group's code

**Requirements:**
- Safe Rust API
- Comprehensive documentation
- Memory safety verification (miri)
- Performance comparison

**Milestones:**
1. Raw FFI bindings (bindgen)
2. Safe wrapper design
3. Testing and documentation
4. Benchmarking

---

## Proposal (Week 6)

Submit a 1-page proposal including:

1. **Project title and option chosen**
2. **Problem description** (what you're solving)
3. **Approach** (algorithms, libraries)
4. **Deliverables** (specific outputs)
5. **Timeline** (weekly goals)

### Proposal Template

```markdown
# Project Proposal: [Title]

## Option
[1/2/3/4]

## Problem Description
[2-3 paragraphs describing the problem and motivation]

## Technical Approach
- Algorithm: [description]
- Libraries: [list]
- Data structures: [key types]

## Deliverables
1. [Specific output 1]
2. [Specific output 2]
3. [...]

## Timeline
- Week 1: [goals]
- Week 2: [goals]
- Week 3: [goals]

## References
[Papers, documentation, existing implementations]
```

---

## Repository Structure

```
final-project/
├── Cargo.toml
├── README.md
├── LICENSE
├── src/
│   ├── lib.rs
│   ├── module1/
│   │   ├── mod.rs
│   │   └── ...
│   └── module2/
│       └── ...
├── tests/
│   ├── integration_tests.rs
│   └── property_tests.rs
├── benches/
│   └── benchmarks.rs
├── examples/
│   ├── basic_usage.rs
│   └── advanced_example.rs
├── docs/
│   └── report.md
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Grading Rubric

### Code Quality (25 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (1-2) |
|-----------|--------------|----------|------------------|------------------|
| Functionality | Fully working, handles edge cases | Works correctly | Minor bugs | Major issues |
| Organization | Clean modules, good separation | Well organized | Adequate | Poor structure |
| Error handling | Comprehensive, informative | Good coverage | Basic | Missing |
| Documentation | Excellent docs, examples | Good coverage | Adequate | Sparse |
| Style | Idiomatic Rust, clippy clean | Minor issues | Some warnings | Many issues |

### Testing (10 points)

| Criterion | Excellent (4-5) | Good (3) | Satisfactory (2) | Needs Work (0-1) |
|-----------|----------------|----------|------------------|------------------|
| Unit tests | \>80% coverage | \>60% | \>40% | \<40% |
| Property tests | Comprehensive | Good | Basic | Missing |
| Integration | Full workflow | Partial | Minimal | None |

### Report (5 points)

| Criterion | Points |
|-----------|--------|
| Clear problem statement | 1 |
| Implementation details | 2 |
| Results and validation | 2 |

---

## Presentation

**Duration:** 10 minutes + 5 minutes Q&A

### Structure
1. **Problem introduction** (2 min)
2. **Design decisions** (3 min)
3. **Demo** (3 min)
4. **Challenges and lessons** (2 min)

### Tips
- Show code running, not just slides
- Highlight interesting Rust features used
- Discuss what you learned
- Be prepared for technical questions

---

## Submission

### Deadline
End of Week 8, 23:59

### Required Files
1. GitHub repository URL
2. Written report (PDF or Markdown)
3. Presentation slides

### Checklist
- [ ] Code compiles with `cargo build`
- [ ] Tests pass with `cargo test`
- [ ] Documentation builds with `cargo doc`
- [ ] README includes build/run instructions
- [ ] CI pipeline green
- [ ] Report complete
- [ ] Presentation prepared

---

## Getting Help

- **Office hours:** [Schedule]
- **Piazza/Discord:** For questions
- **Code reviews:** Submit draft for feedback
- **Peer discussion:** Encouraged for design, not code sharing
