---
sidebar_position: 2
title: Assessment Structure
---

# Assessment Structure

## Overview

| Component | Weight | Description |
|-----------|--------|-------------|
| Homework | 30% | Weekly assignments |
| Midterm | 20% | Practical coding exam |
| Final Project | 40% | Substantial implementation |
| Participation | 10% | Seminar engagement |

---

## Homework (30%)

### Structure
- **14 assignments** (one per seminar)
- Each worth ~2% of final grade
- Due one week after seminar

### Grading Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Correctness | 40% | Code produces correct results |
| Code Quality | 25% | Style, organization, documentation |
| Testing | 20% | Unit tests, edge cases |
| Efficiency | 15% | Appropriate algorithms |

### Submission
- Push to course repository
- Branch naming: `homework-X-Y` (e.g., `homework-3-1`)
- Include README with build instructions

### Late Policy
- 10% penalty per day late
- Maximum 3 days late
- Extensions by request before deadline

---

## Midterm Exam (20%)

### Format
- **Duration:** 2 hours
- **Type:** Practical coding exam
- **Coverage:** Blocks I and II (Lectures 1-4)

### Structure

| Section | Points | Description |
|---------|--------|-------------|
| Part A | 30 | Short coding problems (3 × 10 pts) |
| Part B | 40 | Algorithm implementation |
| Part C | 30 | Debug and fix provided code |

### Topics Covered
1. Ownership and borrowing
2. Traits and generics
3. Error handling
4. Linear algebra with nalgebra
5. Basic data analysis

### Rules
- Open documentation (docs.rs, official docs)
- No external communication
- No AI assistance
- Pre-approved crates only

### Sample Problems

**Part A Example:**
```rust
// Fix the following code to compile:
fn longest_string(strings: &[String]) -> &str {
    let mut longest = "";
    for s in strings {
        if s.len() > longest.len() {
            longest = s;
        }
    }
    longest
}
```

**Part B Example:**
```
Implement a function that solves a tridiagonal system of equations
using the Thomas algorithm. Include proper error handling for
singular systems.
```

---

## Final Project (40%)

### Timeline
- **Week 6:** Project proposal due
- **Week 7:** Progress check-in
- **Week 8:** Final submission + presentation

### Requirements

#### Code (25%)
| Criterion | Points |
|-----------|--------|
| Functionality | 10 |
| Code organization | 5 |
| Error handling | 5 |
| Documentation | 5 |

#### Testing (10%)
| Criterion | Points |
|-----------|--------|
| Unit tests | 3 |
| Property tests | 4 |
| Integration tests | 3 |

#### Report (5%)
| Criterion | Points |
|-----------|--------|
| Problem description | 1 |
| Implementation details | 2 |
| Results and validation | 2 |

### Project Options

1. **2D Elasticity Solver**
   - Triangular elements
   - Sparse assembly
   - Visualization

2. **ODE Integrator Library**
   - Multiple methods
   - Adaptive stepping
   - Error estimation

3. **Data Analysis Pipeline**
   - Data loading
   - Statistical analysis
   - Visualization

4. **C++ Library Wrapper**
   - Safe Rust API
   - FFI bindings
   - Testing

### Deliverables
1. Source code (GitHub repository)
2. README with build instructions
3. Written report (3-5 pages)
4. 10-minute presentation

---

## Participation (10%)

### Components

| Activity | Points |
|----------|--------|
| Seminar attendance | 4 |
| Code review participation | 3 |
| Helping classmates | 2 |
| Office hours engagement | 1 |

### Seminar Attendance
- 14 seminars total
- Each attendance = 0.3 points
- Must be present for full session

### Code Review
- Review 2+ homework submissions from peers
- Provide constructive feedback
- Use course review checklist

---

## Grading Scale

| Grade | Percentage | Description |
|-------|------------|-------------|
| A | 90-100% | Excellent |
| B | 80-89% | Good |
| C | 70-79% | Satisfactory |
| D | 60-69% | Passing |
| F | \<60% | Failing |

---

## Academic Integrity

### Allowed
- Consulting official documentation
- Discussing concepts with classmates
- Using course-provided examples
- Referencing Stack Overflow (with citation)

### Not Allowed
- Copying code from classmates
- Using AI to generate solutions
- Submitting others' work as your own
- Sharing solutions publicly

### Consequences
- First offense: Zero on assignment
- Second offense: Course failure
- All violations reported to department

---

## Accommodations

- Contact instructor in first week for accommodations
- Extended deadlines available with documentation
- Alternative exam formats if needed
