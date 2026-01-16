---
sidebar_position: 5
title: "Lecture 5: Numerical Methods and FEM"
---

# Lecture 5: Numerical Methods and Continuum Mechanics Tools

**Duration:** 90 minutes
**Block:** III — Domain Applications

---

## Learning Objectives

By the end of this lecture, you will:
- Use ODE solvers from `russell_ode`
- Implement FDM for PDEs
- Understand FEM basics with `fenris`
- Work with sparse matrices

---

## 1. ODE Solvers with russell_ode

### Installation

```toml
[dependencies]
russell_ode = "1.0"
russell_lab = "1.0"
```

### Basic Usage

```rust
use russell_ode::{Method, OdeSolver, Params, System, Output, Stats};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Solve: dy/dt = -λy, y(0) = 1
    // Exact: y(t) = exp(-λt)

    let lambda = 1.0;

    let system = System::new(1, |f, t, y, _| {
        f[0] = -lambda * y[0];
        Ok(())
    });

    let params = Params::new(Method::DoPri5);
    let mut solver = OdeSolver::new(params, system)?;

    let mut y = vec![1.0];  // Initial condition
    let t0 = 0.0;
    let t1 = 5.0;

    solver.solve(&mut y, t0, t1, None)?;

    let exact = (-lambda * t1).exp();
    println!("Numerical: {:.10}", y[0]);
    println!("Exact:     {:.10}", exact);
    println!("Error:     {:.2e}", (y[0] - exact).abs());

    Ok(())
}
```

### Runge-Kutta Methods

```rust
use russell_ode::{Method, OdeSolver, Params, System, Output};

fn compare_methods() -> Result<(), Box<dyn std::error::Error>> {
    // Stiff problem: dy/dt = -1000*(y - sin(t)) + cos(t)
    let system = System::new(1, |f, t, y, _| {
        f[0] = -1000.0 * (y[0] - t.sin()) + t.cos();
        Ok(())
    });

    let methods = [
        ("Euler", Method::FwEuler),
        ("DoPri5", Method::DoPri5),
        ("DoPri8", Method::DoPri8),
        ("Radau5", Method::Radau5),  // Implicit, good for stiff
    ];

    let t0 = 0.0;
    let t1 = 1.0;
    let exact = t1.sin();  // Exact solution

    println!("{:10} {:>15} {:>12} {:>10}", "Method", "Result", "Error", "Steps");

    for (name, method) in methods {
        let params = Params::new(method);
        let mut solver = OdeSolver::new(params, system.clone())?;
        let mut y = vec![0.0];

        solver.solve(&mut y, t0, t1, None)?;

        let error = (y[0] - exact).abs();
        let stats = solver.stats();

        println!("{:10} {:>15.10} {:>12.2e} {:>10}",
                 name, y[0], error, stats.n_accepted);
    }

    Ok(())
}
```

### System of ODEs

```rust
/// Lorenz system
fn lorenz_system() -> Result<(), Box<dyn std::error::Error>> {
    let sigma = 10.0;
    let rho = 28.0;
    let beta = 8.0 / 3.0;

    let system = System::new(3, move |f, _, y, _| {
        f[0] = sigma * (y[1] - y[0]);           // dx/dt
        f[1] = y[0] * (rho - y[2]) - y[1];      // dy/dt
        f[2] = y[0] * y[1] - beta * y[2];       // dz/dt
        Ok(())
    });

    let params = Params::new(Method::DoPri5);
    let mut solver = OdeSolver::new(params, system)?;

    let mut y = vec![1.0, 1.0, 1.0];  // Initial condition
    let mut trajectory = Vec::new();

    // Record trajectory
    let dt = 0.01;
    let mut t = 0.0;
    while t < 50.0 {
        solver.solve(&mut y, t, t + dt, None)?;
        trajectory.push((y[0], y[1], y[2]));
        t += dt;
    }

    println!("Final state: ({:.4}, {:.4}, {:.4})", y[0], y[1], y[2]);
    println!("Trajectory points: {}", trajectory.len());

    Ok(())
}
```

---

## 2. PDE Discretization: Finite Differences

### 1D Heat Equation

```rust
/// Solve: ∂u/∂t = α ∂²u/∂x²
/// with u(0,t) = u(L,t) = 0 and u(x,0) = sin(πx/L)
fn heat_equation_fdm() {
    let l = 1.0;           // Domain length
    let alpha = 0.01;      // Thermal diffusivity
    let nx = 50;           // Spatial points
    let dx = l / (nx - 1) as f64;
    let dt = 0.4 * dx * dx / alpha;  // CFL condition

    let t_final = 0.5;
    let nt = (t_final / dt) as usize;

    // Initial condition
    let mut u: Vec<f64> = (0..nx)
        .map(|i| (std::f64::consts::PI * i as f64 * dx / l).sin())
        .collect();

    let mut u_new = vec![0.0; nx];

    // Time stepping (explicit Euler)
    let r = alpha * dt / (dx * dx);

    for _ in 0..nt {
        // Interior points
        for i in 1..nx-1 {
            u_new[i] = u[i] + r * (u[i+1] - 2.0 * u[i] + u[i-1]);
        }

        // Boundary conditions
        u_new[0] = 0.0;
        u_new[nx-1] = 0.0;

        std::mem::swap(&mut u, &mut u_new);
    }

    // Exact solution
    let exact: Vec<f64> = (0..nx)
        .map(|i| {
            let x = i as f64 * dx;
            (std::f64::consts::PI * x / l).sin()
                * (-alpha * std::f64::consts::PI.powi(2) * t_final / (l * l)).exp()
        })
        .collect();

    // Error
    let error: f64 = u.iter()
        .zip(exact.iter())
        .map(|(n, e)| (n - e).powi(2))
        .sum::<f64>()
        .sqrt() / nx as f64;

    println!("1D Heat Equation (FDM):");
    println!("  Grid points: {}", nx);
    println!("  Time steps: {}", nt);
    println!("  RMS Error: {:.6}", error);
}
```

### 2D Laplace Equation (Jacobi Iteration)

```rust
/// Solve: ∂²u/∂x² + ∂²u/∂y² = 0
fn laplace_2d() {
    let nx = 50;
    let ny = 50;

    let mut u = vec![vec![0.0; ny]; nx];
    let mut u_new = vec![vec![0.0; ny]; nx];

    // Boundary conditions
    for i in 0..nx {
        u[i][ny-1] = 1.0;  // Top boundary = 1
    }

    // Jacobi iteration
    let max_iter = 10000;
    let tolerance = 1e-6;

    for iter in 0..max_iter {
        let mut max_diff = 0.0;

        for i in 1..nx-1 {
            for j in 1..ny-1 {
                u_new[i][j] = 0.25 * (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]);
                max_diff = max_diff.max((u_new[i][j] - u[i][j]).abs());
            }
        }

        // Copy boundaries
        for i in 0..nx {
            u_new[i][ny-1] = 1.0;
        }

        std::mem::swap(&mut u, &mut u_new);

        if max_diff < tolerance {
            println!("Converged after {} iterations", iter);
            break;
        }
    }

    // Print center value
    println!("u(0.5, 0.5) = {:.6}", u[nx/2][ny/2]);
}
```

---

## 3. Finite Element Method Concepts

### FEM Workflow

1. **Discretization**: Divide domain into elements
2. **Shape functions**: Define interpolation within elements
3. **Weak form**: Convert PDE to integral form
4. **Assembly**: Build global stiffness matrix
5. **Solve**: Solve linear system Ku = f
6. **Post-process**: Compute derived quantities

### 1D Bar Element (Manual Implementation)

```rust
use nalgebra::{DMatrix, DVector};

/// 1D bar element with linear shape functions
struct Bar1D {
    length: f64,
    area: f64,
    youngs_modulus: f64,
}

impl Bar1D {
    fn stiffness_matrix(&self) -> DMatrix<f64> {
        let k = self.youngs_modulus * self.area / self.length;
        DMatrix::from_row_slice(2, 2, &[
            k, -k,
            -k, k,
        ])
    }
}

fn simple_truss() {
    // Two-bar truss
    //
    //    [1]----(bar1)----[2]----(bar2)----[3]
    //    Fixed           Load              Free

    let bar1 = Bar1D { length: 1.0, area: 0.01, youngs_modulus: 200e9 };
    let bar2 = Bar1D { length: 1.0, area: 0.01, youngs_modulus: 200e9 };

    // Global stiffness matrix (3 nodes, 1 DOF each)
    let mut k_global = DMatrix::zeros(3, 3);

    // Assemble bar 1 (nodes 1-2, indices 0-1)
    let k1 = bar1.stiffness_matrix();
    for i in 0..2 {
        for j in 0..2 {
            k_global[(i, j)] += k1[(i, j)];
        }
    }

    // Assemble bar 2 (nodes 2-3, indices 1-2)
    let k2 = bar2.stiffness_matrix();
    for i in 0..2 {
        for j in 0..2 {
            k_global[(i + 1, j + 1)] += k2[(i, j)];
        }
    }

    println!("Global stiffness matrix (GPa/m):");
    for i in 0..3 {
        for j in 0..3 {
            print!("{:12.2} ", k_global[(i, j)] / 1e9);
        }
        println!();
    }

    // Apply boundary conditions (node 1 fixed)
    // Reduce system: solve for u2, u3

    let k_reduced = DMatrix::from_row_slice(2, 2, &[
        k_global[(1, 1)], k_global[(1, 2)],
        k_global[(2, 1)], k_global[(2, 2)],
    ]);

    // Force: 1000 N at node 2
    let f_reduced = DVector::from_row_slice(&[1000.0, 0.0]);

    // Solve
    let u_reduced = k_reduced.lu().solve(&f_reduced).unwrap();

    println!("\nDisplacements:");
    println!("  u1 = 0 (fixed)");
    println!("  u2 = {:.6e} m", u_reduced[0]);
    println!("  u3 = {:.6e} m", u_reduced[1]);

    // Stresses
    let strain1 = u_reduced[0] / bar1.length;
    let stress1 = bar1.youngs_modulus * strain1;
    println!("\nBar 1 stress: {:.2} MPa", stress1 / 1e6);
}
```

---

## 4. Using fenris Library

### Installation

```toml
[dependencies]
fenris = "0.0.14"
nalgebra = "0.33"
```

### Basic fenris Usage

```rust
use fenris::element::{Tri3d2Element, ElementConnectivity};
use fenris::assembly::{assemble_scalar, assemble_vector};
use nalgebra::{Point2, Vector2};

fn fenris_example() {
    // Define a simple triangular mesh
    let vertices = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(0.5, 1.0),
    ];

    // Single triangle element
    let connectivity = vec![[0, 1, 2]];

    // Create element
    let element = Tri3d2Element::from_vertices([
        vertices[0],
        vertices[1],
        vertices[2],
    ]);

    // Element properties
    println!("Triangle element:");
    println!("  Vertices: {:?}", vertices);
    println!("  Area: {}", element.reference_jacobian().determinant() / 2.0);
}
```

---

## 5. Sparse Matrices with russell_sparse

### Installation

```toml
[dependencies]
russell_sparse = "1.0"
```

### Sparse Matrix Operations

```rust
use russell_sparse::{SparseTriplet, LinSolver, Sym};

fn sparse_solver_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create sparse matrix in triplet format
    //
    //     | 2  3  0  0  0 |
    //     | 3  0  4  0  6 |
    // A = | 0 -1 -3  2  0 |
    //     | 0  0  1  0  0 |
    //     | 0  4  2  0  1 |

    let mut trip = SparseTriplet::new(5, 5, 13, Sym::No)?;

    // Row 0
    trip.put(0, 0, 2.0)?;
    trip.put(0, 1, 3.0)?;

    // Row 1
    trip.put(1, 0, 3.0)?;
    trip.put(1, 2, 4.0)?;
    trip.put(1, 4, 6.0)?;

    // Row 2
    trip.put(2, 1, -1.0)?;
    trip.put(2, 2, -3.0)?;
    trip.put(2, 3, 2.0)?;

    // Row 3
    trip.put(3, 2, 1.0)?;

    // Row 4
    trip.put(4, 1, 4.0)?;
    trip.put(4, 2, 2.0)?;
    trip.put(4, 4, 1.0)?;

    // Right-hand side
    let b = vec![8.0, 45.0, -3.0, 3.0, 19.0];

    // Solve
    let mut x = vec![0.0; 5];
    let mut solver = LinSolver::new()?;
    solver.compute(&trip)?;
    solver.solve(&mut x, &b)?;

    println!("Solution: {:?}", x);

    Ok(())
}
```

---

## Summary

| Tool | Purpose | Use Case |
|------|---------|----------|
| `russell_ode` | ODE integration | Time-dependent problems |
| FDM (manual) | PDE discretization | Simple geometries |
| `fenris` | FEM library | Complex geometries |
| `russell_sparse` | Sparse linear algebra | Large systems |

---

## Next Steps

- **Seminar 5.1**: [ODE Solvers and Time Integration](../seminars/seminar-05-1.md)
- **Seminar 5.2**: [Introduction to FEM in Rust](../seminars/seminar-05-2.md)
