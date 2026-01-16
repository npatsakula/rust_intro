---
sidebar_position: 10
title: "Seminar 5.2: Introduction to FEM"
---

# Seminar 5.2: Introduction to FEM in Rust

**Duration:** 90 minutes
**Prerequisites:** Lecture 5, Seminar 5.1

---

## Objectives

- Implement 1D finite elements from scratch
- Assemble global stiffness matrices
- Solve simple structural problems
- Visualize results

---

## Task 1: 1D Bar Element (25 min)

### 1.1 Element Formulation

```rust
use nalgebra::{DMatrix, DVector, Matrix2};

/// 1D bar element
#[derive(Debug, Clone)]
pub struct BarElement {
    pub node1: usize,
    pub node2: usize,
    pub length: f64,
    pub area: f64,
    pub youngs_modulus: f64,
}

impl BarElement {
    pub fn new(node1: usize, node2: usize, length: f64, area: f64, e: f64) -> Self {
        BarElement {
            node1,
            node2,
            length,
            area,
            youngs_modulus: e,
        }
    }

    /// Local stiffness matrix (2x2)
    pub fn stiffness_local(&self) -> Matrix2<f64> {
        let k = self.youngs_modulus * self.area / self.length;
        Matrix2::new(
            k, -k,
            -k, k,
        )
    }

    /// Element stress from nodal displacements
    pub fn stress(&self, u1: f64, u2: f64) -> f64 {
        let strain = (u2 - u1) / self.length;
        self.youngs_modulus * strain
    }

    /// Element internal force
    pub fn internal_force(&self, u1: f64, u2: f64) -> f64 {
        self.stress(u1, u2) * self.area
    }
}
```

### 1.2 Simple Truss Solver

```rust
/// 1D truss structure
pub struct Truss1D {
    pub n_nodes: usize,
    pub elements: Vec<BarElement>,
    pub fixed_nodes: Vec<usize>,
    pub forces: DVector<f64>,
}

impl Truss1D {
    pub fn new(n_nodes: usize) -> Self {
        Truss1D {
            n_nodes,
            elements: Vec::new(),
            fixed_nodes: Vec::new(),
            forces: DVector::zeros(n_nodes),
        }
    }

    pub fn add_element(&mut self, elem: BarElement) {
        self.elements.push(elem);
    }

    pub fn fix_node(&mut self, node: usize) {
        if !self.fixed_nodes.contains(&node) {
            self.fixed_nodes.push(node);
        }
    }

    pub fn apply_force(&mut self, node: usize, force: f64) {
        self.forces[node] += force;
    }

    /// Assemble global stiffness matrix
    pub fn assemble_stiffness(&self) -> DMatrix<f64> {
        let mut k_global = DMatrix::zeros(self.n_nodes, self.n_nodes);

        for elem in &self.elements {
            let k_local = elem.stiffness_local();
            let dofs = [elem.node1, elem.node2];

            for i in 0..2 {
                for j in 0..2 {
                    k_global[(dofs[i], dofs[j])] += k_local[(i, j)];
                }
            }
        }

        k_global
    }

    /// Solve for displacements
    pub fn solve(&self) -> Option<(DVector<f64>, DVector<f64>)> {
        let k_global = self.assemble_stiffness();

        // Create list of free DOFs
        let free_dofs: Vec<usize> = (0..self.n_nodes)
            .filter(|n| !self.fixed_nodes.contains(n))
            .collect();

        let n_free = free_dofs.len();

        // Extract reduced system
        let mut k_reduced = DMatrix::zeros(n_free, n_free);
        let mut f_reduced = DVector::zeros(n_free);

        for (i, &dof_i) in free_dofs.iter().enumerate() {
            f_reduced[i] = self.forces[dof_i];
            for (j, &dof_j) in free_dofs.iter().enumerate() {
                k_reduced[(i, j)] = k_global[(dof_i, dof_j)];
            }
        }

        // Solve reduced system
        let u_reduced = k_reduced.lu().solve(&f_reduced)?;

        // Reconstruct full displacement vector
        let mut u_full = DVector::zeros(self.n_nodes);
        for (i, &dof) in free_dofs.iter().enumerate() {
            u_full[dof] = u_reduced[i];
        }

        // Compute reaction forces: R = K*u - F
        let reactions = &k_global * &u_full - &self.forces;

        Some((u_full, reactions))
    }
}
```

### 1.3 Example: Three-Bar Truss

```rust
fn three_bar_truss() {
    //
    //  Fixed [0]-----(1)-----[1]-----(2)-----[2]-----(3)-----[3] Force
    //
    // All bars: E = 200 GPa, A = 100 mm²

    let mut truss = Truss1D::new(4);

    let e = 200e9;      // Pa
    let a = 100e-6;     // m²
    let l = 1.0;        // m

    // Add elements
    truss.add_element(BarElement::new(0, 1, l, a, e));
    truss.add_element(BarElement::new(1, 2, l, a, e));
    truss.add_element(BarElement::new(2, 3, l, a, e));

    // Boundary conditions
    truss.fix_node(0);

    // Applied force: 10 kN at node 3
    truss.apply_force(3, 10000.0);

    // Solve
    if let Some((displacements, reactions)) = truss.solve() {
        println!("Three-Bar Truss Results:");
        println!("\nDisplacements:");
        for (i, u) in displacements.iter().enumerate() {
            println!("  Node {}: u = {:.6} mm", i, u * 1000.0);
        }

        println!("\nReactions:");
        for (i, r) in reactions.iter().enumerate() {
            if r.abs() > 1e-6 {
                println!("  Node {}: R = {:.2} N", i, r);
            }
        }

        println!("\nElement Stresses:");
        for (i, elem) in truss.elements.iter().enumerate() {
            let u1 = displacements[elem.node1];
            let u2 = displacements[elem.node2];
            let stress = elem.stress(u1, u2);
            println!("  Element {}: σ = {:.2} MPa", i, stress / 1e6);
        }
    }
}
```

---

## Task 2: 1D Beam Element (25 min)

```rust
use nalgebra::Matrix4;

/// Euler-Bernoulli beam element
#[derive(Debug, Clone)]
pub struct BeamElement {
    pub node1: usize,
    pub node2: usize,
    pub length: f64,
    pub area: f64,
    pub moment_of_inertia: f64,
    pub youngs_modulus: f64,
}

impl BeamElement {
    /// Local stiffness matrix (4x4)
    /// DOFs: [v1, θ1, v2, θ2]
    pub fn stiffness_local(&self) -> Matrix4<f64> {
        let l = self.length;
        let ei = self.youngs_modulus * self.moment_of_inertia;
        let l2 = l * l;
        let l3 = l2 * l;

        let k = ei / l3;

        Matrix4::new(
            12.0 * k,      6.0 * l * k,    -12.0 * k,      6.0 * l * k,
            6.0 * l * k,   4.0 * l2 * k,   -6.0 * l * k,   2.0 * l2 * k,
            -12.0 * k,     -6.0 * l * k,   12.0 * k,       -6.0 * l * k,
            6.0 * l * k,   2.0 * l2 * k,   -6.0 * l * k,   4.0 * l2 * k,
        )
    }
}

fn cantilever_beam() {
    // Cantilever beam with point load at free end
    //
    //  Fixed ################[============]-------> F
    //        Node 0         Node 1

    let l = 1.0;            // Length: 1 m
    let b = 0.1;            // Width: 100 mm
    let h = 0.2;            // Height: 200 mm
    let e = 200e9;          // Steel: 200 GPa

    let area = b * h;
    let i = b * h.powi(3) / 12.0;  // Moment of inertia

    let beam = BeamElement {
        node1: 0,
        node2: 1,
        length: l,
        area,
        moment_of_inertia: i,
        youngs_modulus: e,
    };

    let k = beam.stiffness_local();

    // Boundary conditions: v1 = θ1 = 0 (fixed at node 0)
    // Free DOFs: v2, θ2

    // Extract 2x2 system for free DOFs
    let k_free = nalgebra::Matrix2::new(
        k[(2, 2)], k[(2, 3)],
        k[(3, 2)], k[(3, 3)],
    );

    // Point load at free end
    let p = -10000.0;  // 10 kN downward
    let f_free = nalgebra::Vector2::new(p, 0.0);

    // Solve
    let u_free = k_free.lu().solve(&f_free).unwrap();
    let v2 = u_free[0];
    let theta2 = u_free[1];

    println!("Cantilever Beam Analysis:");
    println!("  Length: {} m", l);
    println!("  Load: {} kN", p / 1000.0);
    println!("\nResults:");
    println!("  Tip deflection: {:.4} mm", v2 * 1000.0);
    println!("  Tip rotation: {:.6} rad ({:.4}°)", theta2, theta2.to_degrees());

    // Analytical solution
    let v_exact = p * l.powi(3) / (3.0 * e * i);
    let theta_exact = p * l.powi(2) / (2.0 * e * i);

    println!("\nComparison with analytical:");
    println!("  Deflection error: {:.2e}", (v2 - v_exact).abs());
    println!("  Rotation error: {:.2e}", (theta2 - theta_exact).abs());
}
```

---

## Task 3: 2D Truss Assembly (20 min)

```rust
use nalgebra::{DMatrix, DVector, Matrix2, Matrix4, Rotation2};

/// 2D bar element
pub struct Bar2D {
    pub node1: usize,
    pub node2: usize,
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub area: f64,
    pub youngs_modulus: f64,
}

impl Bar2D {
    pub fn length(&self) -> f64 {
        ((self.x2 - self.x1).powi(2) + (self.y2 - self.y1).powi(2)).sqrt()
    }

    pub fn angle(&self) -> f64 {
        (self.y2 - self.y1).atan2(self.x2 - self.x1)
    }

    /// 4x4 stiffness matrix in global coordinates
    pub fn stiffness_global(&self) -> Matrix4<f64> {
        let l = self.length();
        let c = (self.x2 - self.x1) / l;  // cos(θ)
        let s = (self.y2 - self.y1) / l;  // sin(θ)

        let k = self.youngs_modulus * self.area / l;

        Matrix4::new(
            c*c*k,   c*s*k,   -c*c*k,  -c*s*k,
            c*s*k,   s*s*k,   -c*s*k,  -s*s*k,
            -c*c*k,  -c*s*k,  c*c*k,   c*s*k,
            -c*s*k,  -s*s*k,  c*s*k,   s*s*k,
        )
    }
}

fn simple_2d_truss() {
    // Simple triangular truss
    //
    //           [1]
    //          /   \
    //         /     \
    //        /       \
    //     [0]---------[2]
    //     Fixed       Roller
    //                   |
    //                   v F

    let e = 200e9;
    let a = 100e-6;

    // Node coordinates
    let nodes = [
        (0.0, 0.0),      // Node 0
        (0.5, 0.866),    // Node 1 (equilateral)
        (1.0, 0.0),      // Node 2
    ];

    // Elements
    let elements = vec![
        Bar2D {
            node1: 0, node2: 1,
            x1: nodes[0].0, y1: nodes[0].1,
            x2: nodes[1].0, y2: nodes[1].1,
            area: a, youngs_modulus: e,
        },
        Bar2D {
            node1: 1, node2: 2,
            x1: nodes[1].0, y1: nodes[1].1,
            x2: nodes[2].0, y2: nodes[2].1,
            area: a, youngs_modulus: e,
        },
        Bar2D {
            node1: 0, node2: 2,
            x1: nodes[0].0, y1: nodes[0].1,
            x2: nodes[2].0, y2: nodes[2].1,
            area: a, youngs_modulus: e,
        },
    ];

    // Assemble global stiffness (3 nodes × 2 DOF = 6 DOF total)
    let n_dof = 6;
    let mut k_global = DMatrix::zeros(n_dof, n_dof);

    for elem in &elements {
        let k_elem = elem.stiffness_global();
        let dofs = [
            elem.node1 * 2,     // ux1
            elem.node1 * 2 + 1, // uy1
            elem.node2 * 2,     // ux2
            elem.node2 * 2 + 1, // uy2
        ];

        for i in 0..4 {
            for j in 0..4 {
                k_global[(dofs[i], dofs[j])] += k_elem[(i, j)];
            }
        }
    }

    // Boundary conditions:
    // Node 0: ux=0, uy=0 (fixed)
    // Node 2: uy=0 (roller)
    // Free DOFs: ux1, uy1, ux2 (indices 2, 3, 4)

    let free_dofs = [2, 3, 4];
    let n_free = 3;

    let mut k_free = DMatrix::zeros(n_free, n_free);
    for (i, &di) in free_dofs.iter().enumerate() {
        for (j, &dj) in free_dofs.iter().enumerate() {
            k_free[(i, j)] = k_global[(di, dj)];
        }
    }

    // Force: 10 kN downward at node 2
    let mut f_free = DVector::zeros(n_free);
    f_free[2] = 0.0;      // ux2
    // f_free already zero for uy2, but let's add force at uy1 or modify
    // Actually, let's apply force at node 1 (uy direction)
    f_free[1] = -10000.0;  // uy1 = -10 kN

    // Solve
    let u_free = k_free.lu().solve(&f_free).unwrap();

    println!("2D Truss Analysis:");
    println!("  Node 1: ux = {:.4} mm, uy = {:.4} mm",
             u_free[0] * 1000.0, u_free[1] * 1000.0);
    println!("  Node 2: ux = {:.4} mm", u_free[2] * 1000.0);
}
```

---

## Task 4: Visualization (10 min)

```rust
use plotters::prelude::*;

fn plot_deformed_truss(
    nodes: &[(f64, f64)],
    elements: &[(usize, usize)],
    displacements: &[(f64, f64)],
    scale: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("truss.png", (600, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Truss Deformation", ("sans-serif", 20))
        .margin(20)
        .build_cartesian_2d(-0.5..1.5, -0.5..1.5)?;

    chart.configure_mesh().draw()?;

    // Draw original structure
    for &(n1, n2) in elements {
        chart.draw_series(LineSeries::new(
            vec![nodes[n1], nodes[n2]],
            &BLUE,
        ))?;
    }

    // Draw deformed structure
    let deformed: Vec<(f64, f64)> = nodes.iter()
        .zip(displacements.iter())
        .map(|((x, y), (dx, dy))| (x + scale * dx, y + scale * dy))
        .collect();

    for &(n1, n2) in elements {
        chart.draw_series(LineSeries::new(
            vec![deformed[n1], deformed[n2]],
            &RED,
        ))?;
    }

    root.present()?;
    Ok(())
}
```

---

## Homework

### Assignment: 2D Heat Conduction

Implement FEM for 2D heat conduction:
- Triangle elements with linear shape functions
- Assemble conductivity matrix
- Apply boundary conditions (fixed temperature, flux)
- Solve and visualize temperature field

**Submission:** Branch `homework-5-2`

---

## Summary

- ✅ Implemented 1D bar elements
- ✅ Built truss solver with assembly
- ✅ Extended to 2D with coordinate transformation
- ✅ Created visualization

## Next

Continue to [Lecture 6: Unsafe Rust and FFI](../lectures/lecture-06.md)
