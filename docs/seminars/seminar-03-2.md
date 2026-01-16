---
sidebar_position: 6
title: "Seminar 3.2: Tensor Operations"
---

# Seminar 3.2: Tensor Operations for Mechanics

**Duration:** 90 minutes
**Prerequisites:** Lecture 3, Seminar 3.1

---

## Objectives

- Work with `russell_tensor` for stress/strain tensors
- Implement tensor transformations
- Compute invariants and deviatoric stress
- Build linear elasticity constitutive matrix

---

## Task 1: Stress Tensor Basics (20 min)

### 1.1 Setup

```bash
cargo new tensor_seminar
cd tensor_seminar
```

```toml
[dependencies]
russell_tensor = "1.0"
nalgebra = "0.33"
plotters = "0.3"
```

### 1.2 Create and Manipulate Stress Tensors

```rust
use russell_tensor::{Mandel, Tensor2};

fn main() {
    // Create symmetric stress tensor
    let mut stress = Tensor2::new(Mandel::Symmetric);

    // Set components (using engineering notation)
    // σ = | σ_xx  σ_xy  σ_xz |
    //     | σ_xy  σ_yy  σ_yz |
    //     | σ_xz  σ_yz  σ_zz |

    stress.sym_set(0, 0, 100.0);  // σ_xx = 100 MPa
    stress.sym_set(1, 1, 50.0);   // σ_yy = 50 MPa
    stress.sym_set(2, 2, 30.0);   // σ_zz = 30 MPa
    stress.sym_set(0, 1, 20.0);   // σ_xy = 20 MPa
    stress.sym_set(1, 2, 10.0);   // σ_yz = 10 MPa
    stress.sym_set(0, 2, 5.0);    // σ_xz = 5 MPa

    println!("Stress tensor:");
    println!("σ_xx = {} MPa", stress.get(0, 0));
    println!("σ_yy = {} MPa", stress.get(1, 1));
    println!("σ_zz = {} MPa", stress.get(2, 2));
    println!("σ_xy = {} MPa", stress.get(0, 1));

    // Trace (first invariant I₁)
    let trace = stress.trace();
    println!("\nFirst invariant I₁ = tr(σ) = {:.2} MPa", trace);

    // Hydrostatic pressure
    let pressure = -trace / 3.0;
    println!("Hydrostatic pressure p = {:.2} MPa", pressure);
}
```

### 1.3 Tensor Invariants

```rust
fn compute_invariants(stress: &Tensor2) {
    // First invariant: I₁ = tr(σ) = σ_ii
    let i1 = stress.trace();

    // Second invariant: I₂ = ½(tr(σ)² - tr(σ²))
    let trace_sq = stress.trace() * stress.trace();

    // For symmetric tensor, compute tr(σ²) manually
    let mut sigma_sq = Tensor2::new(Mandel::Symmetric);
    // σ² = σ · σ
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += stress.get(i, k) * stress.get(k, j);
            }
            sigma_sq.set(i, j, sum);
        }
    }
    let trace_sigma_sq = sigma_sq.trace();
    let i2 = 0.5 * (trace_sq - trace_sigma_sq);

    // Third invariant: I₃ = det(σ)
    let i3 = stress.determinant();

    println!("Stress Invariants:");
    println!("  I₁ = {:.2}", i1);
    println!("  I₂ = {:.2}", i2);
    println!("  I₃ = {:.2}", i3);

    // J₂ invariant of deviatoric stress
    let j2 = stress.invariant_jj();
    println!("  J₂ = {:.2}", j2);
}
```

---

## Task 2: Deviatoric Stress and Von Mises (25 min)

### 2.1 Compute Deviatoric Stress

```rust
fn deviatoric_analysis(stress: &Tensor2) {
    // Deviatoric stress: s = σ - (1/3)tr(σ)I
    let mut dev = Tensor2::new(Mandel::Symmetric);
    stress.deviator(&mut dev);

    println!("Deviatoric stress tensor:");
    for i in 0..3 {
        print!("[");
        for j in 0..3 {
            print!("{:8.2}", dev.get(i, j));
        }
        println!("]");
    }

    // Verify: tr(s) = 0
    println!("tr(s) = {:.2e} (should be ~0)", dev.trace());

    // J₂ invariant of deviatoric stress
    let j2 = dev.invariant_jj();
    println!("J₂ = {:.2}", j2);

    // Von Mises stress: σ_vm = √(3J₂)
    let von_mises = (3.0 * j2).sqrt();
    println!("Von Mises stress: σ_vm = {:.2} MPa", von_mises);
}
```

### 2.2 Alternative Von Mises Calculation

```rust
fn von_mises_formula(stress: &Tensor2) -> f64 {
    let s_xx = stress.get(0, 0);
    let s_yy = stress.get(1, 1);
    let s_zz = stress.get(2, 2);
    let s_xy = stress.get(0, 1);
    let s_yz = stress.get(1, 2);
    let s_xz = stress.get(0, 2);

    // σ_vm = √(½[(σ_xx-σ_yy)² + (σ_yy-σ_zz)² + (σ_zz-σ_xx)² + 6(σ_xy² + σ_yz² + σ_xz²)])
    let term1 = (s_xx - s_yy).powi(2) + (s_yy - s_zz).powi(2) + (s_zz - s_xx).powi(2);
    let term2 = 6.0 * (s_xy.powi(2) + s_yz.powi(2) + s_xz.powi(2));

    (0.5 * (term1 + term2)).sqrt()
}
```

### 2.3 Principal Stresses

```rust
use nalgebra::{Matrix3, SymmetricEigen, Vector3};

fn principal_stress_analysis(stress: &Tensor2) {
    // Convert to nalgebra matrix for eigendecomposition
    let mat = Matrix3::new(
        stress.get(0, 0), stress.get(0, 1), stress.get(0, 2),
        stress.get(1, 0), stress.get(1, 1), stress.get(1, 2),
        stress.get(2, 0), stress.get(2, 1), stress.get(2, 2),
    );

    let eigen = SymmetricEigen::new(mat);

    // Sort eigenvalues (principal stresses): σ₁ ≥ σ₂ ≥ σ₃
    let mut principals: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
    principals.sort_by(|a, b| b.partial_cmp(a).unwrap());

    println!("Principal stresses:");
    println!("  σ₁ = {:.2} MPa (maximum)", principals[0]);
    println!("  σ₂ = {:.2} MPa (intermediate)", principals[1]);
    println!("  σ₃ = {:.2} MPa (minimum)", principals[2]);

    // Maximum shear stress
    let tau_max = (principals[0] - principals[2]) / 2.0;
    println!("Maximum shear stress: τ_max = {:.2} MPa", tau_max);

    // Verify Von Mises from principals
    let s1 = principals[0];
    let s2 = principals[1];
    let s3 = principals[2];
    let von_mises = ((s1-s2).powi(2) + (s2-s3).powi(2) + (s3-s1).powi(2)).sqrt() / 2.0_f64.sqrt();
    println!("Von Mises (from principals): σ_vm = {:.2} MPa", von_mises);
}
```

---

## Task 3: Stress Transformation (20 min)

### 3.1 Rotation Matrix

```rust
use nalgebra::{Matrix3, Rotation3, Vector3 as NVector3};

fn stress_rotation(stress: &Tensor2, axis: [f64; 3], angle_deg: f64) -> Tensor2 {
    // Convert stress to matrix
    let sigma = Matrix3::new(
        stress.get(0, 0), stress.get(0, 1), stress.get(0, 2),
        stress.get(1, 0), stress.get(1, 1), stress.get(1, 2),
        stress.get(2, 0), stress.get(2, 1), stress.get(2, 2),
    );

    // Create rotation matrix
    let axis_vec = NVector3::new(axis[0], axis[1], axis[2]).normalize();
    let angle_rad = angle_deg.to_radians();
    let rot = Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis_vec), angle_rad);
    let q = rot.matrix();

    // Transform: σ' = Q σ Q^T
    let sigma_prime = q * sigma * q.transpose();

    // Convert back to Tensor2
    let mut result = Tensor2::new(Mandel::Symmetric);
    for i in 0..3 {
        for j in 0..3 {
            result.set(i, j, sigma_prime[(i, j)]);
        }
    }

    result
}

fn test_rotation() {
    let mut stress = Tensor2::new(Mandel::Symmetric);
    stress.sym_set(0, 0, 100.0);  // Uniaxial stress in x

    // Rotate 45° around z-axis
    let rotated = stress_rotation(&stress, [0.0, 0.0, 1.0], 45.0);

    println!("Original stress (uniaxial x):");
    println!("  σ_xx = {:.2}", stress.get(0, 0));
    println!("  σ_yy = {:.2}", stress.get(1, 1));
    println!("  σ_xy = {:.2}", stress.get(0, 1));

    println!("\nRotated 45° around z:");
    println!("  σ_xx' = {:.2}", rotated.get(0, 0));
    println!("  σ_yy' = {:.2}", rotated.get(1, 1));
    println!("  σ_xy' = {:.2}", rotated.get(0, 1));

    // Invariants should be preserved
    println!("\nInvariant check:");
    println!("  tr(σ) = {:.2}", stress.trace());
    println!("  tr(σ') = {:.2}", rotated.trace());
}
```

---

## Task 4: Voigt Notation (10 min)

### 4.1 Tensor to Voigt Conversion

```rust
/// Convert symmetric 2nd order tensor to Voigt vector
/// [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
fn tensor_to_voigt(t: &Tensor2) -> [f64; 6] {
    [
        t.get(0, 0),  // σ_xx
        t.get(1, 1),  // σ_yy
        t.get(2, 2),  // σ_zz
        t.get(1, 2),  // σ_yz
        t.get(0, 2),  // σ_xz
        t.get(0, 1),  // σ_xy
    ]
}

/// Convert Voigt vector to symmetric tensor
fn voigt_to_tensor(v: &[f64; 6]) -> Tensor2 {
    let mut t = Tensor2::new(Mandel::Symmetric);
    t.sym_set(0, 0, v[0]);  // σ_xx
    t.sym_set(1, 1, v[1]);  // σ_yy
    t.sym_set(2, 2, v[2]);  // σ_zz
    t.sym_set(1, 2, v[3]);  // σ_yz
    t.sym_set(0, 2, v[4]);  // σ_xz
    t.sym_set(0, 1, v[5]);  // σ_xy
    t
}

fn test_voigt() {
    let mut stress = Tensor2::new(Mandel::Symmetric);
    stress.sym_set(0, 0, 100.0);
    stress.sym_set(1, 1, 50.0);
    stress.sym_set(2, 2, 30.0);
    stress.sym_set(0, 1, 20.0);

    let voigt = tensor_to_voigt(&stress);
    println!("Voigt: {:?}", voigt);

    let recovered = voigt_to_tensor(&voigt);
    println!("Recovered σ_xx: {}", recovered.get(0, 0));
    println!("Recovered σ_xy: {}", recovered.get(0, 1));
}
```

---

## Task 5: Elasticity Matrix (15 min)

### 5.1 Isotropic Linear Elasticity

```rust
use nalgebra::Matrix6;

/// Create isotropic elasticity matrix in Voigt notation
/// σ = C : ε  →  {σ} = [C] {ε}
fn isotropic_stiffness(e: f64, nu: f64) -> Matrix6<f64> {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));

    let c11 = lambda + 2.0 * mu;
    let c12 = lambda;
    let c44 = mu;

    Matrix6::new(
        c11, c12, c12, 0.0, 0.0, 0.0,
        c12, c11, c12, 0.0, 0.0, 0.0,
        c12, c12, c11, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, c44, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, c44, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, c44,
    )
}

/// Create compliance matrix S = C⁻¹
fn isotropic_compliance(e: f64, nu: f64) -> Matrix6<f64> {
    let s11 = 1.0 / e;
    let s12 = -nu / e;
    let s44 = 2.0 * (1.0 + nu) / e;  // = 1/G

    Matrix6::new(
        s11, s12, s12, 0.0, 0.0, 0.0,
        s12, s11, s12, 0.0, 0.0, 0.0,
        s12, s12, s11, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, s44, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, s44, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, s44,
    )
}

fn test_elasticity() {
    // Steel: E = 200 GPa, ν = 0.3
    let e = 200e9;  // Pa
    let nu = 0.3;

    let c = isotropic_stiffness(e, nu);
    let s = isotropic_compliance(e, nu);

    println!("Stiffness matrix C (GPa):");
    for i in 0..6 {
        for j in 0..6 {
            print!("{:8.1} ", c[(i, j)] / 1e9);
        }
        println!();
    }

    // Verify C * S = I
    let identity = c * s;
    println!("\nC * S (should be identity):");
    for i in 0..6 {
        for j in 0..6 {
            print!("{:6.3} ", identity[(i, j)]);
        }
        println!();
    }

    // Apply strain, compute stress
    let strain = nalgebra::Vector6::new(0.001, 0.0, 0.0, 0.0, 0.0, 0.0);  // 0.1% uniaxial
    let stress = c * strain;

    println!("\nStress from 0.1% uniaxial strain:");
    println!("  σ_xx = {:.2} MPa", stress[0] / 1e6);
    println!("  σ_yy = {:.2} MPa", stress[1] / 1e6);
    println!("  σ_zz = {:.2} MPa", stress[2] / 1e6);
}
```

---

## Homework

### Assignment: Mohr's Circle Visualization

Create a program that:

1. Takes a 2D plane stress state (σ_xx, σ_yy, σ_xy)
2. Computes principal stresses and maximum shear
3. Plots Mohr's circle using `plotters`
4. Shows stress transformation for arbitrary angle

**Output:** PNG image of Mohr's circle with:
- Circle centered at ((σ₁+σ₂)/2, 0)
- Radius (σ₁-σ₂)/2
- Points marked for original and rotated states

**Submission:** Branch `homework-3-2`

---

## Summary

- ✅ Created and analyzed stress tensors
- ✅ Computed invariants and deviatoric stress
- ✅ Calculated Von Mises and principal stresses
- ✅ Implemented stress transformation
- ✅ Built elasticity matrices

## Next

Continue to [Lecture 4: Statistics, Data Analysis, and ML](../lectures/lecture-04.md)
