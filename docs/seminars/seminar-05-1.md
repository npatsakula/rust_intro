---
sidebar_position: 9
title: "Seminar 5.1: ODE Solvers"
---

# Seminar 5.1: ODE Solvers and Time Integration

**Duration:** 90 minutes
**Prerequisites:** Lecture 5

---

## Objectives

- Solve classical mechanics problems
- Implement adaptive time stepping
- Compare explicit vs implicit methods
- Visualize phase portraits

---

## Task 1: Simple Pendulum (25 min)

### 1.1 Setup

```toml
[dependencies]
russell_ode = "1.0"
plotters = "0.3"
```

### 1.2 Pendulum Equations

```rust
use russell_ode::{Method, OdeSolver, Params, System};

/// Simple pendulum: θ'' + (g/L)sin(θ) = 0
/// State: y = [θ, θ']
fn pendulum_system(g: f64, l: f64) -> System {
    System::new(2, move |f, _, y, _| {
        f[0] = y[1];                    // dθ/dt = θ'
        f[1] = -(g / l) * y[0].sin();   // dθ'/dt = -(g/L)sin(θ)
        Ok(())
    })
}

fn solve_pendulum() -> Result<Vec<(f64, f64, f64)>, Box<dyn std::error::Error>> {
    let g = 9.81;
    let l = 1.0;

    let system = pendulum_system(g, l);
    let params = Params::new(Method::DoPri5);
    let mut solver = OdeSolver::new(params, system)?;

    // Initial conditions: θ₀ = π/4, θ'₀ = 0
    let mut y = vec![std::f64::consts::FRAC_PI_4, 0.0];

    let dt = 0.01;
    let t_final = 10.0;
    let mut t = 0.0;

    let mut trajectory = vec![(t, y[0], y[1])];

    while t < t_final {
        solver.solve(&mut y, t, t + dt, None)?;
        t += dt;
        trajectory.push((t, y[0], y[1]));
    }

    // Compare with small angle approximation
    let omega = (g / l).sqrt();
    let exact_period = 2.0 * std::f64::consts::PI / omega;
    println!("Small angle period: {:.4} s", exact_period);

    // Find numerical period
    let mut crossings = Vec::new();
    for i in 1..trajectory.len() {
        if trajectory[i-1].1 > 0.0 && trajectory[i].1 <= 0.0 {
            crossings.push(trajectory[i].0);
        }
    }

    if crossings.len() >= 2 {
        let numerical_period = 2.0 * (crossings[1] - crossings[0]);
        println!("Numerical period: {:.4} s", numerical_period);
    }

    Ok(trajectory)
}
```

### 1.3 Visualize Results

```rust
use plotters::prelude::*;

fn plot_pendulum(trajectory: &[(f64, f64, f64)]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("pendulum.png", (1200, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left, right) = root.split_horizontally(600);

    // Time series plot
    {
        let t_max = trajectory.last().unwrap().0;
        let theta_max = trajectory.iter().map(|(_, theta, _)| theta.abs()).fold(0.0, f64::max);

        let mut chart = ChartBuilder::on(&left)
            .caption("Pendulum Angle vs Time", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..t_max, -theta_max..theta_max)?;

        chart.configure_mesh()
            .x_desc("Time (s)")
            .y_desc("θ (rad)")
            .draw()?;

        chart.draw_series(LineSeries::new(
            trajectory.iter().map(|(t, theta, _)| (*t, *theta)),
            &BLUE,
        ))?;
    }

    // Phase portrait
    {
        let theta_max = trajectory.iter().map(|(_, theta, _)| theta.abs()).fold(0.0, f64::max);
        let omega_max = trajectory.iter().map(|(_, _, omega)| omega.abs()).fold(0.0, f64::max);

        let mut chart = ChartBuilder::on(&right)
            .caption("Phase Portrait", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(-theta_max..theta_max, -omega_max..omega_max)?;

        chart.configure_mesh()
            .x_desc("θ (rad)")
            .y_desc("θ' (rad/s)")
            .draw()?;

        chart.draw_series(LineSeries::new(
            trajectory.iter().map(|(_, theta, omega)| (*theta, *omega)),
            &RED,
        ))?;
    }

    root.present()?;
    Ok(())
}
```

---

## Task 2: Orbital Mechanics (20 min)

```rust
/// Two-body problem: planet around star
/// State: y = [x, y, vx, vy]
fn orbital_system(gm: f64) -> System {
    System::new(4, move |f, _, y, _| {
        let x = y[0];
        let y_pos = y[1];
        let vx = y[2];
        let vy = y[3];

        let r = (x * x + y_pos * y_pos).sqrt();
        let r3 = r * r * r;

        f[0] = vx;                      // dx/dt = vx
        f[1] = vy;                      // dy/dt = vy
        f[2] = -gm * x / r3;            // dvx/dt = -GM*x/r³
        f[3] = -gm * y_pos / r3;        // dvy/dt = -GM*y/r³
        Ok(())
    })
}

fn kepler_orbit() -> Result<(), Box<dyn std::error::Error>> {
    let gm = 1.0;  // Normalized units

    // Elliptical orbit: e = 0.5
    // Initial conditions at perihelion
    let a = 1.0;   // Semi-major axis
    let e = 0.5;   // Eccentricity
    let r_peri = a * (1.0 - e);
    let v_peri = ((gm / a) * (1.0 + e) / (1.0 - e)).sqrt();

    let mut y = vec![r_peri, 0.0, 0.0, v_peri];

    let system = orbital_system(gm);
    let params = Params::new(Method::DoPri8);
    let mut solver = OdeSolver::new(params, system)?;

    // One orbital period
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / gm).sqrt();
    let dt = period / 1000.0;
    let mut t = 0.0;

    let mut orbit = vec![(y[0], y[1])];

    // Initial energy
    let compute_energy = |y: &[f64]| {
        let r = (y[0] * y[0] + y[1] * y[1]).sqrt();
        let v2 = y[2] * y[2] + y[3] * y[3];
        0.5 * v2 - gm / r
    };

    let e0 = compute_energy(&y);

    while t < period {
        solver.solve(&mut y, t, t + dt, None)?;
        t += dt;
        orbit.push((y[0], y[1]));
    }

    let ef = compute_energy(&y);

    println!("Kepler Orbit:");
    println!("  Period: {:.4} (expected: {:.4})", t, period);
    println!("  Initial energy: {:.6}", e0);
    println!("  Final energy: {:.6}", ef);
    println!("  Energy drift: {:.2e}", (ef - e0).abs());

    // Check closure
    let closure_error = ((y[0] - r_peri).powi(2) + y[1].powi(2)).sqrt();
    println!("  Orbit closure error: {:.2e}", closure_error);

    Ok(())
}
```

---

## Task 3: Stiff Systems (20 min)

```rust
/// Stiff chemical kinetics: A → B → C
/// dy1/dt = -k1*y1
/// dy2/dt = k1*y1 - k2*y2
/// dy3/dt = k2*y2
fn chemical_kinetics() -> Result<(), Box<dyn std::error::Error>> {
    let k1 = 0.04;      // Slow reaction
    let k2 = 3e7;       // Fast reaction (stiff!)

    let system = System::new(3, move |f, _, y, _| {
        f[0] = -k1 * y[0];
        f[1] = k1 * y[0] - k2 * y[1];
        f[2] = k2 * y[1];
        Ok(())
    });

    // Initial: all species A
    let y0 = vec![1.0, 0.0, 0.0];
    let t_final = 100.0;

    // Compare methods
    for (name, method) in [
        ("DoPri5 (explicit)", Method::DoPri5),
        ("Radau5 (implicit)", Method::Radau5),
    ] {
        let mut y = y0.clone();
        let params = Params::new(method);
        let mut solver = OdeSolver::new(params, system.clone())?;

        let start = std::time::Instant::now();
        solver.solve(&mut y, 0.0, t_final, None)?;
        let elapsed = start.elapsed();

        let stats = solver.stats();

        println!("\n{}:", name);
        println!("  Final state: [{:.6}, {:.6e}, {:.6}]", y[0], y[1], y[2]);
        println!("  Steps: {} accepted, {} rejected",
                 stats.n_accepted, stats.n_rejected);
        println!("  Time: {:?}", elapsed);

        // Conservation check (should sum to 1)
        let total: f64 = y.iter().sum();
        println!("  Mass conservation error: {:.2e}", (total - 1.0).abs());
    }

    Ok(())
}
```

---

## Task 4: Adaptive Time Stepping (15 min)

```rust
fn demonstrate_adaptivity() -> Result<(), Box<dyn std::error::Error>> {
    // System with varying timescales
    // y' = sin(10*t) * y
    let system = System::new(1, |f, t, y, _| {
        f[0] = (10.0 * t).sin() * y[0];
        Ok(())
    });

    let mut params = Params::new(Method::DoPri5);
    params.set_tolerances(1e-6, 1e-8)?;

    let mut solver = OdeSolver::new(params, system)?;
    let mut y = vec![1.0];

    // Solve and record step sizes
    let dt_records = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let dt_clone = dt_records.clone();

    let mut t = 0.0;
    let t_final = 10.0;

    while t < t_final {
        let t_next = (t + 0.5).min(t_final);
        solver.solve(&mut y, t, t_next, None)?;

        let stats = solver.stats();
        // Note: step sizes are managed internally by the solver

        t = t_next;
    }

    println!("Adaptive stepping demonstration:");
    println!("  Final value: {:.10}", y[0]);
    println!("  Stats: {:?}", solver.stats());

    Ok(())
}
```

---

## Task 5: Lorenz Attractor (10 min)

```rust
fn lorenz_attractor() -> Result<(), Box<dyn std::error::Error>> {
    let sigma = 10.0;
    let rho = 28.0;
    let beta = 8.0 / 3.0;

    let system = System::new(3, move |f, _, y, _| {
        f[0] = sigma * (y[1] - y[0]);
        f[1] = y[0] * (rho - y[2]) - y[1];
        f[2] = y[0] * y[1] - beta * y[2];
        Ok(())
    });

    let params = Params::new(Method::DoPri5);
    let mut solver = OdeSolver::new(params, system)?;

    let mut y = vec![1.0, 1.0, 1.0];
    let mut trajectory = Vec::new();

    let dt = 0.01;
    let mut t = 0.0;

    while t < 50.0 {
        solver.solve(&mut y, t, t + dt, None)?;
        trajectory.push((y[0], y[1], y[2]));
        t += dt;
    }

    println!("Lorenz Attractor:");
    println!("  Points generated: {}", trajectory.len());
    println!("  Final state: ({:.4}, {:.4}, {:.4})", y[0], y[1], y[2]);

    // Could plot 3D trajectory or projections
    Ok(())
}
```

---

## Homework

### Assignment: Symplectic Integrator

Implement the Störmer-Verlet (leapfrog) method for Hamiltonian systems:

```rust
/// Störmer-Verlet for H = T(p) + V(q)
/// q_{n+1} = q_n + dt * p_{n+1/2}
/// p_{n+1} = p_{n+1/2} - dt/2 * ∇V(q_{n+1})
fn stormer_verlet(
    q: &mut [f64],
    p: &mut [f64],
    dt: f64,
    grad_v: impl Fn(&[f64]) -> Vec<f64>,
) {
    // Half step for momentum
    // Full step for position
    // Half step for momentum
    todo!()
}
```

Apply to:
1. Harmonic oscillator: V = kx²/2
2. Kepler problem: V = -GM/r
3. Compare energy conservation with RK4

**Submission:** Branch `homework-5-1`

---

## Summary

- ✅ Solved pendulum and orbital mechanics problems
- ✅ Compared explicit vs implicit methods for stiff systems
- ✅ Demonstrated adaptive time stepping
- ✅ Visualized phase portraits

## Next

Continue to [Seminar 5.2: Introduction to FEM in Rust](./seminar-05-2.md)
