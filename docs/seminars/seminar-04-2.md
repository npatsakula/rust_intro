---
sidebar_position: 8
title: "Seminar 4.2: ML for Scientific Data"
---

# Seminar 4.2: Machine Learning for Scientific Data

**Duration:** 90 minutes
**Prerequisites:** Lecture 4, Seminar 4.1

---

## Objectives

- Build regression models for material property prediction
- Apply PCA for dimensionality reduction
- Implement cross-validation
- Create surrogate models

---

## Task 1: Setup and Data Generation (15 min)

```toml
[dependencies]
linfa = "0.7"
linfa-linear = "0.7"
linfa-clustering = "0.7"
linfa-reduction = "0.7"
linfa-trees = "0.7"
ndarray = "0.16"
ndarray-rand = "0.15"
rand = "0.8"
```

```rust
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// Generate synthetic material property data
/// Features: temperature, pressure, composition (3 features)
/// Target: yield strength
fn generate_material_data(n_samples: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();

    let mut features = Array2::zeros((n_samples, 3));
    let mut targets = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // Temperature: 300-800 K
        let temp = rng.gen_range(300.0..800.0);
        // Pressure: 1-100 MPa
        let pressure = rng.gen_range(1.0..100.0);
        // Composition: 0-1 (fraction of alloying element)
        let composition = rng.gen_range(0.0..1.0);

        features[[i, 0]] = temp;
        features[[i, 1]] = pressure;
        features[[i, 2]] = composition;

        // Synthetic relationship with noise
        // Yield strength decreases with temperature, increases with pressure and composition
        let yield_strength = 500.0
            - 0.3 * temp
            + 2.0 * pressure
            + 100.0 * composition
            + 50.0 * composition * pressure / 100.0  // Interaction term
            + rng.gen_range(-20.0..20.0);  // Noise

        targets[i] = yield_strength.max(0.0);
    }

    (features, targets)
}
```

---

## Task 2: Linear Regression (20 min)

```rust
use linfa::prelude::*;
use linfa_linear::LinearRegression;

fn train_linear_model() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let (features, targets) = generate_material_data(200);

    // Split into train/test
    let (train_features, test_features) = features.view().split_at(Axis(0), 160);
    let (train_targets, test_targets) = targets.view().split_at(Axis(0), 160);

    let train = Dataset::new(train_features.to_owned(), train_targets.to_owned());
    let test = Dataset::new(test_features.to_owned(), test_targets.to_owned());

    // Train model
    let model = LinearRegression::default().fit(&train)?;

    // Evaluate
    let train_pred = model.predict(&train);
    let test_pred = model.predict(&test);

    let train_r2 = train_pred.r2(&train)?;
    let test_r2 = test_pred.r2(&test)?;

    println!("Linear Regression Results:");
    println!("  Train R²: {:.4}", train_r2);
    println!("  Test R²: {:.4}", test_r2);

    // Model coefficients
    println!("\nModel Coefficients:");
    println!("  Intercept: {:.4}", model.intercept());
    let params = model.params();
    println!("  Temperature: {:.4}", params[0]);
    println!("  Pressure: {:.4}", params[1]);
    println!("  Composition: {:.4}", params[2]);

    // Compare with true coefficients
    println!("\nExpected (approximate):");
    println!("  Intercept: 500");
    println!("  Temperature: -0.3");
    println!("  Pressure: 2.0");
    println!("  Composition: 100 + interaction");

    Ok(())
}
```

### Feature Scaling

```rust
/// Standardize features (zero mean, unit variance)
fn standardize(features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let means = features.mean_axis(Axis(0)).unwrap();
    let stds = features.std_axis(Axis(0), 0.0);

    let mut scaled = features.clone();
    for mut col in scaled.columns_mut() {
        let idx = col.as_slice().unwrap().as_ptr() as usize;
        // Manual column-wise standardization
    }

    // Simpler approach
    let scaled = features.clone();
    let n_features = features.ncols();

    let mut result = Array2::zeros(features.dim());
    for j in 0..n_features {
        let col = features.column(j);
        let mean = means[j];
        let std = if stds[j] > 1e-10 { stds[j] } else { 1.0 };

        for i in 0..features.nrows() {
            result[[i, j]] = (features[[i, j]] - mean) / std;
        }
    }

    (result, means, stds)
}
```

---

## Task 3: PCA Dimensionality Reduction (20 min)

```rust
use linfa_reduction::Pca;

fn pca_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Generate high-dimensional data (e.g., spectroscopy data)
    let n_samples = 200;
    let n_features = 50;  // Many correlated features

    let mut rng = rand::thread_rng();

    // Create data with underlying low-dimensional structure
    let latent_dim = 3;
    let latent = Array2::from_shape_fn((n_samples, latent_dim), |_| rng.gen::<f64>());

    // Project to high dimensions with random projection + noise
    let projection = Array2::from_shape_fn((latent_dim, n_features), |_| rng.gen::<f64>());
    let noise = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>() * 0.1);

    let high_dim = latent.dot(&projection) + noise;

    println!("Original data shape: {:?}", high_dim.dim());

    // Fit PCA
    let pca = Pca::params(5).fit(&high_dim)?;

    println!("\nPCA Results:");
    println!("Explained variance ratio:");
    for (i, var) in pca.explained_variance_ratio().iter().enumerate() {
        println!("  PC{}: {:.4} ({:.1}%)", i + 1, var, var * 100.0);
    }

    let total_variance: f64 = pca.explained_variance_ratio().iter().take(3).sum();
    println!("\nFirst 3 PCs explain {:.1}% of variance", total_variance * 100.0);

    // Transform data
    let reduced = pca.transform(&high_dim);
    println!("\nReduced data shape: {:?}", reduced.dim());

    Ok(())
}
```

### Visualize PCA Components

```rust
use plotters::prelude::*;

fn plot_pca_variance(explained_variance: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("pca_variance.png", (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let cumulative: Vec<f64> = explained_variance.iter()
        .scan(0.0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect();

    let n = explained_variance.len();

    let mut chart = ChartBuilder::on(&root)
        .caption("PCA Explained Variance", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..n, 0.0..1.0)?;

    chart.configure_mesh()
        .x_desc("Principal Component")
        .y_desc("Variance Ratio")
        .draw()?;

    // Bar chart for individual variance
    chart.draw_series(
        explained_variance.iter().enumerate().map(|(i, &v)| {
            Rectangle::new([(i, 0.0), (i + 1, v)], BLUE.filled())
        })
    )?;

    // Line for cumulative variance
    chart.draw_series(LineSeries::new(
        cumulative.iter().enumerate().map(|(i, &v)| (i, v)),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}
```

---

## Task 4: Cross-Validation (20 min)

```rust
/// K-fold cross-validation
fn cross_validate<M, F>(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    k: usize,
    train_fn: F,
) -> Vec<f64>
where
    F: Fn(&Dataset<f64, f64, ndarray::Dim<[usize; 1]>>) -> M,
    M: Predict<Array2<f64>, Array1<f64>>,
{
    let n = features.nrows();
    let fold_size = n / k;

    let mut scores = Vec::new();

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { n } else { (fold + 1) * fold_size };

        // Create train/test split
        let mut train_indices: Vec<usize> = (0..test_start).chain(test_end..n).collect();
        let test_indices: Vec<usize> = (test_start..test_end).collect();

        let train_features = features.select(Axis(0), &train_indices);
        let train_targets = targets.select(Axis(0), &train_indices);
        let test_features = features.select(Axis(0), &test_indices);
        let test_targets = targets.select(Axis(0), &test_indices);

        let train = Dataset::new(train_features, train_targets);
        let test = Dataset::new(test_features.clone(), test_targets.clone());

        // Train and evaluate
        let model = train_fn(&train);
        let predictions = model.predict(&test_features);

        // R² score
        let ss_res: f64 = predictions.iter()
            .zip(test_targets.iter())
            .map(|(&pred, &actual)| (actual - pred).powi(2))
            .sum();

        let mean_actual = test_targets.mean().unwrap();
        let ss_tot: f64 = test_targets.iter()
            .map(|&actual| (actual - mean_actual).powi(2))
            .sum();

        let r2 = 1.0 - ss_res / ss_tot;
        scores.push(r2);
    }

    scores
}

fn run_cross_validation() -> Result<(), Box<dyn std::error::Error>> {
    let (features, targets) = generate_material_data(200);

    let scores = cross_validate(&features, &targets, 5, |train| {
        LinearRegression::default().fit(train).unwrap()
    });

    println!("5-Fold Cross-Validation Results:");
    for (i, &score) in scores.iter().enumerate() {
        println!("  Fold {}: R² = {:.4}", i + 1, score);
    }

    let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
    let std_score = (scores.iter()
        .map(|&s| (s - mean_score).powi(2))
        .sum::<f64>() / (scores.len() - 1) as f64)
        .sqrt();

    println!("\nMean R²: {:.4} ± {:.4}", mean_score, std_score);

    Ok(())
}
```

---

## Task 5: K-Means Clustering (15 min)

```rust
use linfa_clustering::KMeans;

fn cluster_experiments() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data representing different experimental conditions
    let (features, _) = generate_material_data(300);

    // Standardize features
    let (scaled, _, _) = standardize(&features);

    // Find optimal number of clusters using elbow method
    println!("Elbow Method Analysis:");
    println!("{:>4} {:>12}", "k", "Inertia");

    for k in 2..=8 {
        let model = KMeans::params(k)
            .max_n_iterations(100)
            .fit(&scaled)?;

        let predictions = model.predict(&scaled);

        // Calculate inertia (sum of squared distances to centroids)
        let inertia: f64 = scaled.rows()
            .into_iter()
            .zip(predictions.iter())
            .map(|(point, &cluster)| {
                let centroid = model.centroids().row(cluster);
                point.iter()
                    .zip(centroid.iter())
                    .map(|(&p, &c)| (p - c).powi(2))
                    .sum::<f64>()
            })
            .sum();

        println!("{:>4} {:>12.2}", k, inertia);
    }

    // Use k=3 based on elbow
    let model = KMeans::params(3)
        .max_n_iterations(100)
        .fit(&scaled)?;

    println!("\nCluster Centroids (standardized):");
    for (i, centroid) in model.centroids().rows().into_iter().enumerate() {
        println!("  Cluster {}: [{:.2}, {:.2}, {:.2}]",
                 i, centroid[0], centroid[1], centroid[2]);
    }

    Ok(())
}
```

---

## Homework

### Assignment: Surrogate Model

Build a surrogate model that approximates an expensive simulation:

1. **Target function** (simulating expensive computation):
   ```rust
   fn expensive_simulation(x1: f64, x2: f64, x3: f64) -> f64 {
       // Simulate complex physics
       let a = (x1 * x2).sin();
       let b = (x2 + x3).cos();
       let c = x1.powi(2) + x3.ln().abs();
       a * b + c + 0.1 * a * b * c
   }
   ```

2. **Requirements:**
   - Generate training data (100-200 samples)
   - Train multiple models (linear, polynomial features)
   - Compare with true function on test set
   - Plot predicted vs actual values
   - Report RMSE and R² scores

3. **Bonus:** Implement adaptive sampling (sample more where error is high)

**Submission:** Branch `homework-4-2`

---

## Summary

- ✅ Built regression models for material properties
- ✅ Applied PCA for dimensionality reduction
- ✅ Implemented cross-validation
- ✅ Used K-means clustering
- ✅ Compared model performance

## Next

Continue to [Lecture 5: Numerical Methods and Continuum Mechanics](../lectures/lecture-05.md)
