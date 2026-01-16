---
sidebar_position: 4
title: "Lecture 4: Statistics, Data Analysis, and ML"
---

# Lecture 4: Statistics, Data Analysis, and Machine Learning

**Duration:** 90 minutes
**Block:** II — Mathematical Tooling

---

## Learning Objectives

By the end of this lecture, you will:
- Use `statrs` for statistical computing
- Manipulate data with `polars` DataFrames
- Apply machine learning with `linfa`
- Implement optimization with `argmin`

---

## 1. Statistical Computing with statrs

### Installation

```toml
[dependencies]
statrs = "0.18"
rand = "0.8"
rand_distr = "0.4"
```

### Probability Distributions

```rust
use statrs::distribution::{Normal, Continuous, ContinuousCDF};
use statrs::distribution::{Uniform, Exponential, Gamma, Beta};

fn main() {
    // Normal distribution N(μ=0, σ=1)
    let normal = Normal::new(0.0, 1.0).unwrap();

    println!("Normal(0, 1):");
    println!("  PDF at x=0: {:.4}", normal.pdf(0.0));
    println!("  CDF at x=0: {:.4}", normal.cdf(0.0));
    println!("  Mean: {}", normal.mean().unwrap());
    println!("  Variance: {}", normal.variance().unwrap());

    // Quantiles (inverse CDF)
    println!("  95th percentile: {:.4}", normal.inverse_cdf(0.95));

    // Other distributions
    let uniform = Uniform::new(0.0, 1.0).unwrap();
    let exponential = Exponential::new(1.0).unwrap();
    let gamma = Gamma::new(2.0, 1.0).unwrap();

    println!("\nExponential(λ=1) mean: {}", exponential.mean().unwrap());
    println!("Gamma(α=2, β=1) mean: {}", gamma.mean().unwrap());
}
```

### Random Sampling

```rust
use rand::thread_rng;
use rand_distr::{Normal, Distribution};

fn generate_samples() {
    let mut rng = thread_rng();
    let normal = Normal::new(100.0, 15.0).unwrap();  // IQ distribution

    // Generate samples
    let samples: Vec<f64> = (0..1000)
        .map(|_| normal.sample(&mut rng))
        .collect();

    // Compute sample statistics
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance: f64 = samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (samples.len() - 1) as f64;

    println!("Sample mean: {:.2} (expected: 100)", mean);
    println!("Sample std: {:.2} (expected: 15)", variance.sqrt());
}
```

### Hypothesis Testing

```rust
use statrs::distribution::{Normal, StudentsT, ContinuousCDF};

/// One-sample t-test
fn t_test(sample: &[f64], hypothesized_mean: f64) -> (f64, f64) {
    let n = sample.len() as f64;
    let mean: f64 = sample.iter().sum::<f64>() / n;
    let variance: f64 = sample.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);
    let std_error = (variance / n).sqrt();

    let t_stat = (mean - hypothesized_mean) / std_error;
    let t_dist = StudentsT::new(0.0, 1.0, n - 1.0).unwrap();

    // Two-tailed p-value
    let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));

    (t_stat, p_value)
}

fn test_hypothesis() {
    let data = vec![102.0, 98.0, 105.0, 99.0, 101.0, 97.0, 103.0, 100.0];
    let (t, p) = t_test(&data, 100.0);

    println!("t-statistic: {:.4}", t);
    println!("p-value: {:.4}", p);

    if p < 0.05 {
        println!("Reject null hypothesis at α=0.05");
    } else {
        println!("Fail to reject null hypothesis");
    }
}
```

---

## 2. Data Manipulation with polars

### Installation

```toml
[dependencies]
polars = { version = "0.46", features = ["lazy", "csv", "parquet"] }
```

### DataFrame Basics

```rust
use polars::prelude::*;

fn main() -> Result<(), PolarsError> {
    // Create DataFrame
    let df = df! {
        "name" => ["Alice", "Bob", "Charlie", "Diana"],
        "age" => [25, 30, 35, 28],
        "score" => [85.5, 92.0, 78.5, 88.0],
    }?;

    println!("{}", df);

    // Select columns
    let selected = df.select(["name", "score"])?;
    println!("\nSelected:\n{}", selected);

    // Filter rows
    let filtered = df.clone().lazy()
        .filter(col("age").gt(lit(27)))
        .collect()?;
    println!("\nAge > 27:\n{}", filtered);

    // Add computed column
    let with_grade = df.clone().lazy()
        .with_column(
            when(col("score").gt(lit(90.0)))
                .then(lit("A"))
                .when(col("score").gt(lit(80.0)))
                .then(lit("B"))
                .otherwise(lit("C"))
                .alias("grade")
        )
        .collect()?;
    println!("\nWith grades:\n{}", with_grade);

    Ok(())
}
```

### Reading/Writing Files

```rust
use polars::prelude::*;
use std::fs::File;

fn io_operations() -> Result<(), PolarsError> {
    // Read CSV
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("data.csv".into()))?
        .finish()?;

    // Write CSV
    let mut file = File::create("output.csv")?;
    CsvWriter::new(&mut file).finish(&mut df.clone())?;

    // Read Parquet (efficient binary format)
    // let df = ParquetReader::new(File::open("data.parquet")?).finish()?;

    Ok(())
}
```

### GroupBy and Aggregation

```rust
fn aggregation_example() -> Result<(), PolarsError> {
    let df = df! {
        "experiment" => ["A", "A", "A", "B", "B", "B"],
        "trial" => [1, 2, 3, 1, 2, 3],
        "value" => [10.5, 11.2, 10.8, 15.1, 14.8, 15.3],
    }?;

    let grouped = df.lazy()
        .group_by([col("experiment")])
        .agg([
            col("value").mean().alias("mean"),
            col("value").std(1).alias("std"),
            col("value").min().alias("min"),
            col("value").max().alias("max"),
            col("value").count().alias("n"),
        ])
        .collect()?;

    println!("Grouped statistics:\n{}", grouped);

    Ok(())
}
```

---

## 3. Machine Learning with linfa

### Installation

```toml
[dependencies]
linfa = "0.7"
linfa-linear = "0.7"
linfa-clustering = "0.7"
linfa-reduction = "0.7"
ndarray = "0.16"
```

### Linear Regression

```rust
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{array, Array1, Array2};

fn linear_regression_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create training data
    let x = array![
        [1.0], [2.0], [3.0], [4.0], [5.0],
        [6.0], [7.0], [8.0], [9.0], [10.0]
    ];
    let y = array![2.1, 4.0, 5.9, 8.1, 10.0, 12.1, 13.9, 16.0, 18.1, 20.0];

    let dataset = Dataset::new(x, y);

    // Fit model
    let model = LinearRegression::default().fit(&dataset)?;

    // Predict
    let predictions = model.predict(&dataset);

    // Evaluate
    let r2 = predictions.r2(&dataset)?;
    println!("R² score: {:.4}", r2);

    // Model parameters
    println!("Intercept: {:.4}", model.intercept());
    println!("Coefficients: {:?}", model.params());

    Ok(())
}
```

### K-Means Clustering

```rust
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use rand::Rng;

fn kmeans_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();

    // Generate clustered data
    let mut data = Vec::new();
    // Cluster 1: centered at (0, 0)
    for _ in 0..50 {
        data.push([rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]);
    }
    // Cluster 2: centered at (5, 5)
    for _ in 0..50 {
        data.push([rng.gen_range(4.0..6.0), rng.gen_range(4.0..6.0)]);
    }
    // Cluster 3: centered at (0, 5)
    for _ in 0..50 {
        data.push([rng.gen_range(-1.0..1.0), rng.gen_range(4.0..6.0)]);
    }

    let dataset = Array2::from(data);

    // Fit K-Means
    let model = KMeans::params(3)
        .max_n_iterations(100)
        .tolerance(1e-4)
        .fit(&dataset)?;

    // Get cluster assignments
    let predictions = model.predict(&dataset);

    println!("Cluster centroids:");
    for (i, centroid) in model.centroids().outer_iter().enumerate() {
        println!("  Cluster {}: ({:.2}, {:.2})", i, centroid[0], centroid[1]);
    }

    // Count points per cluster
    let mut counts = [0; 3];
    for &label in predictions.iter() {
        counts[label] += 1;
    }
    println!("Points per cluster: {:?}", counts);

    Ok(())
}
```

### PCA Dimensionality Reduction

```rust
use linfa::prelude::*;
use linfa_reduction::Pca;
use ndarray::Array2;

fn pca_example() -> Result<(), Box<dyn std::error::Error>> {
    // High-dimensional data (e.g., 10 features)
    let data: Array2<f64> = Array2::from_shape_fn((100, 10), |_| rand::random());

    // Reduce to 2 dimensions
    let pca = Pca::params(2).fit(&data)?;

    let reduced = pca.transform(&data);

    println!("Original shape: {:?}", data.shape());
    println!("Reduced shape: {:?}", reduced.shape());

    // Explained variance
    println!("Explained variance ratio: {:?}", pca.explained_variance_ratio());

    Ok(())
}
```

---

## 4. Optimization with argmin

### Installation

```toml
[dependencies]
argmin = "0.10"
argmin-math = { version = "0.4", features = ["ndarray_latest-serde"] }
ndarray = "0.16"
```

### Gradient Descent

```rust
use argmin::core::{CostFunction, Error, Gradient, Executor};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use ndarray::{array, Array1};

/// Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
struct Rosenbrock {
    a: f64,
    b: f64,
}

impl CostFunction for Rosenbrock {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let x = p[0];
        let y = p[1];
        Ok((self.a - x).powi(2) + self.b * (y - x.powi(2)).powi(2))
    }
}

impl Gradient for Rosenbrock {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let x = p[0];
        let y = p[1];
        Ok(array![
            -2.0 * (self.a - x) - 4.0 * self.b * x * (y - x.powi(2)),
            2.0 * self.b * (y - x.powi(2))
        ])
    }
}

fn optimize_rosenbrock() -> Result<(), Error> {
    let problem = Rosenbrock { a: 1.0, b: 100.0 };

    let linesearch = MoreThuenteLineSearch::new();
    let solver = SteepestDescent::new(linesearch);

    let init_param = array![-1.0, 1.0];

    let result = Executor::new(problem, solver)
        .configure(|state| state.param(init_param).max_iters(1000))
        .run()?;

    println!("Optimization result:");
    println!("  Best cost: {:.6}", result.state().best_cost);
    println!("  Best param: {:?}", result.state().best_param);
    println!("  Iterations: {}", result.state().iter);

    Ok(())
}
```

### Curve Fitting with BFGS

```rust
use argmin::core::{CostFunction, Gradient, Error, Executor};
use argmin::solver::quasinewton::BFGS;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use ndarray::{array, Array1};

/// Fit y = a * exp(-b * x) to data
struct ExponentialFit {
    x_data: Vec<f64>,
    y_data: Vec<f64>,
}

impl CostFunction for ExponentialFit {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let a = p[0];
        let b = p[1];

        let sum_sq: f64 = self.x_data.iter()
            .zip(self.y_data.iter())
            .map(|(&x, &y)| {
                let pred = a * (-b * x).exp();
                (y - pred).powi(2)
            })
            .sum();

        Ok(sum_sq / self.x_data.len() as f64)
    }
}

impl Gradient for ExponentialFit {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let a = p[0];
        let b = p[1];
        let n = self.x_data.len() as f64;

        let (grad_a, grad_b) = self.x_data.iter()
            .zip(self.y_data.iter())
            .fold((0.0, 0.0), |(ga, gb), (&x, &y)| {
                let exp_term = (-b * x).exp();
                let residual = y - a * exp_term;
                (
                    ga - 2.0 * residual * exp_term,
                    gb + 2.0 * residual * a * x * exp_term,
                )
            });

        Ok(array![grad_a / n, grad_b / n])
    }
}
```

---

## 5. Visualization with plotters

```toml
[dependencies]
plotters = "0.3"
```

```rust
use plotters::prelude::*;

fn plot_data() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Sample Data", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..10.0, 0.0..100.0)?;

    chart.configure_mesh().draw()?;

    // Plot data points
    let data: Vec<(f64, f64)> = (0..10)
        .map(|x| (x as f64, (x as f64).powi(2)))
        .collect();

    chart.draw_series(
        data.iter().map(|&(x, y)| Circle::new((x, y), 5, BLUE.filled()))
    )?;

    // Plot line
    chart.draw_series(LineSeries::new(
        (0..100).map(|x| {
            let xf = x as f64 / 10.0;
            (xf, xf.powi(2))
        }),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}
```

---

## Summary

| Library | Purpose | Python Equivalent |
|---------|---------|-------------------|
| `statrs` | Distributions, stats | scipy.stats |
| `polars` | DataFrames | pandas |
| `linfa` | ML algorithms | scikit-learn |
| `argmin` | Optimization | scipy.optimize |
| `plotters` | Visualization | matplotlib |

---

## Next Steps

- **Seminar 4.1**: [Statistical Analysis Pipeline](../seminars/seminar-04-1.md)
- **Seminar 4.2**: [Machine Learning for Scientific Data](../seminars/seminar-04-2.md)
