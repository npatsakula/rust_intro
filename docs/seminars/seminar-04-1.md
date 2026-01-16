---
sidebar_position: 7
title: "Seminar 4.1: Statistical Analysis"
---

# Seminar 4.1: Statistical Analysis Pipeline

**Duration:** 90 minutes
**Prerequisites:** Lecture 4

---

## Objectives

- Load and explore experimental data with polars
- Compute descriptive statistics
- Fit probability distributions
- Perform hypothesis testing
- Visualize results

---

## Task 1: Data Loading (15 min)

### 1.1 Setup Project

```bash
cargo new stats_seminar
cd stats_seminar
```

```toml
[dependencies]
polars = { version = "0.46", features = ["lazy", "csv"] }
statrs = "0.18"
rand = "0.8"
rand_distr = "0.4"
plotters = "0.3"
```

### 1.2 Generate Sample Data

```rust
use std::fs::File;
use std::io::Write;
use rand::thread_rng;
use rand_distr::{Normal, Distribution};

fn generate_experiment_data() -> std::io::Result<()> {
    let mut rng = thread_rng();

    // Simulate experimental measurements with two groups
    let control = Normal::new(100.0, 10.0).unwrap();
    let treatment = Normal::new(105.0, 12.0).unwrap();

    let mut file = File::create("experiment.csv")?;
    writeln!(file, "group,subject,measurement")?;

    for i in 0..50 {
        let value = control.sample(&mut rng);
        writeln!(file, "control,{},{:.2}", i, value)?;
    }

    for i in 0..50 {
        let value = treatment.sample(&mut rng);
        writeln!(file, "treatment,{},{:.2}", i + 50, value)?;
    }

    println!("Generated experiment.csv");
    Ok(())
}
```

### 1.3 Load and Explore Data

```rust
use polars::prelude::*;

fn load_and_explore() -> Result<DataFrame, PolarsError> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("experiment.csv".into()))?
        .finish()?;

    println!("Shape: {:?}", df.shape());
    println!("\nFirst 5 rows:\n{}", df.head(Some(5)));
    println!("\nData types:\n{:?}", df.dtypes());
    println!("\nDescription:\n{}", df.describe(None)?);

    Ok(df)
}
```

---

## Task 2: Descriptive Statistics (20 min)

### 2.1 Group-wise Statistics

```rust
fn compute_group_stats(df: &DataFrame) -> Result<DataFrame, PolarsError> {
    let stats = df.clone().lazy()
        .group_by([col("group")])
        .agg([
            col("measurement").count().alias("n"),
            col("measurement").mean().alias("mean"),
            col("measurement").std(1).alias("std"),
            col("measurement").median().alias("median"),
            col("measurement").min().alias("min"),
            col("measurement").max().alias("max"),
            col("measurement").quantile(lit(0.25), QuantileMethod::Linear).alias("q25"),
            col("measurement").quantile(lit(0.75), QuantileMethod::Linear).alias("q75"),
        ])
        .collect()?;

    println!("Group Statistics:\n{}", stats);
    Ok(stats)
}
```

### 2.2 Custom Statistics Functions

```rust
/// Compute skewness of a sample
fn skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let m2: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3: f64 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;

    m3 / m2.powf(1.5)
}

/// Compute kurtosis (excess) of a sample
fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let m2: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4: f64 = data.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;

    m4 / m2.powi(2) - 3.0  // Excess kurtosis
}

/// Standard error of the mean
fn standard_error(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance: f64 = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);

    (variance / n).sqrt()
}

fn analyze_distribution(data: &[f64], label: &str) {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let std = standard_error(data) * (n as f64).sqrt();

    println!("\n{} Distribution Analysis:", label);
    println!("  N = {}", n);
    println!("  Mean = {:.2}", mean);
    println!("  Std Dev = {:.2}", std);
    println!("  Std Error = {:.4}", standard_error(data));
    println!("  Skewness = {:.4}", skewness(data));
    println!("  Kurtosis = {:.4}", kurtosis(data));
}
```

---

## Task 3: Distribution Fitting (20 min)

### 3.1 Fit Normal Distribution

```rust
use statrs::distribution::{Normal, ContinuousCDF, Continuous};

fn fit_normal(data: &[f64]) -> Normal {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance: f64 = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);

    Normal::new(mean, variance.sqrt()).unwrap()
}

fn goodness_of_fit(data: &[f64], dist: &Normal) {
    let n = data.len();
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Kolmogorov-Smirnov statistic
    let ks_stat: f64 = sorted.iter()
        .enumerate()
        .map(|(i, &x)| {
            let empirical = (i + 1) as f64 / n as f64;
            let theoretical = dist.cdf(x);
            (empirical - theoretical).abs()
        })
        .fold(0.0, f64::max);

    println!("Kolmogorov-Smirnov statistic: {:.4}", ks_stat);

    // Critical values (approximate)
    let alpha_05 = 1.36 / (n as f64).sqrt();  // α = 0.05
    println!("Critical value (α=0.05): {:.4}", alpha_05);

    if ks_stat < alpha_05 {
        println!("→ Cannot reject normal distribution hypothesis");
    } else {
        println!("→ Reject normal distribution hypothesis");
    }
}
```

### 3.2 Compare Distributions

```rust
use statrs::distribution::{Exponential, LogNormal, Gamma};

fn compare_distributions(data: &[f64]) {
    // Sort data for empirical CDF
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance: f64 = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);

    // Fit different distributions
    let normal = Normal::new(mean, variance.sqrt()).unwrap();

    // For lognormal, fit to log of data
    let log_data: Vec<f64> = data.iter().filter(|&&x| x > 0.0).map(|&x| x.ln()).collect();
    let log_mean = log_data.iter().sum::<f64>() / log_data.len() as f64;
    let log_var: f64 = log_data.iter()
        .map(|x| (x - log_mean).powi(2))
        .sum::<f64>() / (log_data.len() - 1) as f64;

    println!("\nDistribution Comparison:");
    println!("  Normal:    μ={:.2}, σ={:.2}", mean, variance.sqrt());
    println!("  LogNormal: μ_log={:.2}, σ_log={:.2}", log_mean, log_var.sqrt());
}
```

---

## Task 4: Hypothesis Testing (25 min)

### 4.1 Two-Sample t-Test

```rust
use statrs::distribution::{StudentsT, ContinuousCDF};

/// Welch's t-test (unequal variances)
fn welch_t_test(group1: &[f64], group2: &[f64]) -> (f64, f64, f64) {
    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;

    let mean1 = group1.iter().sum::<f64>() / n1;
    let mean2 = group2.iter().sum::<f64>() / n2;

    let var1: f64 = group1.iter()
        .map(|x| (x - mean1).powi(2))
        .sum::<f64>() / (n1 - 1.0);
    let var2: f64 = group2.iter()
        .map(|x| (x - mean2).powi(2))
        .sum::<f64>() / (n2 - 1.0);

    // t-statistic
    let t_stat = (mean1 - mean2) / (var1 / n1 + var2 / n2).sqrt();

    // Welch-Satterthwaite degrees of freedom
    let num = (var1 / n1 + var2 / n2).powi(2);
    let denom = (var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0);
    let df = num / denom;

    // Two-tailed p-value
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));

    (t_stat, df, p_value)
}

fn compare_groups(df: &DataFrame) -> Result<(), PolarsError> {
    // Extract data for each group
    let control: Vec<f64> = df.clone().lazy()
        .filter(col("group").eq(lit("control")))
        .select([col("measurement")])
        .collect()?
        .column("measurement")?
        .f64()?
        .into_no_null_iter()
        .collect();

    let treatment: Vec<f64> = df.clone().lazy()
        .filter(col("group").eq(lit("treatment")))
        .select([col("measurement")])
        .collect()?
        .column("measurement")?
        .f64()?
        .into_no_null_iter()
        .collect();

    let (t, df_val, p) = welch_t_test(&control, &treatment);

    println!("\nWelch's t-test Results:");
    println!("  t-statistic: {:.4}", t);
    println!("  Degrees of freedom: {:.2}", df_val);
    println!("  p-value: {:.4}", p);

    let alpha = 0.05;
    if p < alpha {
        println!("  → Significant difference at α={} (reject H₀)", alpha);
    } else {
        println!("  → No significant difference at α={} (fail to reject H₀)", alpha);
    }

    // Effect size (Cohen's d)
    let pooled_std = ((control.len() as f64 - 1.0) * variance(&control)
        + (treatment.len() as f64 - 1.0) * variance(&treatment))
        / (control.len() + treatment.len() - 2) as f64;
    let cohens_d = (mean(&treatment) - mean(&control)) / pooled_std.sqrt();

    println!("  Cohen's d: {:.4}", cohens_d);
    println!("  Effect size: {}",
        if cohens_d.abs() < 0.2 { "negligible" }
        else if cohens_d.abs() < 0.5 { "small" }
        else if cohens_d.abs() < 0.8 { "medium" }
        else { "large" }
    );

    Ok(())
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn variance(data: &[f64]) -> f64 {
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}
```

### 4.2 Confidence Intervals

```rust
fn confidence_interval(data: &[f64], confidence: f64) -> (f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let se = standard_error(data);

    let t_dist = StudentsT::new(0.0, 1.0, n - 1.0).unwrap();
    let alpha = 1.0 - confidence;
    let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);

    let margin = t_crit * se;
    (mean - margin, mean + margin)
}

fn report_confidence_intervals(df: &DataFrame) -> Result<(), PolarsError> {
    println!("\n95% Confidence Intervals:");

    for group_name in ["control", "treatment"] {
        let data: Vec<f64> = df.clone().lazy()
            .filter(col("group").eq(lit(group_name)))
            .select([col("measurement")])
            .collect()?
            .column("measurement")?
            .f64()?
            .into_no_null_iter()
            .collect();

        let (lower, upper) = confidence_interval(&data, 0.95);
        let mean = data.iter().sum::<f64>() / data.len() as f64;

        println!("  {}: {:.2} [{:.2}, {:.2}]", group_name, mean, lower, upper);
    }

    Ok(())
}
```

---

## Task 5: Visualization (10 min)

```rust
use plotters::prelude::*;

fn plot_distributions(control: &[f64], treatment: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("distributions.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left, right) = root.split_horizontally(400);

    // Histogram for control
    plot_histogram(&left, control, "Control Group", &BLUE)?;

    // Histogram for treatment
    plot_histogram(&right, treatment, "Treatment Group", &RED)?;

    root.present()?;
    println!("Saved distributions.png");
    Ok(())
}

fn plot_histogram(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    data: &[f64],
    title: &str,
    color: &RGBColor,
) -> Result<(), Box<dyn std::error::Error>> {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d((min..max).step(5.0).use_round().into_segmented(), 0..20)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(color.filled())
            .margin(1)
            .data(data.iter().map(|&x| (x, 1)))
    )?;

    Ok(())
}
```

---

## Homework

### Assignment: Monte Carlo Uncertainty Quantification

Simulate measurement uncertainty propagation:

1. Given: `y = f(x₁, x₂, x₃)` where each `xᵢ` has known distribution
2. Generate N samples from each input distribution
3. Compute output distribution through function
4. Report mean, std, confidence intervals of output
5. Compare with analytical propagation (if possible)

**Example function:** `y = x₁² + x₂*x₃` where `x₁ ~ N(10, 1)`, `x₂ ~ U(0, 5)`, `x₃ ~ Exp(0.1)`

**Submission:** Branch `homework-4-1`

---

## Summary

- ✅ Loaded and explored experimental data
- ✅ Computed descriptive statistics
- ✅ Fitted probability distributions
- ✅ Performed hypothesis testing (Welch's t-test)
- ✅ Calculated confidence intervals

## Next

Continue to [Seminar 4.2: Machine Learning for Scientific Data](./seminar-04-2.md)
