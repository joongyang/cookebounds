# cookebounds

`cookebounds` is a Python package for non-parametric estimation of finite distribution bounds (minimum and maximum thresholds) based on Extreme Value Theory (EVT). 

It implements the base spacing-estimator developed by Cooke (1979) alongside a novel **weight-calibrated Jackknife estimator** designed to handle complex survey weights, heavy-tailed distributions, and tied observations common in empirical microdata.

## Features
* **Basic Cooke Estimator:** Non-parametric bounds using minimax-optimized spacings of extreme order statistics.
* **Jackknife Cooke Estimator:** Reduces location bias in extreme order statistics by iteratively omitting clusters.
* **Weight-Calibrated Jackknife:** Resolves the zero-spacing problem caused by duplicated frequencies in weighted survey data. Includes an empirical plateau detection algorithm to dynamically select the optimal truncation parameter (k*).
* **Bidirectional:** Estimate the minimum bound, maximum bound, or both simultaneously.

## Installation
```bash
pip install cookebounds
