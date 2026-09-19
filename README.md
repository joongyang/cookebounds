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
```

# Quick Start

```python
import numpy as np
from cookebounds import ThresholdEstimator

# Generate empirical data with a strict minimum threshold of 0.5
y = np.random.weibull(a=1.5, size=1000) * 10 + 0.5
weights = np.random.randint(1, 10, size=1000)

estimator = ThresholdEstimator(k_min=20, window_size=5, tail_depth=0.05)

# Calculate weight-calibrated bounds
results = estimator.fit(y, weights=weights, method='weight_calibrated', direction='min')

print(f"Estimated Minimum Threshold: {results['min_threshold']:.4f}")
print(f"Optimal Truncation (k*): {results['min_k_used']}")
```

# References

Cooke, P. (1979). Statistical inference for bounds of random variables. Biometrika, 66(2), 367-374.

