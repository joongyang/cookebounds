import numpy as np
import warnings
from typing import Tuple
from .exceptions import SampleTooSmallError

class CookeEngine:
    """
    The mathematical engine executing Extreme Value Theory (EVT) threshold estimators.
    Strictly computes lower bounds (minimums).
    """

    def __init__(self, k_min: int = 20, window_size: int = 5, tail_depth: float = 0.05):
        self.k_min = k_min
        self.window_size = window_size
        self.tail_depth = tail_depth

    def _minimax_weights(self, k: int) -> np.ndarray:
        """
        Calculates minimax weights for Cooke's base estimator[cite: 1].
        """
        if k < 2:
            raise ValueError("k must be at least 2 to compute spacings.")
        j = np.arange(1, k)
        return 2 * (k - j) / (k * (k - 1))

    def basic_cooke(self, y: np.ndarray, k: int) -> float:
        """
        Calculates the base Cooke (1979) estimator using the spacing between 
        the lowest extreme order statistics[cite: 1].
        """
        y_sorted = np.sort(y)
        if len(y_sorted) < k:
            raise SampleTooSmallError(f"Sample size ({len(y_sorted)}) is smaller than k ({k}).")
            
        y_k = y_sorted[:k]
        diffs = np.diff(y_k)
        xi = self._minimax_weights(k)
        
        # Base estimator formula subtracting weighted spacings from the minimum[cite: 1]
        return float(y_k[0] - np.sum(xi * diffs))

    def jackknife_cooke(self, y: np.ndarray, k: int) -> float:
        """
        Calculates the Jackknife estimator to reduce location bias by iteratively 
        omitting clusters of extreme order statistics[cite: 1].
        """
        y_sorted = np.sort(y)
        if len(y_sorted) < k:
            raise SampleTooSmallError(f"Sample size ({len(y_sorted)}) is smaller than k ({k}).")
            
        y_k = y_sorted[:k]
        base_est = self.basic_cooke(y_k, k)
        
        leave_one_out_ests = np.zeros(k)
        for m in range(k):
            # Form subset by omitting the m-th order statistic[cite: 1]
            y_omitted = np.delete(y_k, m)
            leave_one_out_ests[m] = self.basic_cooke(y_omitted, k - 1)
            
        # Jackknife adjustment formula[cite: 1]
        return float(k * base_est - ((k - 1) / k) * np.sum(leave_one_out_ests))

    def weight_calibrated_min(self, y: np.ndarray, weights: np.ndarray) -> Tuple[float, int]:
        """
        Executes Algorithms 1, 2, and 3 to compute the weight-calibrated Jackknife estimate.
        Resolves zero-spacing issues in empirical data with duplicated frequencies.
        """
        sort_idx = np.argsort(y)
        y_sorted = y[sort_idx]
        w_sorted = weights[sort_idx]
        
        # Algorithm 1: Weight Calibration - identify unique support points[cite: 1]
        u, unique_indices = np.unique(y_sorted, return_inverse=True)
        w_u = np.bincount(unique_indices, weights=w_sorted)
        
        N = np.sum(w_u)
        target_weight = self.tail_depth * N
        cumsum_w = np.cumsum(w_u)
        
        valid_k = np.where(cumsum_w <= target_weight)[0]
        if len(valid_k) == 0 or (valid_k[-1] + 1) < (self.k_min + self.window_size):
            warnings.warn("Tail depth is too shallow for the window size. Adjusting K_max dynamically.")
            K_max = self.k_min + self.window_size + 5
        else:
            K_max = valid_k[-1] + 1 

        # Algorithm 2: Jackknife Estimation over unique support points[cite: 1]
        k_range = np.arange(self.k_min, K_max + 1)
        jackknife_vals = np.zeros(len(k_range))
        
        for idx, k in enumerate(k_range):
            jackknife_vals[idx] = self.jackknife_cooke(u, k)
            
        # Algorithm 3: Empirical Plateau Detection via rolling variance[cite: 1]
        variances = []
        n_windows = len(k_range) - self.window_size + 1
        
        for i in range(n_windows):
            window_vals = jackknife_vals[i : i + self.window_size]
            variances.append(np.var(window_vals))
            
        opt_idx = int(np.argmin(variances))
        optimal_window_k = k_range[opt_idx : opt_idx + self.window_size]
        
        # Optimal k* is the median of the lowest-variance window[cite: 1]
        k_star = int(np.median(optimal_window_k))
        k_star_idx = np.where(k_range == k_star)[0][0]
        
        return float(jackknife_vals[k_star_idx]), k_star
