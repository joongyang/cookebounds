import numpy as np
from typing import Optional, Union, Dict
from .calibration import CookeEngine
from .exceptions import InvalidParameterError

class ThresholdEstimator:
    """
    Non-parametric threshold estimator using Extreme Value Theory.
    
    Provides an interface to calculate the minimum and maximum boundaries 
    of a distribution using basic, Jackknife, or weight-calibrated methods.
    """

    def __init__(self, k_min: int = 20, window_size: int = 5, tail_depth: float = 0.05):
        """
        Parameters
        ----------
        k_min : int
            The minimum order statistic bound for plateau detection.
        window_size : int
            The rolling variance window size (W) to isolate the stable region.
        tail_depth : float
            The target fractional depth of the tail (p) used to determine 
            the maximum search space (K_max) for weighted plateau detection.
        """
        self.engine = CookeEngine(k_min, window_size, tail_depth)

    def fit(self, y: Union[list, np.ndarray], weights: Optional[Union[list, np.ndarray]] = None, 
            method: str = 'weight_calibrated', direction: str = 'min', k: Optional[int] = None) -> Dict[str, float]:
        """
        Estimates the threshold(s) of the dataset.
        
        Parameters
        ----------
        y : array-like
            1D array of continuous focal variable data.
        weights : array-like, optional
            Array of survey weights/frequencies. Defaults to unweighted (1s).
        method : {'basic', 'jackknife', 'weight_calibrated'}
            The EVT estimation technique to apply.
        direction : {'min', 'max', 'both'}
            Whether to estimate the minimum threshold, maximum threshold, or both.
        k : int, optional
            Required if method is 'basic' or 'jackknife'. Overridden dynamically 
            if method is 'weight_calibrated'.
            
        Returns
        -------
        dict
            Dictionary containing the estimated threshold(s) and metadata.
        """
        # Validate and standardise inputs
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise InvalidParameterError("Input 'y' must be a 1D array.")
            
        if weights is None:
            weights = np.ones_like(y)
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != y.shape:
                raise InvalidParameterError("Dimensions of 'y' and 'weights' must match.")

        if method in ['basic', 'jackknife'] and k is None:
            raise InvalidParameterError(f"Parameter 'k' must be provided for method '{method}'.")
        if direction not in ['min', 'max', 'both']:
            raise InvalidParameterError("Direction must be 'min', 'max', or 'both'.")

        results = {}
        directions_to_run = ['min', 'max'] if direction == 'both' else [direction]
        
        for d in directions_to_run:
            # Exploit EVT symmetry: max bound is the inverted min bound of inverted data
            active_y = y if d == 'min' else -y
            
            if method == 'basic':
                est = self.engine.basic_cooke(active_y, k)
                results[f'{d}_threshold'] = est if d == 'min' else -est
                results[f'{d}_k_used'] = k
                
            elif method == 'jackknife':
                est = self.engine.jackknife_cooke(active_y, k)
                results[f'{d}_threshold'] = est if d == 'min' else -est
                results[f'{d}_k_used'] = k
                
            elif method == 'weight_calibrated':
                est, k_star = self.engine.weight_calibrated_min(active_y, weights)
                results[f'{d}_threshold'] = est if d == 'min' else -est
                results[f'{d}_k_used'] = k_star
                
            else:
                raise InvalidParameterError(f"Unknown method '{method}'")
                
        return results
