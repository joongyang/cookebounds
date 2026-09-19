import numpy as np
import pytest
from cookebounds.calibration import CookeEngine
from cookebounds.estimator import ThresholdEstimator
from cookebounds.exceptions import SampleTooSmallError, InvalidParameterError

class TestCookeEngine:
    """Unit tests for the pure mathematical engine."""
    
    @pytest.fixture
    def engine(self):
        # Use small parameters for testing speed
        return CookeEngine(k_min=3, window_size=2, tail_depth=0.5)

    def test_minimax_weights(self, engine):
        """Test if minimax weights sum to 1 and calculate correctly for k=3."""
        weights = engine._minimax_weights(k=3)
        # For k=3, j=[1, 2] -> 2*(3-j)/(3*2) -> [4/6, 2/6] -> [2/3, 1/3]
        np.testing.assert_allclose(weights, [2/3, 1/3])
        assert np.isclose(np.sum(weights), 1.0)
        
        with pytest.raises(ValueError):
            engine._minimax_weights(k=1)

    def test_basic_cooke_deterministic(self, engine):
        """Test the basic Cooke formula on a known array."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # k=3 -> y_k=[1, 2, 3], diffs=[1, 1], weights=[2/3, 1/3]
        # est = 1.0 - (1 * 2/3 + 1 * 1/3) = 0.0
        est = engine.basic_cooke(y, k=3)
        assert np.isclose(est, 0.0)

    def test_jackknife_cooke_deterministic(self, engine):
        """Test the Jackknife adjustment on a known array."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        est = engine.jackknife_cooke(y, k=3)
        # The Jackknife estimate should naturally fall below the sample minimum
        assert est < np.min(y)

    def test_sample_too_small_error(self, engine):
        """Ensure EVT estimators reject datasets smaller than k."""
        y = np.array([1.0, 2.0])
        with pytest.raises(SampleTooSmallError):
            engine.basic_cooke(y, k=5)
            
    def test_weight_calibrated_min(self, engine):
        """Test weight calibration and plateau detection with duplicated data."""
        # Create a dataset with heavy ties
        y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0])
        w = np.array([2, 1, 2, 3, 2, 1, 1, 1, 1])
        
        est, k_star = engine.weight_calibrated_min(y, w)
        assert est < np.min(y)
        assert k_star >= engine.k_min


class TestThresholdEstimator:
    """Unit tests for the API orchestrator and validation logic."""
    
    @pytest.fixture
    def orchestrator(self):
        return ThresholdEstimator(k_min=3, window_size=2, tail_depth=0.5)

    def test_input_validation(self, orchestrator):
        """Test that malformed inputs raise standard InvalidParameterErrors."""
        y_valid = np.array([1, 2, 3, 4, 5])
        
        # 1. 2D array
        with pytest.raises(InvalidParameterError, match="1D array"):
            orchestrator.fit(np.array([[1, 2], [3, 4]]))
            
        # 2. Mismatched weights
        with pytest.raises(InvalidParameterError, match="Dimensions"):
            orchestrator.fit(y_valid, weights=np.array([1, 2]))
            
        # 3. Missing k for basic method
        with pytest.raises(InvalidParameterError, match="Parameter 'k' must be provided"):
            orchestrator.fit(y_valid, method='basic')
            
        # 4. Unknown direction
        with pytest.raises(InvalidParameterError, match="Direction must be"):
            orchestrator.fit(y_valid, method='weight_calibrated', direction='diagonal')

    def test_direction_symmetry(self, orchestrator):
        """
        Verify that estimating the maximum of an array yields the exact symmetrical 
        bound as estimating the minimum of its inverted counterpart.
        """
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        y_inverted = -y
        
        # Min of y
        res_min = orchestrator.fit(y, method='basic', direction='min', k=4)
        min_bound = res_min['min_threshold']
        
        # Max of inverted y
        res_max = orchestrator.fit(y_inverted, method='basic', direction='max', k=4)
        max_bound = res_max['max_threshold']
        
        # They should be perfectly symmetrical
        assert np.isclose(min_bound, -max_bound)

    def test_both_directions(self, orchestrator):
        """Test returning both bounds simultaneously."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        res = orchestrator.fit(y, method='jackknife', direction='both', k=4)
        
        assert 'min_threshold' in res
        assert 'max_threshold' in res
        assert res['min_threshold'] < np.min(y)
        assert res['max_threshold'] > np.max(y)