"""
Custom exceptions for the cookebounds package.
"""

class CookeBoundsError(Exception):
    """Base exception class for cookebounds errors."""
    pass

class SampleTooSmallError(CookeBoundsError):
    """Raised when the sample size is smaller than the required order statistics (k)."""
    pass

class InvalidParameterError(CookeBoundsError):
    """Raised when an invalid parameter is passed to the estimator."""
    pass