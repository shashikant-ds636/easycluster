"""easycluster: a beginner-friendly, consistent API for K-Means and
Agglomerative Clustering on pandas DataFrames."""

from .cluster import Cluster
from .exceptions import EasyClusterError, InputValidationError, NotFittedError, ParameterError

__version__ = "0.1.1"

__all__ = [
    "Cluster",
    "EasyClusterError",
    "InputValidationError",
    "NotFittedError",
    "ParameterError",
]
