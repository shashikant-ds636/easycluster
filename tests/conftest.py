import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def blob_df():
    """Small synthetic 2D dataset with 3 well-separated blobs."""
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=60, centers=3, n_features=2, random_state=42, cluster_std=0.6)
    return pd.DataFrame(X, columns=["Annual Income", "Spending Score"])


@pytest.fixture
def blob_df_3d():
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=50, centers=3, n_features=3, random_state=0, cluster_std=0.6)
    return pd.DataFrame(X, columns=["a", "b", "c"])
