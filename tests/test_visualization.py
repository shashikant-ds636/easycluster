import matplotlib

matplotlib.use("Agg")

import pytest

from clustereasy import Cluster
from clustereasy.exceptions import InputValidationError, NotFittedError


def test_plot_clusters_returns_none(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    result = c.plot_clusters(x="Annual Income", y="Spending Score", show=False)
    assert result is None


def test_plot_clusters_invalid_column(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    with pytest.raises(InputValidationError):
        c.plot_clusters(x="Annual Income", y="Nonexistent", show=False)


def test_plot_clusters_requires_fit(blob_df):
    c = Cluster(data=blob_df)
    with pytest.raises(NotFittedError):
        c.plot_clusters(x="Annual Income", y="Spending Score", show=False)


def test_plot_cluster_sizes_returns_none(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    result = c.plot_cluster_sizes(show=False)
    assert result is None


def test_plot_silhouette_returns_none(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    result = c.plot_silhouette(show=False)
    assert result is None


def test_plot_pca_3d_data_returns_none(blob_df_3d):
    c = Cluster(data=blob_df_3d).fit_kmeans(n_clusters=3)
    result = c.plot_pca(show=False)
    assert result is None


def test_elbow_plot_returns_none(blob_df):
    c = Cluster(data=blob_df)
    result = c.elbow_method(max_clusters=5, show=False)
    assert result is None


def test_dendrogram_plot_returns_none(blob_df):
    c = Cluster(data=blob_df)
    result = c.plot_dendrogram(show=False)
    assert result is None


def test_fit_methods_still_chain(blob_df):
    """fit_* methods return self, so fitting can be chained before plotting."""
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    assert isinstance(c, Cluster)
    c.plot_clusters(x="Annual Income", y="Spending Score", show=False)
    c.plot_cluster_sizes(show=False)
    c.plot_silhouette(show=False)
