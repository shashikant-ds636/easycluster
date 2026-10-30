import matplotlib

matplotlib.use("Agg")

import pytest

from easycluster import Cluster
from easycluster.exceptions import InputValidationError, NotFittedError


def test_plot_clusters_default_columns(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    ax = c.plot_clusters(x="Annual Income", y="Spending Score", show=False)
    assert ax is not None


def test_plot_clusters_invalid_column(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    with pytest.raises(InputValidationError):
        c.plot_clusters(x="Annual Income", y="Nonexistent", show=False)


def test_plot_clusters_requires_fit(blob_df):
    c = Cluster(data=blob_df)
    with pytest.raises(NotFittedError):
        c.plot_clusters(x="Annual Income", y="Spending Score", show=False)


def test_plot_cluster_sizes(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    ax = c.plot_cluster_sizes(show=False)
    assert ax is not None


def test_plot_silhouette(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    ax = c.plot_silhouette(show=False)
    assert ax is not None


def test_plot_pca_3d_data(blob_df_3d):
    c = Cluster(data=blob_df_3d).fit_kmeans(n_clusters=3)
    ax = c.plot_pca(show=False)
    assert ax is not None


def test_elbow_plot_runs(blob_df):
    c = Cluster(data=blob_df)
    c.elbow_method(max_clusters=5, show=False)


def test_dendrogram_plot_runs(blob_df):
    c = Cluster(data=blob_df)
    c.plot_dendrogram(show=False)
