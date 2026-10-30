import pytest

from easycluster import Cluster
from easycluster.exceptions import ParameterError


def test_fit_agglomerative_basic(blob_df):
    c = Cluster(data=blob_df).fit_agglomerative(n_clusters=3, linkage="ward")
    labels = c.get_labels()
    assert len(labels) == len(blob_df)
    assert set(labels.unique()) <= {0, 1, 2}


def test_ward_requires_euclidean(blob_df):
    c = Cluster(data=blob_df)
    with pytest.raises(ParameterError):
        c.fit_agglomerative(n_clusters=3, linkage="ward", metric="manhattan")


def test_non_ward_linkage_with_other_metric(blob_df):
    c = Cluster(data=blob_df).fit_agglomerative(n_clusters=3, linkage="average", metric="manhattan")
    assert len(c.get_labels()) == len(blob_df)


def test_invalid_linkage_rejected(blob_df):
    c = Cluster(data=blob_df)
    with pytest.raises(ParameterError):
        c.fit_agglomerative(n_clusters=3, linkage="bogus")


def test_distance_threshold_requires_n_clusters_none(blob_df):
    c = Cluster(data=blob_df)
    with pytest.raises(ParameterError):
        c.fit_agglomerative(n_clusters=3, distance_threshold=5.0)


def test_distance_threshold_mode_works(blob_df):
    c = Cluster(data=blob_df).fit_agglomerative(n_clusters=None, distance_threshold=5.0)
    labels = c.get_labels()
    assert len(labels) == len(blob_df)


def test_agglomerative_has_no_predict(blob_df):
    c = Cluster(data=blob_df).fit_agglomerative(n_clusters=3)
    with pytest.raises(ParameterError):
        c.predict(blob_df.iloc[:3])


def test_agglomerative_has_no_inertia(blob_df):
    c = Cluster(data=blob_df).fit_agglomerative(n_clusters=3)
    with pytest.raises(ParameterError):
        c.inertia()


def test_dendrogram_returns_linkage_matrix(blob_df):
    c = Cluster(data=blob_df)
    Z = c.plot_dendrogram(method="ward", show=False)
    assert Z.shape[0] == len(blob_df) - 1
    assert (c.get_linkage_matrix() == Z).all()


def test_dendrogram_ward_requires_euclidean(blob_df):
    c = Cluster(data=blob_df)
    with pytest.raises(ParameterError):
        c.plot_dendrogram(method="ward", metric="cosine", show=False)
