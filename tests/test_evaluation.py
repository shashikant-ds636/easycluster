import numpy as np
import pandas as pd
import pytest

from clustereasy import Cluster
from clustereasy.exceptions import ParameterError


def test_silhouette_score_range(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3, random_state=42)
    score = c.silhouette_score()
    assert -1.0 <= score <= 1.0


def test_silhouette_samples_length(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    vals = c.silhouette_samples()
    assert len(vals) == len(blob_df)
    assert vals.index.equals(c.get_labels().index)


def test_davies_bouldin_and_calinski(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    assert c.davies_bouldin_score() >= 0
    assert c.calinski_harabasz_score() > 0


def test_single_cluster_metrics_raise(blob_df):
    c = Cluster(data=blob_df).fit_agglomerative(n_clusters=1)
    with pytest.raises(ParameterError):
        c.silhouette_score()
    with pytest.raises(ParameterError):
        c.davies_bouldin_score()
    with pytest.raises(ParameterError):
        c.calinski_harabasz_score()


def test_evaluate_kmeans_includes_inertia(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    table = c.evaluate()
    assert "inertia" in table["metric"].values
    assert "silhouette_score" in table["metric"].values


def test_evaluate_agglomerative_excludes_inertia(blob_df):
    c = Cluster(data=blob_df).fit_agglomerative(n_clusters=3)
    table = c.evaluate()
    assert "inertia" not in table["metric"].values


def test_evaluate_degenerate_case_has_nan_not_crash(blob_df):
    c = Cluster(data=blob_df).fit_agglomerative(n_clusters=1)
    table = c.evaluate()
    assert table.loc[table["metric"] == "silhouette_score", "value"].isna().all()
