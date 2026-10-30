import numpy as np
import pandas as pd
import pytest

from easycluster import Cluster


def test_get_labels_preserves_index(blob_df):
    df = blob_df.copy()
    df.index = [f"row_{i}" for i in range(len(df))]
    c = Cluster(data=df).fit_kmeans(n_clusters=3)
    labels = c.get_labels()
    assert list(labels.index) == list(df.index)


def test_get_clustered_data_preserves_original_columns(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    out = c.get_clustered_data()
    for col in blob_df.columns:
        assert col in out.columns
    assert "Cluster" in out.columns
    assert len(out) == len(blob_df)


def test_get_clustered_data_custom_column_name(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    out = c.get_clustered_data(column_name="segment")
    assert "segment" in out.columns
    assert "Cluster" not in out.columns


def test_get_clustered_data_does_not_mutate_original(blob_df):
    original_cols = list(blob_df.columns)
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    c.get_clustered_data()
    assert list(blob_df.columns) == original_cols


def test_cluster_summary_sizes_sum_to_total(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    summary = c.cluster_summary()
    assert summary["size"].sum() == len(blob_df)
    assert np.isclose(summary["proportion"].sum(), 1.0)


def test_cluster_summary_median_agg(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    summary = c.cluster_summary(agg="median")
    assert "Annual Income" in summary.columns


def test_index_preserved_with_dropped_missing_rows(blob_df):
    df = blob_df.copy()
    df.iloc[0, 0] = np.nan
    c = Cluster(data=df, missing="drop").fit_kmeans(n_clusters=3)
    labels = c.get_labels()
    assert 0 not in labels.index  # row 0 was dropped
    assert len(labels) == len(df) - 1
