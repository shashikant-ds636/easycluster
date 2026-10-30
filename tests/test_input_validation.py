import numpy as np
import pandas as pd
import pytest

from easycluster import Cluster
from easycluster.exceptions import InputValidationError


def test_valid_dataframe_constructs(blob_df):
    c = Cluster(data=blob_df)
    assert c.features == list(blob_df.columns)
    assert c.data.shape == blob_df.shape


def test_non_dataframe_rejected():
    with pytest.raises(InputValidationError):
        Cluster(data=[[1, 2], [3, 4]])


def test_empty_dataframe_rejected():
    with pytest.raises(InputValidationError):
        Cluster(data=pd.DataFrame())


def test_missing_feature_column_rejected(blob_df):
    with pytest.raises(InputValidationError):
        Cluster(data=blob_df, features=["Annual Income", "Nonexistent"])


def test_duplicate_feature_columns_rejected(blob_df):
    with pytest.raises(InputValidationError):
        Cluster(data=blob_df, features=["Annual Income", "Annual Income"])


def test_non_numeric_feature_rejected(blob_df):
    df = blob_df.copy()
    df["category"] = ["a", "b"] * 30
    with pytest.raises(InputValidationError):
        Cluster(data=df, features=["Annual Income", "category"])


def test_non_numeric_column_ignored_if_not_selected(blob_df):
    df = blob_df.copy()
    df["category"] = ["a", "b"] * 30
    c = Cluster(data=df, features=["Annual Income", "Spending Score"])
    assert "category" not in c.features


def test_missing_values_raise_by_default(blob_df):
    df = blob_df.copy()
    df.iloc[0, 0] = np.nan
    with pytest.raises(InputValidationError):
        Cluster(data=df, missing="raise")


def test_missing_values_drop(blob_df):
    df = blob_df.copy()
    df.iloc[0, 0] = np.nan
    c = Cluster(data=df, missing="drop")
    assert c._X.shape[0] == len(df) - 1


def test_missing_values_mean(blob_df):
    df = blob_df.copy()
    df.iloc[0, 0] = np.nan
    c = Cluster(data=df, missing="mean")
    assert c._X.shape[0] == len(df)
    assert not np.isnan(c._X).any()


def test_missing_values_median(blob_df):
    df = blob_df.copy()
    df.iloc[0, 0] = np.nan
    c = Cluster(data=df, missing="median")
    assert not np.isnan(c._X).any()


def test_invalid_missing_strategy_rejected(blob_df):
    with pytest.raises(InputValidationError):
        Cluster(data=blob_df, missing="bogus")


def test_infinite_values_rejected(blob_df):
    df = blob_df.copy()
    df.iloc[0, 0] = np.inf
    with pytest.raises(InputValidationError):
        Cluster(data=df)


def test_original_dataframe_not_mutated(blob_df):
    original = blob_df.copy()
    c = Cluster(data=blob_df, scale=True)
    c.fit_kmeans(n_clusters=3)
    pd.testing.assert_frame_equal(blob_df, original)


def test_empty_features_list_rejected(blob_df):
    with pytest.raises(InputValidationError):
        Cluster(data=blob_df, features=[])
