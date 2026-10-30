import numpy as np
import pandas as pd
import pytest

from easycluster import Cluster
from easycluster.exceptions import InputValidationError, NotFittedError, ParameterError


def test_fit_kmeans_basic(blob_df):
    c = Cluster(data=blob_df)
    c.fit_kmeans(n_clusters=3, random_state=42)
    labels = c.get_labels()
    assert len(labels) == len(blob_df)
    assert set(labels.unique()) <= {0, 1, 2}


def test_kmeans_invalid_cluster_count(blob_df):
    c = Cluster(data=blob_df)
    with pytest.raises(InputValidationError):
        c.fit_kmeans(n_clusters=0)
    with pytest.raises(InputValidationError):
        c.fit_kmeans(n_clusters=1000)


def test_kmeans_reproducibility(blob_df):
    c1 = Cluster(data=blob_df).fit_kmeans(n_clusters=3, random_state=42)
    c2 = Cluster(data=blob_df).fit_kmeans(n_clusters=3, random_state=42)
    assert (c1.get_labels().to_numpy() == c2.get_labels().to_numpy()).all()


def test_elbow_method_returns_dataframe(blob_df):
    c = Cluster(data=blob_df)
    result = c.elbow_method(max_clusters=5, show=False)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["n_clusters", "wcss"]
    assert len(result) == 5
    assert result["wcss"].is_monotonic_decreasing


def test_elbow_method_validates_max_clusters(blob_df):
    c = Cluster(data=blob_df)
    with pytest.raises(InputValidationError):
        c.elbow_method(max_clusters=10_000, show=False)


def test_cluster_centers_shape(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    centers = c.cluster_centers()
    assert centers.shape == (3, 2)
    assert list(centers.columns) == list(blob_df.columns)


def test_cluster_centers_scaled_vs_original(blob_df):
    c = Cluster(data=blob_df, scale=True).fit_kmeans(n_clusters=3)
    centers_scaled_space = c._model.cluster_centers_
    centers_original = c.cluster_centers(original_scale=True).to_numpy()
    # inverse transform should differ from raw (scaled-space) centers
    assert not np.allclose(centers_scaled_space, centers_original)
    # but round-tripping should match the scaler's inverse_transform
    expected = c._scaler.inverse_transform(centers_scaled_space)
    assert np.allclose(centers_original, expected)


def test_inertia_available_after_fit(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    assert c.inertia() > 0


def test_predict_new_data(blob_df):
    c = Cluster(data=blob_df, scale=True).fit_kmeans(n_clusters=3, random_state=42)
    new_points = blob_df.iloc[:5].copy()
    preds = c.predict(new_points)
    assert len(preds) == 5
    assert preds.index.equals(new_points.index)


def test_predict_requires_fitted_model(blob_df):
    c = Cluster(data=blob_df)
    with pytest.raises(NotFittedError):
        c.predict(blob_df.iloc[:5])


def test_predict_missing_columns_rejected(blob_df):
    c = Cluster(data=blob_df).fit_kmeans(n_clusters=3)
    bad = pd.DataFrame({"Annual Income": [1, 2]})
    with pytest.raises(InputValidationError):
        c.predict(bad)


def test_methods_before_fit_raise(blob_df):
    c = Cluster(data=blob_df)
    with pytest.raises(NotFittedError):
        c.get_labels()
    with pytest.raises(NotFittedError):
        c.cluster_centers()
    with pytest.raises(NotFittedError):
        c.evaluate()
