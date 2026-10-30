"""Core Cluster class: a consistent, beginner-friendly interface over
scikit-learn's KMeans and AgglomerativeClustering."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import warnings

from . import evaluation as _eval
from . import visualization as _viz
from .exceptions import InputValidationError, NotFittedError, ParameterError
from .preprocessing import (
    check_infinite,
    check_numeric,
    handle_missing,
    make_scaler,
    resolve_features,
    validate_dataframe,
    validate_min_samples,
)

_VALID_LINKAGE = {"ward", "complete", "average", "single"}
_WARD_REQUIRES_EUCLIDEAN = {"ward"}


class Cluster:
    """A consistent interface for K-Means and Agglomerative clustering on a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input data. Never modified in place.
    features : list of str, optional
        Columns to use for clustering. Defaults to all columns.
    scale : bool or {'standard', 'minmax', 'robust'}, default False
        Whether/how to scale features before clustering. True == 'standard'.
    missing : {'raise', 'drop', 'mean', 'median'}, default 'raise'
        How to handle missing values in the selected features.

    Notes
    -----
    Scaling changes distances, and therefore can change which clusters are
    found. K-Means cluster centers are stored in both the transformed
    (scaled) space and the original feature space (inverse-transformed).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[Iterable[str]] = None,
        scale=False,
        missing: str = "raise",
    ):
        data = validate_dataframe(data)
        self.data = data  # original, untouched
        self.features = resolve_features(data, features)
        check_numeric(data, self.features)

        feature_df = data[self.features]
        feature_df = handle_missing(feature_df, missing)
        check_infinite(feature_df)

        self._index = feature_df.index  # may be a subset if rows were dropped
        self._raw_features = feature_df.copy()  # original-scale, post-missing-handling

        self.scale = scale
        self._scaler = make_scaler(scale)
        if self._scaler is not None:
            self._X = self._scaler.fit_transform(feature_df.to_numpy(dtype=float))
        else:
            self._X = feature_df.to_numpy(dtype=float)

        # Fitted-model state
        self._fitted_algorithm: Optional[str] = None
        self._model = None
        self._labels: Optional[np.ndarray] = None
        self._elbow_results: Optional[pd.DataFrame] = None
        self._linkage_matrix = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _require_fitted(self):
        if self._labels is None:
            raise NotFittedError(
                "No model has been fitted yet. Call fit_kmeans(...) or "
                "fit_agglomerative(...) first."
            )

    def _centers_original_scale(self, centers_transformed: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            return self._scaler.inverse_transform(centers_transformed)
        return centers_transformed

    # ------------------------------------------------------------------ #
    # K-Means
    # ------------------------------------------------------------------ #
    def elbow_method(self, max_clusters: int = 10, random_state: int = 42, show: bool = True) -> pd.DataFrame:
        """Compute and plot WCSS (inertia) for k = 1..max_clusters.

        The elbow point is a heuristic for choosing k, not a guaranteed
        optimum — use it alongside silhouette_score() and domain knowledge.
        """
        from sklearn.cluster import KMeans

        n_samples = self._X.shape[0]
        if max_clusters < 2:
            raise ParameterError("`max_clusters` must be at least 2.")
        if max_clusters > n_samples:
            raise InputValidationError(
                f"`max_clusters`={max_clusters} exceeds the number of samples ({n_samples})."
            )

        ks = list(range(1, max_clusters + 1))
        wcss = []
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            km.fit(self._X)
            wcss.append(km.inertia_)

        self._elbow_results = pd.DataFrame({"n_clusters": ks, "wcss": wcss})
        _viz.plot_elbow(ks, wcss, show=show)
        return self._elbow_results

    def fit_kmeans(
        self,
        n_clusters: int = 3,
        init: str = "k-means++",
        n_init=10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
        algorithm: str = "lloyd",
    ) -> "Cluster":
        """Fit K-Means with explicit, reproducible defaults. Returns self for chaining."""
        from sklearn.cluster import KMeans

        validate_min_samples(self._X.shape[0], n_clusters, context="K-Means")

        km = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            algorithm=algorithm,
        )
        labels = km.fit_predict(self._X)

        self._model = km
        self._labels = labels
        self._fitted_algorithm = "kmeans"
        return self

    def cluster_centers(self, original_scale: bool = True) -> pd.DataFrame:
        """Return K-Means cluster centers. Only available after fit_kmeans()."""
        self._require_fitted()
        if self._fitted_algorithm != "kmeans":
            raise ParameterError("cluster_centers() is only available for K-Means models.")
        centers = self._model.cluster_centers_
        if original_scale:
            centers = self._centers_original_scale(centers)
        return pd.DataFrame(centers, columns=self.features)

    def inertia(self) -> float:
        self._require_fitted()
        if self._fitted_algorithm != "kmeans":
            raise ParameterError(
                "inertia() is a K-Means-specific attribute; Agglomerative Clustering "
                "does not provide it natively."
            )
        return float(self._model.inertia_)

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        """Predict cluster labels for new observations. K-Means only.

        AgglomerativeClustering has no native predict() in scikit-learn, so
        this method is intentionally unavailable for agglomerative models.
        """
        self._require_fitted()
        if self._fitted_algorithm != "kmeans":
            raise ParameterError(
                "predict() is only supported for K-Means. scikit-learn's "
                "AgglomerativeClustering has no native predict() method, so "
                "no approximation is offered here to avoid implying it is a "
                "true model prediction."
            )
        missing_cols = [c for c in self.features if c not in new_data.columns]
        if missing_cols:
            raise InputValidationError(f"`new_data` is missing required columns: {missing_cols}")
        X_new = new_data[self.features].to_numpy(dtype=float)
        if self._scaler is not None:
            X_new = self._scaler.transform(X_new)
        preds = self._model.predict(X_new)
        return pd.Series(preds, index=new_data.index, name="Cluster")

    # ------------------------------------------------------------------ #
    # Hierarchical / Agglomerative
    # ------------------------------------------------------------------ #
    def plot_dendrogram(self, method: str = "ward", metric: str = "euclidean", truncate_mode=None, p: int = 5, show: bool = True):
        """Plot a dendrogram and return the linkage matrix."""
        import scipy.cluster.hierarchy as sch

        if method not in _VALID_LINKAGE:
            raise ParameterError(f"`method` must be one of {sorted(_VALID_LINKAGE)}, got {method!r}.")
        if method in _WARD_REQUIRES_EUCLIDEAN and metric != "euclidean":
            raise ParameterError("Ward linkage requires metric='euclidean'.")

        n_samples = self._X.shape[0]
        if n_samples > 2000 and truncate_mode is None:
            warnings.warn(
                f"Computing a full dendrogram for {n_samples} samples can be slow and produce "
                "an unreadable plot. Consider truncate_mode='lastp' with a small `p`.",
                stacklevel=2,
            )

        Z = sch.linkage(self._X, method=method, metric=metric)
        self._linkage_matrix = Z
        _viz.plot_dendrogram_from_linkage(Z, truncate_mode=truncate_mode, p=p, show=show)
        return Z

    def get_linkage_matrix(self):
        if self._linkage_matrix is None:
            raise NotFittedError("Call plot_dendrogram(...) first to compute the linkage matrix.")
        return self._linkage_matrix

    def fit_agglomerative(
        self,
        n_clusters: Optional[int] = 3,
        linkage: str = "ward",
        metric: str = "euclidean",
        distance_threshold: Optional[float] = None,
        connectivity=None,
        compute_distances: bool = False,
    ) -> "Cluster":
        """Fit Agglomerative Clustering. Returns self for chaining.

        Note: scikit-learn's AgglomerativeClustering has no native predict();
        use Cluster.predict() only after fit_kmeans().
        """
        from sklearn.cluster import AgglomerativeClustering

        if linkage not in _VALID_LINKAGE:
            raise ParameterError(f"`linkage` must be one of {sorted(_VALID_LINKAGE)}, got {linkage!r}.")
        if linkage in _WARD_REQUIRES_EUCLIDEAN and metric != "euclidean":
            raise ParameterError("Ward linkage requires metric='euclidean'.")

        if distance_threshold is not None:
            if n_clusters is not None:
                raise ParameterError(
                    "Set only one of `n_clusters` or `distance_threshold`, not both "
                    "(scikit-learn requires n_clusters=None when distance_threshold is set)."
                )
            compute_distances = True
        else:
            if n_clusters is None:
                raise ParameterError("`n_clusters` must be set when `distance_threshold` is not used.")
            validate_min_samples(self._X.shape[0], n_clusters, context="Agglomerative Clustering")

        ag = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
            distance_threshold=distance_threshold,
            connectivity=connectivity,
            compute_distances=compute_distances,
        )
        labels = ag.fit_predict(self._X)

        self._model = ag
        self._labels = labels
        self._fitted_algorithm = "agglomerative"
        return self

    # ------------------------------------------------------------------ #
    # Shared visualization
    # ------------------------------------------------------------------ #
    def plot_clusters(self, x: str, y: str, show_centers: bool = True, title=None, figsize=(7, 5), show: bool = True):
        self._require_fitted()
        if x not in self.features or y not in self.features:
            raise InputValidationError(
                f"'{x}' and '{y}' must both be in the features used for clustering: {self.features}"
            )
        centers_original = None
        if show_centers and self._fitted_algorithm == "kmeans":
            centers_df = self.cluster_centers(original_scale=True)
            centers_original = centers_df[[x, y]].to_numpy()
        return _viz.plot_cluster_scatter(
            self._raw_features,
            x,
            y,
            self._labels,
            centers_original=centers_original,
            show_centers=show_centers,
            title=title,
            figsize=figsize,
            show=show,
        )

    def plot_cluster_sizes(self, show: bool = True):
        self._require_fitted()
        sizes = pd.Series(self._labels).value_counts()
        return _viz.plot_cluster_sizes(sizes, show=show)

    def plot_silhouette(self, show: bool = True):
        self._require_fitted()
        return _viz.plot_silhouette(self._X, self._labels, show=show)

    def plot_pca(self, show: bool = True):
        self._require_fitted()
        return _viz.plot_pca_projection(self._X, self._labels, show=show)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def silhouette_score(self) -> float:
        self._require_fitted()
        return _eval.safe_silhouette_score(self._X, self._labels)

    def silhouette_samples(self) -> pd.Series:
        self._require_fitted()
        vals = _eval.safe_silhouette_samples(self._X, self._labels)
        return pd.Series(vals, index=self._index, name="silhouette")

    def davies_bouldin_score(self) -> float:
        self._require_fitted()
        return _eval.safe_davies_bouldin(self._X, self._labels)

    def calinski_harabasz_score(self) -> float:
        self._require_fitted()
        return _eval.safe_calinski_harabasz(self._X, self._labels)

    def evaluate(self) -> pd.DataFrame:
        """Return a DataFrame of applicable metrics for the fitted model.

        These metrics are diagnostic tools for comparing clustering
        configurations, not proof that a given clustering is 'correct'.
        """
        self._require_fitted()
        inertia = self._model.inertia_ if self._fitted_algorithm == "kmeans" else None
        return _eval.build_evaluate_table(self._X, self._labels, self._fitted_algorithm, inertia=inertia)

    # ------------------------------------------------------------------ #
    # Results / DataFrame integration
    # ------------------------------------------------------------------ #
    def get_labels(self) -> pd.Series:
        self._require_fitted()
        return pd.Series(self._labels, index=self._index, name="Cluster")

    def get_clustered_data(self, column_name: str = "Cluster") -> pd.DataFrame:
        """Return a copy of the original data with a cluster-label column added.

        Note: cluster IDs are arbitrary identifiers with no inherent order
        or meaning (cluster 0 is not "better" than cluster 1).
        """
        self._require_fitted()
        out = self.data.loc[self._index].copy()
        out[column_name] = self._labels
        return out

    def cluster_summary(self, agg: str = "mean") -> pd.DataFrame:
        """Return cluster sizes, proportions, and per-feature aggregates (mean/median)."""
        self._require_fitted()
        if agg not in {"mean", "median"}:
            raise ParameterError("`agg` must be 'mean' or 'median'.")

        labels = pd.Series(self._labels, index=self._index, name="Cluster")
        sizes = labels.value_counts().sort_index()
        proportions = (sizes / sizes.sum()).rename("proportion")

        feat_df = self._raw_features.copy()
        feat_df["Cluster"] = labels
        agg_table = feat_df.groupby("Cluster").agg(agg)

        summary = agg_table.copy()
        summary.insert(0, "size", sizes)
        summary.insert(1, "proportion", proportions)
        return summary

    def elbow_results(self) -> pd.DataFrame:
        if self._elbow_results is None:
            raise NotFittedError("Call elbow_method(...) first.")
        return self._elbow_results

    def __repr__(self) -> str:
        fitted = self._fitted_algorithm or "not fitted"
        return f"Cluster(n_samples={self._X.shape[0]}, n_features={len(self.features)}, model={fitted})"
