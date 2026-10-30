"""Evaluation metrics for clustering results (no ground-truth labels needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score

from .exceptions import ParameterError


def _n_distinct(labels: np.ndarray) -> int:
    return len(set(labels.tolist()))


def safe_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    k = _n_distinct(labels)
    if k < 2 or k > len(labels) - 1:
        raise ParameterError(
            f"Silhouette score requires between 2 and n_samples-1 distinct clusters; "
            f"found {k} distinct cluster(s) across {len(labels)} samples."
        )
    return float(silhouette_score(X, labels))


def safe_silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    k = _n_distinct(labels)
    if k < 2 or k > len(labels) - 1:
        raise ParameterError(
            "Per-sample silhouette values require between 2 and n_samples-1 distinct clusters."
        )
    return silhouette_samples(X, labels)


def safe_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    if _n_distinct(labels) < 2:
        raise ParameterError("Davies-Bouldin index requires at least 2 distinct clusters.")
    return float(davies_bouldin_score(X, labels))


def safe_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    if _n_distinct(labels) < 2:
        raise ParameterError("Calinski-Harabasz index requires at least 2 distinct clusters.")
    return float(calinski_harabasz_score(X, labels))


def build_evaluate_table(X: np.ndarray, labels: np.ndarray, algorithm: str, inertia=None) -> pd.DataFrame:
    rows = []

    def _try(name, fn):
        try:
            rows.append({"metric": name, "value": fn()})
        except ParameterError as exc:
            rows.append({"metric": name, "value": np.nan, "note": str(exc)})

    _try("silhouette_score", lambda: safe_silhouette_score(X, labels))
    _try("davies_bouldin_score", lambda: safe_davies_bouldin(X, labels))
    _try("calinski_harabasz_score", lambda: safe_calinski_harabasz(X, labels))

    if algorithm == "kmeans" and inertia is not None:
        rows.append({"metric": "inertia", "value": float(inertia)})

    df = pd.DataFrame(rows)
    if "note" not in df.columns:
        df["note"] = None
    return df
