"""Input validation and preprocessing helpers for easycluster."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .exceptions import InputValidationError

_MISSING_STRATEGIES = {"drop", "mean", "median", "raise"}
_SCALERS = {"standard", "minmax", "robust", None}


def validate_dataframe(data) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise InputValidationError(
            f"`data` must be a pandas DataFrame, got {type(data).__name__}. "
            "Example: Cluster(data=my_dataframe)"
        )
    if data.empty:
        raise InputValidationError("The DataFrame you passed is empty. Nothing to cluster.")
    return data


def resolve_features(data: pd.DataFrame, features: Optional[Iterable[str]]) -> list:
    if features is None:
        resolved = list(data.columns)
    else:
        resolved = list(features)
        if len(resolved) == 0:
            raise InputValidationError("`features` was given as an empty list.")
        if len(set(resolved)) != len(resolved):
            dupes = {c for c in resolved if resolved.count(c) > 1}
            raise InputValidationError(f"Duplicate column names in `features`: {sorted(dupes)}")
        missing = [c for c in resolved if c not in data.columns]
        if missing:
            raise InputValidationError(
                f"These columns were not found in the DataFrame: {missing}. "
                f"Available columns: {list(data.columns)}"
            )
    if len(resolved) < 1:
        raise InputValidationError("At least one feature column is required.")
    return resolved


def check_numeric(data: pd.DataFrame, features: list) -> None:
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(data[c])]
    if non_numeric:
        raise InputValidationError(
            f"These selected columns are not numeric: {non_numeric}. "
            "easycluster does not silently convert categorical values to numbers. "
            "Either select only numeric columns via `features=[...]`, or encode "
            "these columns yourself first (e.g. pd.get_dummies) before passing them in."
        )


def handle_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy not in _MISSING_STRATEGIES:
        raise InputValidationError(
            f"`missing` must be one of {sorted(_MISSING_STRATEGIES)}, got {strategy!r}."
        )
    n_missing = int(df.isna().sum().sum())
    if n_missing == 0:
        return df
    if strategy == "raise":
        cols_with_na = df.columns[df.isna().any()].tolist()
        raise InputValidationError(
            f"Found {n_missing} missing value(s) in columns {cols_with_na}. "
            "Pass missing='drop', 'mean', or 'median' to handle them, or clean your data first."
        )
    if strategy == "drop":
        out = df.dropna()
        if out.empty:
            raise InputValidationError(
                "Dropping rows with missing values left an empty dataset. "
                "Try missing='mean' or 'median' instead."
            )
        return out
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    if strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    raise InputValidationError(f"Unhandled missing strategy: {strategy}")


def check_infinite(df: pd.DataFrame) -> None:
    arr = df.to_numpy(dtype=float, copy=False)
    if np.isinf(arr).any():
        bad_cols = df.columns[np.isinf(df.to_numpy(dtype=float)).any(axis=0)].tolist()
        raise InputValidationError(
            f"Found infinite values in columns {bad_cols}. "
            "easycluster will not silently drop or convert them. "
            "Please clean infinite values before clustering (e.g. df.replace([np.inf, -np.inf], np.nan))."
        )


def validate_min_samples(n_samples: int, n_clusters: int, context: str = "clustering") -> None:
    if n_clusters < 1:
        raise InputValidationError(f"`n_clusters` must be >= 1 for {context}, got {n_clusters}.")
    if n_samples < 2:
        raise InputValidationError(
            f"At least 2 samples are required for {context}, found {n_samples}."
        )
    if n_clusters > n_samples:
        raise InputValidationError(
            f"`n_clusters`={n_clusters} cannot exceed the number of samples ({n_samples})."
        )


def make_scaler(scale):
    """scale: True/False or one of 'standard','minmax','robust'. Returns a fitted-able scaler or None."""
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    if scale is False or scale is None:
        return None
    if scale is True:
        scale = "standard"
    if scale not in _SCALERS and scale not in {"standard", "minmax", "robust"}:
        raise InputValidationError(
            f"`scale` must be True, False, or one of 'standard'/'minmax'/'robust', got {scale!r}."
        )
    return {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[scale]()
