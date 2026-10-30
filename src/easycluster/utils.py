"""Small shared utilities. Kept minimal on purpose for V1."""

from __future__ import annotations

import pandas as pd


def to_series(labels, index, name="Cluster") -> pd.Series:
    """Wrap a raw label array as a Series with the given index, consistently."""
    return pd.Series(labels, index=index, name=name)
