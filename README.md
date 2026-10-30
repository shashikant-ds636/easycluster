# easycluster

A beginner-friendly, consistent Python API that combines **K-Means** and
**Agglomerative (Hierarchical) Clustering** on pandas DataFrames into one
class: `Cluster`.

Instead of re-learning different scikit-learn/SciPy calling conventions for
each algorithm, you get one object with one set of methods for fitting,
visualizing, evaluating, and exporting results — for both algorithms.

## 1. Why easycluster

- One consistent interface for two different clustering algorithms.
- Keeps your original DataFrame untouched.
- Clear, explicit errors instead of silently doing the wrong thing.
- Advanced users can still pass through scikit-learn parameters.

## 2. Installation

Local development install (editable):

```bash
git clone https://github.com/your-username/easycluster.git
cd easycluster
pip install -e ".[dev]"
```

Once published, it will be installable with:

```bash
pip install easycluster
```

## 3. Requirements

- Python >= 3.9
- pandas >= 1.5
- numpy >= 1.23
- scikit-learn >= 1.2
- scipy >= 1.9
- matplotlib >= 3.6

## 4. Quick start

```python
import pandas as pd
from easycluster import Cluster

df = pd.read_csv("customers.csv")

cluster = Cluster(
    data=df,
    features=["Annual Income", "Spending Score"],
    scale=True,
    missing="median",
)

cluster.elbow_method(max_clusters=10)
cluster.fit_kmeans(n_clusters=4, random_state=42)
cluster.plot_clusters(x="Annual Income", y="Spending Score")

print(cluster.evaluate())
result_df = cluster.get_clustered_data(column_name="Segment")
```

## 5. DataFrame input requirements

`Cluster(data=df, features=[...], scale=..., missing=...)`:

- `data` must be a non-empty `pandas.DataFrame`. It is never modified.
- `features`: optional list of column names to cluster on. Defaults to all
  columns. Columns must exist, be unique, and be numeric — easycluster will
  **not** silently convert text/categorical values into numbers. Encode
  categorical columns yourself first (e.g. `pd.get_dummies`) if you want to
  include them.
- `missing`: one of `"raise"` (default), `"drop"`, `"mean"`, `"median"`.
  Controls how missing values in the selected features are handled.
- Infinite values always raise an explicit error — they are never silently
  dropped or converted.
- `scale`: `False` (default), `True` (same as `"standard"`), or one of
  `"standard"`, `"minmax"`, `"robust"`.

## 6. K-Means tutorial

```python
cluster = Cluster(data=df, features=["Income", "Score"], scale=True)

# Explore how many clusters might make sense
elbow_table = cluster.elbow_method(max_clusters=10, random_state=42)

# Fit
cluster.fit_kmeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)

cluster.cluster_centers()      # DataFrame, original feature scale by default
cluster.inertia()              # WCSS for the fitted model
cluster.predict(new_df)        # assign new rows to existing clusters
```

Supported `fit_kmeans` parameters: `n_clusters`, `init`, `n_init`, `max_iter`,
`tol`, `random_state`, `algorithm` — all pass straight through to
scikit-learn's `KMeans`.

## 7. Elbow Method

`elbow_method(max_clusters=10, random_state=42, show=True)` fits K-Means for
k = 1..max_clusters, plots k vs. WCSS (inertia), and returns a DataFrame of
the results so you can reuse them without refitting.

**The elbow is a heuristic**, not a guaranteed optimum — use it together with
`silhouette_score()` and your domain knowledge, not in isolation.

## 8. Hierarchical Clustering tutorial

```python
Z = cluster.plot_dendrogram(method="ward", metric="euclidean")
cluster.fit_agglomerative(n_clusters=3, linkage="ward")
```

Supported linkage methods: `ward`, `complete`, `average`, `single`.
`ward` linkage requires `metric="euclidean"` — using any other metric with
`ward` raises a clear error instead of silently producing wrong results.

You can also cluster by distance instead of a fixed cluster count:

```python
cluster.fit_agglomerative(n_clusters=None, distance_threshold=5.0)
```

`distance_threshold` and `n_clusters` are mutually exclusive, matching
scikit-learn's own requirement.

**Important:** `AgglomerativeClustering` has no native `predict()` method in
scikit-learn, and `Cluster.predict()` will raise an explicit error if called
on an agglomerative model rather than faking support for it.

## 9. Dendrogram interpretation

- Each leaf is one observation.
- Each join (inverted "U") is a merge of two clusters; its height is the
  distance at which they were merged.
- Cutting the dendrogram horizontally at a given height determines how many
  clusters you get at that distance.
- For large datasets, pass `truncate_mode="lastp"` with a small `p` to avoid
  an unreadable plot — easycluster will warn you above ~2000 samples if you
  don't.

## 10. Evaluation metrics and their limitations

All metrics below work without ground-truth labels:

| Method | Notes |
|---|---|
| `silhouette_score()` | Needs 2..n-1 distinct clusters. |
| `silhouette_samples()` | Per-sample values, same constraints. |
| `davies_bouldin_score()` | Lower is better. Needs >= 2 clusters. |
| `calinski_harabasz_score()` | Higher is better. Needs >= 2 clusters. |
| `inertia()` | K-Means only — not available for Agglomerative models. |
| `evaluate()` | One DataFrame with every metric applicable to the fitted model. |

These are **diagnostic tools**, not proof of a "correct" clustering — always
sanity-check results against domain knowledge. If a metric can't be computed
(e.g. only 1 distinct cluster), `evaluate()` shows `NaN` with a note instead
of crashing; the individual metric methods raise a `ParameterError`.

## 11. Feature scaling and missing-value handling

- Distances (and therefore both K-Means and Agglomerative results) are scale
  -sensitive. If your features are on different scales (e.g. income in
  thousands vs. a 1-100 score), set `scale=True`.
- K-Means cluster centers are available both in the transformed (scaled)
  space (`cluster._model.cluster_centers_`) and inverse-transformed back to
  the original feature scale (`cluster.cluster_centers()`, the default).
- Missing values are never silently dropped by default — you must choose
  `missing="drop"/"mean"/"median"` explicitly, or clean the data yourself.

## 12. Visualization examples

```python
cluster.plot_clusters(x="Income", y="Score", show_centers=True)
cluster.plot_cluster_sizes()
cluster.plot_silhouette()
cluster.plot_pca()   # for > 2 features; clearly labeled as a 2D projection
```

`plot_clusters` requires exactly two selected feature columns to display;
for higher-dimensional data use `plot_pca()` instead, which is explicitly
labeled as a projection, not the original feature space.

## 13. Cluster labels and DataFrame output

```python
labels = cluster.get_labels()                       # pd.Series, original index
clustered_df = cluster.get_clustered_data("Segment") # original df + label column
summary = cluster.cluster_summary()                  # size, proportion, per-feature mean
```

Cluster IDs (0, 1, 2, ...) are arbitrary identifiers — they carry no
inherent ranking or meaning.

## 14. Common errors and solutions

| Error | Likely cause | Fix |
|---|---|---|
| `InputValidationError: not numeric` | Selected a text/categorical column | Encode it first, or exclude it from `features` |
| `InputValidationError: missing values` | NaNs present, `missing="raise"` (default) | Pass `missing="drop"/"mean"/"median"` |
| `ParameterError: Ward linkage requires metric='euclidean'` | `linkage="ward"` with another metric | Use `metric="euclidean"` or a different linkage |
| `NotFittedError` | Called an evaluation/plot/output method before `fit_kmeans`/`fit_agglomerative` | Fit a model first |
| `ParameterError` on `predict()` | Called `predict()` on an Agglomerative model | Not supported — only K-Means supports `predict()` |

## 15. Computational limitations

- Dendrograms require O(n²) memory/time for the distance matrix and are
  impractical much past a few thousand samples.
- `AgglomerativeClustering` scales worse than `KMeans` for large n.
- No GPU acceleration; this wraps scikit-learn/SciPy CPU implementations.

## 16. API reference (summary)

**Construction:** `Cluster(data, features=None, scale=False, missing="raise")`

**K-Means:** `elbow_method`, `fit_kmeans`, `cluster_centers`, `inertia`, `predict`

**Agglomerative:** `plot_dendrogram`, `get_linkage_matrix`, `fit_agglomerative`

**Shared:** `plot_clusters`, `plot_cluster_sizes`, `plot_silhouette`, `plot_pca`

**Evaluation:** `silhouette_score`, `silhouette_samples`, `davies_bouldin_score`,
`calinski_harabasz_score`, `evaluate`

**Output:** `get_labels`, `get_clustered_data`, `cluster_summary`, `elbow_results`

## 17. Testing

```bash
pip install -e ".[dev]"
pytest -q
```

Current suite: **59 tests, all passing**, covering validation, both
algorithms, visualization, evaluation edge cases, and DataFrame-output
guarantees (index preservation, immutability, reproducibility).

## 18. Contribution guidelines

1. Fork, branch from `main`.
2. Add/update tests for any behavior change.
3. Run `pytest -q` before opening a PR.
4. Keep public methods documented with docstrings.

## 19. Roadmap

- `DBSCAN`, `OPTICS`, `GaussianMixture` support behind the same API.
- Automatic comparison across multiple `k` / linkage configurations.
- Model save/load (`joblib`).
- Export results directly to CSV.
- Optional connectivity-constrained agglomerative clustering helpers.

## 20. Preparing a PyPI release

```bash
pip install build twine
python -m build                 # creates dist/*.whl and dist/*.tar.gz
twine check dist/*
pip install dist/easycluster-0.1.0-py3-none-any.whl  # test locally
twine upload --repository testpypi dist/*             # TestPyPI first
twine upload dist/*                                    # then real PyPI
```

Bump the `version` in `pyproject.toml` before each release and update
`CHANGELOG.md`.

## License

MIT — see `LICENSE`.
