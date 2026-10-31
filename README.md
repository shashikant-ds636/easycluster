<div align="center">

<img src="https://raw.githubusercontent.com/shashikant-ds636/easycluster/main/docs/assets/logo.png" alt="clustereasy logo" width="520"/>

<h3>v0.1.1</h3>

<p>
One consistent, beginner-friendly API for <strong>K-Means</strong> and <strong>Agglomerative (Hierarchical) Clustering</strong> on pandas DataFrames.
</p>

[![PyPI version](https://img.shields.io/pypi/v/clustereasy.svg?color=6366F1)](https://pypi.org/project/clustereasy/)
[![Python versions](https://img.shields.io/pypi/pyversions/clustereasy.svg)](https://pypi.org/project/clustereasy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/shashikant-ds636/easycluster/actions/workflows/tests.yml/badge.svg)](https://github.com/shashikant-ds636/easycluster/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shashikant-ds636/easycluster/blob/main/examples/clustereasy_quickstart.ipynb)

<p>
<a href="#-installation">Install</a> ·
<a href="#-quick-start">Quick start</a> ·
<a href="#-run-it-in">Run it in</a> ·
<a href="#-documentation">Documentation</a> ·
<a href="#-api-reference">API reference</a>
</p>

</div>

---

## Why clustereasy

Clustering in scikit-learn works great, but K-Means and Agglomerative Clustering
use **different code, different parameters, and different plotting/evaluation
steps** — which gets confusing if you just want to explore a dataset.

`clustereasy` wraps both behind **one object** with **one consistent set of
methods**:

```python
cluster = Cluster(data=df)
cluster.fit_kmeans(n_clusters=3)          # or cluster.fit_agglomerative(n_clusters=3)
cluster.plot_clusters(x="col_a", y="col_b")
cluster.evaluate()
cluster.get_clustered_data()
```

- 🧭 **One API, two algorithms** — switch between K-Means and Agglomerative without relearning anything
- 🧼 **Never mutates your data** — your original DataFrame stays untouched
- 🛑 **Explicit, beginner-friendly errors** — no silent wrong behavior on bad input
- 🔧 **Escape hatches for advanced users** — every underlying scikit-learn parameter is exposed
- 📊 **Built-in plots & metrics** — elbow curves, dendrograms, silhouette plots, PCA projections

---

## 📦 Installation

```bash
pip install clustereasy
```

<details>
<summary>Local development install (editable, with test tools)</summary>

```bash
git clone https://github.com/shashikant-ds636/easycluster.git
cd easycluster
pip install -e ".[dev]"
```

</details>

**Requirements:** Python ≥ 3.9 · pandas ≥ 1.5 · NumPy ≥ 1.23 · scikit-learn ≥ 1.2 · SciPy ≥ 1.9 · Matplotlib ≥ 3.6

---

## 🚀 Quick start

```python
import pandas as pd
from clustereasy import Cluster

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

---

## 🧪 Run it in

<table>
<tr>
<td width="33%" valign="top">

### Google Colab
No install step needed beyond one cell.

```python
!pip install -q clustereasy
```
then import and use normally.

**[▶ Open the quickstart notebook in Colab](https://colab.research.google.com/github/shashikant-ds636/easycluster/blob/main/examples/clustereasy_quickstart.ipynb)**

</td>
<td width="33%" valign="top">

### Jupyter Notebook
```bash
pip install clustereasy jupyter
jupyter notebook
```
Then, in a cell:
```python
from clustereasy import Cluster
```
Plots render inline automatically via Matplotlib's notebook backend.

</td>
<td width="33%" valign="top">

### Plain Python / VS Code
```bash
pip install clustereasy
```
```python
# script.py
from clustereasy import Cluster
cluster = Cluster(data=df)
cluster.fit_kmeans(n_clusters=3)
cluster.plot_clusters(x="a", y="b")  # opens a window
```
Run with `python script.py`.

</td>
</tr>
</table>

Runnable end-to-end scripts also live in [`examples/`](examples/):
[`kmeans_example.py`](examples/kmeans_example.py) ·
[`agglomerative_example.py`](examples/agglomerative_example.py) ·
[`clustereasy_quickstart.ipynb`](examples/clustereasy_quickstart.ipynb)

---

## 📖 Documentation

<details>
<summary><strong>DataFrame input requirements</strong></summary>

<br>

`Cluster(data=df, features=[...], scale=..., missing=...)`

- `data` — a non-empty `pandas.DataFrame`. Never modified in place.
- `features` — optional list of column names to cluster on. Defaults to all
  columns. Columns must exist, be unique, and be **numeric** — clustereasy
  will not silently convert text/categorical values into numbers. Encode
  categorical columns yourself first (e.g. `pd.get_dummies`) if you want to
  include them.
- `missing` — one of `"raise"` (default), `"drop"`, `"mean"`, `"median"`.
  Controls how missing values in the selected features are handled.
- Infinite values always raise an explicit error — never silently dropped.
- `scale` — `False` (default), `True` (same as `"standard"`), or one of
  `"standard"` / `"minmax"` / `"robust"`.

</details>

<details>
<summary><strong>K-Means tutorial</strong></summary>

<br>

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

**Elbow Method.** `elbow_method(max_clusters=10, random_state=42, show=True)`
fits K-Means for k = 1..max_clusters, plots k vs. WCSS, and returns a
DataFrame of results so you can reuse them without refitting. The elbow is a
**heuristic**, not a guaranteed optimum — pair it with `silhouette_score()`
and domain knowledge.

</details>

<details>
<summary><strong>Hierarchical Clustering tutorial</strong></summary>

<br>

```python
Z = cluster.plot_dendrogram(method="ward", metric="euclidean")
cluster.fit_agglomerative(n_clusters=3, linkage="ward")
```

Supported linkage methods: `ward`, `complete`, `average`, `single`. `ward`
linkage requires `metric="euclidean"` — any other metric raises a clear error
instead of silently producing wrong results.

Cluster by distance instead of a fixed count:

```python
cluster.fit_agglomerative(n_clusters=None, distance_threshold=5.0)
```

`distance_threshold` and `n_clusters` are mutually exclusive, matching
scikit-learn's own requirement.

> **Note:** `AgglomerativeClustering` has no native `predict()` in
> scikit-learn. `Cluster.predict()` will raise a clear error if called on an
> agglomerative model rather than faking support for it.

**Dendrogram interpretation**
- Each leaf is one observation.
- Each join (inverted "U") merges two clusters; its height is the distance at
  which they were merged.
- Cutting the dendrogram horizontally at a given height determines how many
  clusters you get at that distance.
- For large datasets, pass `truncate_mode="lastp"` with a small `p` —
  clustereasy warns above ~2000 samples if you don't.

</details>

<details>
<summary><strong>Evaluation metrics and their limitations</strong></summary>

<br>

All metrics work without ground-truth labels:

| Method | Notes |
|---|---|
| `silhouette_score()` | Needs 2..n-1 distinct clusters |
| `silhouette_samples()` | Per-sample values, same constraints |
| `davies_bouldin_score()` | Lower is better · needs ≥ 2 clusters |
| `calinski_harabasz_score()` | Higher is better · needs ≥ 2 clusters |
| `inertia()` | K-Means only — not available for Agglomerative models |
| `evaluate()` | One DataFrame with every metric applicable to the fitted model |

These are **diagnostic tools**, not proof of a "correct" clustering — sanity
check against domain knowledge. If a metric can't be computed (e.g. 1
distinct cluster), `evaluate()` shows `NaN` with a note instead of crashing;
individual metric methods raise `ParameterError`.

</details>

<details>
<summary><strong>Feature scaling &amp; missing-value handling</strong></summary>

<br>

- Distances (and therefore both algorithms' results) are scale-sensitive. If
  features are on different scales (e.g. income in thousands vs. a 1–100
  score), set `scale=True`.
- K-Means centers are available both in the transformed (scaled) space
  (`cluster._model.cluster_centers_`) and inverse-transformed back to
  original feature scale (`cluster.cluster_centers()`, the default).
- Missing values are never silently dropped — choose
  `missing="drop"/"mean"/"median"` explicitly, or clean the data yourself.

</details>

<details>
<summary><strong>Visualization</strong></summary>

<br>

```python
cluster.plot_clusters(x="Income", y="Score", show_centers=True)
cluster.plot_cluster_sizes()
cluster.plot_silhouette()
cluster.plot_pca()   # for >2 features — clearly labeled as a 2D projection
```

`plot_clusters` requires exactly two selected feature columns. For
higher-dimensional data use `plot_pca()` instead.

</details>

<details>
<summary><strong>Cluster labels &amp; DataFrame output</strong></summary>

<br>

```python
labels = cluster.get_labels()                       # pd.Series, original index
clustered_df = cluster.get_clustered_data("Segment") # original df + label column
summary = cluster.cluster_summary()                  # size, proportion, per-feature mean
```

Cluster IDs (0, 1, 2, ...) are arbitrary identifiers — no inherent ranking or
meaning.

</details>

<details>
<summary><strong>Common errors and solutions</strong></summary>

<br>

| Error | Likely cause | Fix |
|---|---|---|
| `InputValidationError: not numeric` | Selected a text/categorical column | Encode it first, or exclude from `features` |
| `InputValidationError: missing values` | NaNs present, default `missing="raise"` | Pass `missing="drop"/"mean"/"median"` |
| `ParameterError: Ward linkage requires metric='euclidean'` | `linkage="ward"` with another metric | Use `metric="euclidean"` or a different linkage |
| `NotFittedError` | Called an evaluation/plot/output method before fitting | Call `fit_kmeans`/`fit_agglomerative` first |
| `ParameterError` on `predict()` | Called `predict()` on an Agglomerative model | Not supported — K-Means only |

</details>

<details>
<summary><strong>Computational limitations</strong></summary>

<br>

- Dendrograms require O(n²) memory/time and are impractical past a few
  thousand samples.
- `AgglomerativeClustering` scales worse than `KMeans` for large n.
- No GPU acceleration — this wraps scikit-learn/SciPy CPU implementations.

</details>

---

## 🔖 API reference

| Category | Methods |
|---|---|
| **Construction** | `Cluster(data, features=None, scale=False, missing="raise")` |
| **K-Means** | `elbow_method`, `fit_kmeans`, `cluster_centers`, `inertia`, `predict` |
| **Agglomerative** | `plot_dendrogram`, `get_linkage_matrix`, `fit_agglomerative` |
| **Shared plots** | `plot_clusters`, `plot_cluster_sizes`, `plot_silhouette`, `plot_pca` |
| **Evaluation** | `silhouette_score`, `silhouette_samples`, `davies_bouldin_score`, `calinski_harabasz_score`, `evaluate` |
| **Output** | `get_labels`, `get_clustered_data`, `cluster_summary`, `elbow_results` |

---

## ✅ Testing

```bash
pip install -e ".[dev]"
pytest -q
```

Current suite: **59 tests, all passing** — covering validation, both
algorithms, visualization, evaluation edge cases, and DataFrame-output
guarantees (index preservation, immutability, reproducibility).

---

## 🗺️ Roadmap

- `DBSCAN`, `OPTICS`, `GaussianMixture` support behind the same API
- Automatic comparison across multiple `k` / linkage configurations
- Model save/load (`joblib`)
- Direct CSV export of results
- Connectivity-constrained agglomerative clustering helpers

## 🤝 Contributing

1. Fork, branch from `main`.
2. Add/update tests for any behavior change.
3. Run `pytest -q` before opening a PR.
4. Keep public methods documented with docstrings.

## 📤 Preparing a release

```bash
pip install build twine
python -m build
twine check dist/*
pip install dist/clustereasy-0.1.1-py3-none-any.whl  # test locally
twine upload --repository testpypi dist/*             # TestPyPI first
twine upload dist/*                                    # then PyPI
```

Bump `version` in `pyproject.toml` (and `__version__` in
`src/clustereasy/__init__.py`) before each release and update
`CHANGELOG.md`.

---

<div align="center">

**[PyPI](https://pypi.org/project/clustereasy/) · [GitHub](https://github.com/shashikant-ds636/easycluster) · [Issues](https://github.com/shashikant-ds636/easycluster/issues) · MIT License**

</div>
