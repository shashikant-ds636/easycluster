# Changelog

## [0.1.1] - Documentation & branding release

- Redesigned README as full documentation: logo/wordmark, badges (PyPI
  version, Python versions, license, CI tests, Open-in-Colab), collapsible
  reference sections, and side-by-side usage guides for Google Colab,
  Jupyter, and plain Python/VS Code.
- Added `docs/assets/logo.svg` / `logo.png`.
- Added `examples/clustereasy_quickstart.ipynb`, a runnable Colab/Jupyter
  notebook covering the full K-Means + Agglomerative workflow.
- No functional/API changes from 0.1.0.

## [0.1.0] - Initial release

- `Cluster` class wrapping K-Means and Agglomerative Clustering with a shared API.
- DataFrame input validation: empty-data checks, feature selection, duplicate
  column detection, non-numeric column rejection, missing-value handling
  (`drop`/`mean`/`median`/`raise`), infinite-value detection.
- Optional feature scaling (`standard`, `minmax`, `robust`) with correct
  inverse-transform of K-Means centers back to original feature scale.
- K-Means: `fit_kmeans`, `elbow_method`, `cluster_centers`, `inertia`, `predict`.
- Agglomerative: `fit_agglomerative`, `plot_dendrogram`, `get_linkage_matrix`.
  No `predict()` is offered for Agglomerative models (scikit-learn has none).
- Shared visualization: `plot_clusters`, `plot_cluster_sizes`, `plot_silhouette`,
  `plot_pca`.
- Evaluation: `silhouette_score`, `silhouette_samples`, `davies_bouldin_score`,
  `calinski_harabasz_score`, `evaluate()`.
- Output: `get_labels`, `get_clustered_data`, `cluster_summary`.
- Full pytest suite covering validation, both algorithms, evaluation edge
  cases, and DataFrame-output guarantees (index preservation, immutability).
