# Changelog

## [0.1.2] - Visualization refresh

- Redesigned every plot in `visualization.py` with a consistent, modern
  style: shared color palette, minimal chrome (no top/right spines, light
  y-axis grid only), consistent title/label typography, and legends placed
  outside the plot area instead of overlapping data.
- `elbow_method()` now marks a suggested k on the chart as a visual hint
  (still just a heuristic — doesn't change return value or API).
- `plot_cluster_sizes()` now annotates each bar with its count.
- `plot_silhouette()` labels each band by cluster and shows the average
  score in the legend instead of just as an unlabeled line.
- `plot_dendrogram()` colors sub-clusters below a distance threshold instead
  of rendering everything in one flat color.
- No functional/API changes: all method signatures, return values, and
  parameters are unchanged from 0.1.1.

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
