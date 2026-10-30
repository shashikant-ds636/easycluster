"""Shared plotting helpers for easycluster. Matplotlib only, by design."""

from __future__ import annotations

import numpy as np

from .exceptions import InputValidationError, ParameterError


def plot_elbow(ks, wcss, show=True, ax=None):
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(ks), list(wcss), marker="o")
    ax.set_title("The Elbow Method")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS (inertia)")
    if show and created:
        plt.show()
    return ax


def plot_dendrogram_from_linkage(Z, truncate_mode=None, p=5, figsize=(10, 7), show=True, ax=None):
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as sch

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    sch.dendrogram(Z, truncate_mode=truncate_mode, p=p, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Sample index (or cluster size if truncated)")
    ax.set_ylabel("Distance")
    if show and created:
        plt.show()
    return ax


def plot_cluster_scatter(
    df_original,
    x,
    y,
    labels,
    centers_original=None,
    show_centers=True,
    title=None,
    figsize=(7, 5),
    show=True,
    ax=None,
):
    import matplotlib.pyplot as plt

    if x not in df_original.columns or y not in df_original.columns:
        raise InputValidationError(
            f"Columns {x!r} and/or {y!r} were not found in the data. "
            f"Available columns: {list(df_original.columns)}"
        )

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)

    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            df_original.loc[mask, x],
            df_original.loc[mask, y],
            s=40,
            alpha=0.8,
            color=cmap(i % 10),
            label=f"Cluster {lab}",
        )

    if show_centers and centers_original is not None:
        x_idx = list(df_original.columns[: centers_original.shape[1]]).index(x) if False else None
        # centers_original is passed already aligned to [x, y] order by the caller
        ax.scatter(
            centers_original[:, 0],
            centers_original[:, 1],
            s=200,
            c="black",
            marker="X",
            label="Centers",
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"Clusters: {x} vs {y}")
    ax.legend()
    if show and created:
        plt.show()
    return ax


def plot_cluster_sizes(sizes_series, show=True, ax=None):
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6, 4))
    sizes_series.sort_index().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Cluster Sizes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of samples")
    if show and created:
        plt.show()
    return ax


def plot_silhouette(X, labels, show=True, ax=None):
    import matplotlib.pyplot as plt

    from .evaluation import safe_silhouette_samples

    sample_vals = safe_silhouette_samples(np.asarray(X), np.asarray(labels))
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 5))

    cmap = plt.get_cmap("tab10")
    y_lower = 10
    for i, lab in enumerate(unique_labels):
        cluster_vals = np.sort(sample_vals[labels == lab])
        size = cluster_vals.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals, color=cmap(i % 10), alpha=0.8)
        ax.text(-0.02, y_lower + 0.5 * size, str(lab))
        y_lower = y_upper + 10

    ax.axvline(x=float(np.mean(sample_vals)), color="red", linestyle="--", label="Average score")
    ax.set_title("Silhouette Plot")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.legend()
    if show and created:
        plt.show()
    return ax


def plot_pca_projection(X, labels, show=True, ax=None):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    X = np.asarray(X)
    if X.shape[1] < 2:
        raise ParameterError("PCA projection requires at least 2 features.")
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 5))

    labels = np.asarray(labels)
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(sorted(set(labels.tolist()))):
        mask = labels == lab
        ax.scatter(proj[mask, 0], proj[mask, 1], s=40, alpha=0.8, color=cmap(i % 10), label=f"Cluster {lab}")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title("PCA Projection (2D) — Not the original feature space")
    ax.legend()
    if show and created:
        plt.show()
    return ax
