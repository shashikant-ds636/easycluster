"""Shared plotting helpers for easycluster. Matplotlib only, by design.

Design goals for every plot here:
  - one job per plot, nothing stacked or overlaid that doesn't need to be
  - a consistent, modern look (same palette, same spines/grid treatment)
  - labels and titles a beginner can read without guessing what they mean
"""

from __future__ import annotations

import numpy as np

from .exceptions import InputValidationError, ParameterError

# ---------------------------------------------------------------------- #
# Shared style
# ---------------------------------------------------------------------- #
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
_GRID_COLOR = "#D9D9D9"
_TEXT_COLOR = "#333333"


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply the shared, minimal-chrome look to an axes."""
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=_TEXT_COLOR, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, color=_TEXT_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, color=_TEXT_COLOR)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B0B0B0")
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.grid(True, axis="y", color=_GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=9.5)


def _legend(ax, **kwargs):
    return ax.legend(
        frameon=False,
        fontsize=9.5,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        **kwargs,
    )


def _finish(fig, ax, show, created):
    if created:
        fig.tight_layout()
        if show:
            import matplotlib.pyplot as plt

            plt.show()
    return ax


# ---------------------------------------------------------------------- #
# K-Means: elbow method
# ---------------------------------------------------------------------- #
def plot_elbow(ks, wcss, show=True, ax=None):
    """Plot WCSS vs k, and mark the single most elbow-like point as a hint."""
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    ks, wcss = list(ks), list(wcss)
    ax.plot(ks, wcss, marker="o", markersize=5, linewidth=2, color=_color(0), zorder=3)

    elbow_k = _estimate_elbow(ks, wcss)
    if elbow_k is not None:
        idx = ks.index(elbow_k)
        ax.scatter(
            [elbow_k], [wcss[idx]], s=130, facecolors="none",
            edgecolors=_color(3), linewidths=2, zorder=4,
        )
        ax.annotate(
            f"  suggested k = {elbow_k}",
            xy=(elbow_k, wcss[idx]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9.5,
            color=_color(3),
            fontweight="bold",
        )

    ax.set_xticks(ks)
    _style_axes(
        ax,
        title="Elbow Method — choosing k",
        xlabel="Number of clusters (k)",
        ylabel="WCSS (inertia) — lower is tighter",
    )
    return _finish(fig, ax, show, created)


def _estimate_elbow(ks, wcss):
    """Simple 'distance from the line between first and last point' heuristic,
    used only to draw a visual hint — never to silently pick k for the user."""
    if len(ks) < 3:
        return None
    x = np.array(ks, dtype=float)
    y = np.array(wcss, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_vec /= np.linalg.norm(line_vec) + 1e-12
    dists = []
    for xi, yi in zip(x_norm, y_norm):
        p = np.array([xi, yi]) - p1
        proj = np.dot(p, line_vec) * line_vec
        dists.append(np.linalg.norm(p - proj))
    return ks[int(np.argmax(dists))]


# ---------------------------------------------------------------------- #
# Agglomerative: dendrogram
# ---------------------------------------------------------------------- #
def plot_dendrogram_from_linkage(Z, truncate_mode=None, p=5, figsize=(9, 5), show=True, ax=None):
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as sch

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sch.dendrogram(
        Z,
        truncate_mode=truncate_mode,
        p=p,
        ax=ax,
        color_threshold=0.7 * max(Z[:, 2]),
        above_threshold_color="#B0B0B0",
    )
    _style_axes(
        ax,
        title="Hierarchical Clustering Dendrogram",
        xlabel="Sample index (or cluster size, if truncated)",
        ylabel="Distance between merged clusters",
    )
    ax.grid(False)
    return _finish(fig, ax, show, created)


# ---------------------------------------------------------------------- #
# Shared: 2D scatter of clusters
# ---------------------------------------------------------------------- #
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
    else:
        fig = ax.figure

    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            df_original.loc[mask, x],
            df_original.loc[mask, y],
            s=45,
            alpha=0.85,
            linewidths=0.4,
            edgecolors="white",
            color=_color(i),
            label=f"Cluster {lab}",
            zorder=3,
        )

    if show_centers and centers_original is not None:
        ax.scatter(
            centers_original[:, 0],
            centers_original[:, 1],
            s=220,
            c="black",
            marker="X",
            linewidths=1.2,
            edgecolors="white",
            label="Centers",
            zorder=4,
        )

    _style_axes(ax, title=title or f"Clusters: {x} vs {y}", xlabel=x, ylabel=y)
    _legend(ax)
    return _finish(fig, ax, show, created)


# ---------------------------------------------------------------------- #
# Cluster sizes
# ---------------------------------------------------------------------- #
def plot_cluster_sizes(sizes_series, show=True, ax=None):
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    sizes_series = sizes_series.sort_index()
    bars = ax.bar(
        sizes_series.index.astype(str),
        sizes_series.values,
        color=[_color(i) for i in range(len(sizes_series))],
        width=0.6,
        zorder=3,
    )
    for b in bars:
        height = b.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(b.get_x() + b.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color=_TEXT_COLOR,
        )

    _style_axes(ax, title="Cluster Sizes", xlabel="Cluster", ylabel="Number of samples")
    ax.margins(y=0.12)
    return _finish(fig, ax, show, created)


# ---------------------------------------------------------------------- #
# Silhouette plot
# ---------------------------------------------------------------------- #
def plot_silhouette(X, labels, show=True, ax=None):
    import matplotlib.pyplot as plt

    from .evaluation import safe_silhouette_samples

    sample_vals = safe_silhouette_samples(np.asarray(X), np.asarray(labels))
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    y_lower = 10
    yticks, yticklabels = [], []
    for i, lab in enumerate(unique_labels):
        cluster_vals = np.sort(sample_vals[labels == lab])
        size = cluster_vals.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, cluster_vals,
            facecolor=_color(i), edgecolor=_color(i), alpha=0.85, zorder=3,
        )
        yticks.append(y_lower + 0.5 * size)
        yticklabels.append(f"Cluster {lab}")
        y_lower = y_upper + 10

    avg = float(np.mean(sample_vals))
    ax.axvline(x=avg, color=_color(3), linestyle="--", linewidth=1.6, label=f"Average = {avg:.2f}", zorder=4)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim(-0.2, 1.0)
    _style_axes(ax, title="Silhouette Plot", xlabel="Silhouette coefficient", ylabel=None)
    ax.grid(True, axis="x", color=_GRID_COLOR, linewidth=0.8, zorder=0)
    ax.grid(False, axis="y")
    _legend(ax)
    return _finish(fig, ax, show, created)


# ---------------------------------------------------------------------- #
# PCA projection
# ---------------------------------------------------------------------- #
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
    else:
        fig = ax.figure

    labels = np.asarray(labels)
    for i, lab in enumerate(sorted(set(labels.tolist()))):
        mask = labels == lab
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            s=45, alpha=0.85, linewidths=0.4, edgecolors="white",
            color=_color(i), label=f"Cluster {lab}", zorder=3,
        )

    var = pca.explained_variance_ratio_
    _style_axes(
        ax,
        title="PCA Projection (2D) — not the original feature space",
        xlabel=f"PC1 ({var[0]*100:.1f}% variance)",
        ylabel=f"PC2 ({var[1]*100:.1f}% variance)",
    )
    _legend(ax)
    return _finish(fig, ax, show, created)
