"""Runnable Agglomerative Clustering example using easycluster."""

import numpy as np
import pandas as pd

from clustereasy import Cluster

np.random.seed(0)
n = 120
df = pd.DataFrame(
    {
        "Age": np.concatenate([np.random.normal(25, 3, n // 2), np.random.normal(45, 5, n - n // 2)]),
        "Annual Income": np.concatenate(
            [np.random.normal(30, 6, n // 2), np.random.normal(60, 10, n - n // 2)]
        ),
    }
)

cluster = Cluster(data=df, features=["Age", "Annual Income"], scale=True)

# 1. Inspect the dendrogram to get a feel for the structure
Z = cluster.plot_dendrogram(method="ward", show=False)
print("Linkage matrix shape:", Z.shape)

# 2. Fit Agglomerative Clustering
cluster.fit_agglomerative(n_clusters=3, linkage="ward")

# 3. Visualize
cluster.plot_clusters(x="Age", y="Annual Income", show_centers=False, show=False)

# 4. Evaluate (no inertia available for this algorithm)
print(cluster.evaluate())

# 5. Labeled data
print(cluster.get_clustered_data().head())
