"""Runnable K-Means example using easycluster.

Mimics a typical "Mall Customers" style dataset (synthetic here, so this
script runs with no external downloads).
"""

import numpy as np
import pandas as pd

from clustereasy import Cluster

np.random.seed(42)
n = 200
df = pd.DataFrame(
    {
        "CustomerID": range(1, n + 1),
        "Annual Income": np.concatenate(
            [np.random.normal(25, 5, n // 2), np.random.normal(75, 8, n - n // 2)]
        ),
        "Spending Score": np.concatenate(
            [np.random.normal(30, 10, n // 2), np.random.normal(70, 10, n - n // 2)]
        ),
    }
)

# 1. Build the Cluster object, selecting only numeric feature columns
cluster = Cluster(data=df, features=["Annual Income", "Spending Score"], scale=True)

# 2. Use the elbow method to help choose k
elbow_table = cluster.elbow_method(max_clusters=8, show=False)
print(elbow_table)

# 3. Fit K-Means
cluster.fit_kmeans(n_clusters=4, random_state=42)

# 4. Visualize (set show=True to display the plot interactively)
cluster.plot_clusters(x="Annual Income", y="Spending Score", show=False)

# 5. Evaluate
print(cluster.evaluate())

# 6. Get labeled data back
result = cluster.get_clustered_data(column_name="Segment")
print(result.head())

# 7. Cluster-level summary
print(cluster.cluster_summary())
