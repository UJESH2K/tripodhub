"""Lab 10: k-means clustering on breast cancer data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def run(show_plot: bool = True) -> dict[str, float]:
    """Cluster the breast cancer dataset and optionally visualize the result."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = model.fit_predict(features)
    silhouette = silhouette_score(features, clusters)

    reduced = PCA(n_components=2, random_state=42).fit_transform(features)
    result = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    result["cluster"] = clusters

    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(result["PC1"], result["PC2"], c=result["cluster"], cmap="viridis")
        plt.title("K-Means Clustering on Breast Cancer Data")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.tight_layout()
        plt.show()

    return {"silhouette_score": float(silhouette)}
