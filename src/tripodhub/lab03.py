"""Lab 03: PCA on the Iris dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


def run(show_plot: bool = True) -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    features = iris.data
    pca = PCA(n_components=2, random_state=42)
    transformed = pca.fit_transform(features)
    result = pd.DataFrame(transformed, columns=["PC1", "PC2"])
    result["target"] = iris.target

    if show_plot:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(result["PC1"], result["PC2"], c=result["target"], cmap="viridis")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA of Iris Dataset")
        plt.legend(*scatter.legend_elements(), title="Class")
        plt.tight_layout()
        plt.show()

    return result
