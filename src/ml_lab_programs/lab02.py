"""Lab 02: correlation matrix, heatmap, and pair plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing


def load_dataset() -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    data = dataset.frame.copy()
    data["MedHouseVal"] = dataset.target
    return data


def compute_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    return data.corr(numeric_only=True)


def plot_heatmap(correlation_matrix: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("California Housing Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_pairplot(data: pd.DataFrame) -> None:
    sns.pairplot(data.sample(min(500, len(data)), random_state=42))
    plt.show()


def run(show_plots: bool = True) -> pd.DataFrame:
    """Execute the lab and return the correlation matrix."""
    data = load_dataset()
    correlation_matrix = compute_correlation_matrix(data)
    if show_plots:
        plot_heatmap(correlation_matrix)
        plot_pairplot(data)
    return correlation_matrix
