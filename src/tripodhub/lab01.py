"""Lab 01: histograms, box plots, and outlier analysis."""

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


def analyze_outliers(data: pd.DataFrame) -> pd.Series:
    numeric = data.select_dtypes(include="number")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    mask = (numeric < (q1 - 1.5 * iqr)) | (numeric > (q3 + 1.5 * iqr))
    return mask.sum()


def plot_histograms(data: pd.DataFrame) -> None:
    data.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histograms of California Housing Features")
    plt.tight_layout()
    plt.show()


def plot_boxplots(data: pd.DataFrame) -> None:
    plt.figure(figsize=(15, 10))
    for index, column in enumerate(data.columns, start=1):
        plt.subplot(3, 3, index)
        sns.boxplot(y=data[column])
        plt.title(column)
    plt.suptitle("Box Plots of California Housing Features")
    plt.tight_layout()
    plt.show()


def run(show_plots: bool = True) -> dict[str, int]:
    data = load_dataset()
    outliers = analyze_outliers(data)
    if show_plots:
        plot_histograms(data)
        plot_boxplots(data)
    return outliers.to_dict()
