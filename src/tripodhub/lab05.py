"""Lab 05: k-Nearest Neighbours on generated points."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def generate_data(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x_values = np.sort(rng.random(100))
    y_labels = np.where(x_values <= 0.5, 0, 1)
    return x_values.reshape(-1, 1), y_labels


def run(
    k_values: tuple[int, ...] = (1, 2, 3, 4, 5, 20, 30),
    show_plot: bool = True,
    random_state: int = 42,
) -> dict[int, float]:
    x_values, y_labels = generate_data(random_state=random_state)
    x_train, y_train = x_values[:50], y_labels[:50]
    x_test, y_test = x_values[50:], y_labels[50:]
    accuracies: dict[int, float] = {}

    if show_plot:
        plt.figure(figsize=(14, 10))

    for index, k in enumerate(k_values, start=1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracies[k] = accuracy_score(y_test, predictions)

        if show_plot:
            plt.subplot(3, 3, index)
            plt.scatter(x_test[:, 0], y_test, label="True", alpha=0.7)
            plt.scatter(x_test[:, 0], predictions, label="Predicted", marker="x")
            plt.title(f"KNN (k={k})")
            plt.xlabel("x")
            plt.ylabel("class")
            plt.ylim(-0.2, 1.2)

    if show_plot:
        plt.tight_layout()
        plt.show()

    return accuracies
