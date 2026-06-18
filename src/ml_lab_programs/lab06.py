"""Lab 06: Locally Weighted Regression."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def locally_weighted_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    query_points: np.ndarray,
    tau: float = 0.5,
) -> np.ndarray:
    """Predict outputs using locally weighted linear regression."""
    x_train_bias = np.c_[np.ones(len(x_train)), x_train]
    predictions = []

    for point in query_points:
        weights = np.exp(-np.sum((x_train - point) ** 2, axis=1) / (2 * tau**2))
        weighted_x = x_train_bias * weights[:, np.newaxis]
        weighted_y = y_train * weights
        theta, _, _, _ = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)
        point_bias = np.r_[1.0, point]
        predictions.append(point_bias @ theta)

    return np.array(predictions)


def generate_sample_data(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x = np.linspace(-3, 3, 100).reshape(-1, 1)
    noise = rng.normal(0, 0.2, size=len(x))
    y = np.sin(x[:, 0]) + noise
    return x, y


def run(show_plot: bool = True, tau: float = 0.5) -> dict[str, np.ndarray]:
    """Fit sample data using locally weighted regression."""
    x_train, y_train = generate_sample_data()
    x_query = np.linspace(x_train.min(), x_train.max(), 200).reshape(-1, 1)
    y_pred = locally_weighted_regression(x_train, y_train, x_query, tau=tau)

    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(x_train[:, 0], y_train, label="Training Data", alpha=0.7)
        plt.plot(x_query[:, 0], y_pred, color="red", label="LWR Fit")
        plt.title("Locally Weighted Regression")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"x_query": x_query[:, 0], "predictions": y_pred}
