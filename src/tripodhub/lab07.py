"""Lab 07: Linear and Polynomial Regression."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def load_linear_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_california_housing(as_frame=True)
    return dataset.data, dataset.target


def load_polynomial_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    try:
        data = sns.load_dataset("mpg").dropna()
        features = data[["horsepower"]].astype(float)
        target = data["mpg"].astype(float)
        return features, target
    except Exception:
        data = fetch_openml(name="autoMpg", version=1, as_frame=True, parser="auto")
        frame = data.frame.dropna()
        features = frame[["horsepower"]].astype(float)
        target = frame["mpg"].astype(float)
        return features, target


def run(show_plot: bool = True) -> dict[str, float]:
    x_linear, y_linear = load_linear_regression_data()
    x_train_l, x_test_l, y_train_l, y_test_l = train_test_split(
        x_linear, y_linear, test_size=0.2, random_state=42
    )
    linear_model = LinearRegression()
    linear_model.fit(x_train_l, y_train_l)
    linear_predictions = linear_model.predict(x_test_l)

    x_poly, y_poly = load_polynomial_regression_data()
    x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(
        x_poly, y_poly, test_size=0.2, random_state=42
    )
    polynomial_model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )
    polynomial_model.fit(x_train_p, y_train_p)
    polynomial_predictions = polynomial_model.predict(x_test_p)

    if show_plot:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(y_test_l, linear_predictions, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Linear Regression")

        plt.subplot(1, 2, 2)
        sort_index = np.argsort(x_test_p.iloc[:, 0].to_numpy())
        plt.scatter(x_test_p.iloc[:, 0], y_test_p, alpha=0.6, label="Actual")
        plt.plot(
            x_test_p.iloc[:, 0].to_numpy()[sort_index],
            polynomial_predictions[sort_index],
            color="red",
            label="Polynomial Fit",
        )
        plt.xlabel("Horsepower")
        plt.ylabel("MPG")
        plt.title("Polynomial Regression")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return {
        "linear_r2": r2_score(y_test_l, linear_predictions),
        "linear_rmse": mean_squared_error(y_test_l, linear_predictions) ** 0.5,
        "polynomial_r2": r2_score(y_test_p, polynomial_predictions),
        "polynomial_rmse": mean_squared_error(y_test_p, polynomial_predictions) ** 0.5,
    }
