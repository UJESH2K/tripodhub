"""Lab 08: Decision tree classification."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def run() -> dict[str, object]:
    """Train a decision tree and classify a sample from the test set."""
    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42, stratify=dataset.target
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    new_sample = x_test[0].reshape(1, -1)
    new_prediction = model.predict(new_sample)[0]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "sample_prediction": str(dataset.target_names[int(new_prediction)]),
        "sample_features": np.round(new_sample[0], 4).tolist(),
    }
