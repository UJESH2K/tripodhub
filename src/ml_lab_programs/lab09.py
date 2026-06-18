"""Lab 09: Naive Bayes on the Olivetti faces dataset."""

from __future__ import annotations

from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def run() -> dict[str, float]:
    """Train a Gaussian Naive Bayes classifier on Olivetti faces."""
    dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42, stratify=dataset.target
    )
    model = GaussianNB()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {"accuracy": float(accuracy_score(y_test, predictions))}
