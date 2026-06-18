"""Lab 04: Find-S algorithm."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_DATA = pd.DataFrame(
    [
        ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
        ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
        ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
        ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"],
    ],
    columns=["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast", "EnjoySport"],
)


def load_training_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        return DEFAULT_DATA.copy()
    return pd.read_csv(csv_path)


def find_s(examples: pd.DataFrame, target_column: str) -> list[str]:
    positive_examples = examples[examples[target_column].astype(str).str.lower().isin(["yes", "true", "1"])]
    if positive_examples.empty:
        raise ValueError("Find-S requires at least one positive example.")

    hypothesis = positive_examples.iloc[0].drop(labels=[target_column]).tolist()
    for _, row in positive_examples.iloc[1:].iterrows():
        attributes = row.drop(labels=[target_column]).tolist()
        for index, value in enumerate(attributes):
            if hypothesis[index] != value:
                hypothesis[index] = "?"
    return hypothesis


def format_hypothesis(columns: Iterable[str], hypothesis: list[str]) -> dict[str, str]:
    return dict(zip(columns, hypothesis, strict=True))


def run(csv_path: str | Path | None = None, target_column: str = "EnjoySport") -> dict[str, str]:
    data = load_training_data(csv_path)
    feature_columns = [column for column in data.columns if column != target_column]
    hypothesis = find_s(data, target_column)
    return format_hypothesis(feature_columns, hypothesis)
