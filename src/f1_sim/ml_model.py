"""
Lightweight machine learning model for predicting tire degradation.

The model is trained on synthetic data generated from the analytical
degradation function (with weather & temperature perturbations) so that it can
approximate degradation dynamics while remaining fast and dependency-free.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .degradation import get_degradation
from .strategy import VALID_COMPOUNDS

MAX_LAP_AGE = 45.0
TEMP_BASELINE = 35.0
TEMP_RANGE = 25.0


@dataclass
class LinearModel:
    weights: List[float]
    bias: float

    def predict(self, features: Sequence[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features)) + self.bias


def _feature_vector(
    compound: str,
    lap_age: float,
    track_temp: float,
    weather_penalty: float,
) -> List[float]:
    lap_norm = max(lap_age, 0.0) / MAX_LAP_AGE
    temp_norm = (track_temp - TEMP_BASELINE) / TEMP_RANGE
    features = [lap_norm, temp_norm, weather_penalty]
    for name in VALID_COMPOUNDS:
        features.append(1.0 if compound == name else 0.0)
    return features


def _generate_dataset(samples_per_compound: int = 600) -> List[Tuple[List[float], float]]:
    dataset: List[Tuple[List[float], float]] = []
    for compound in VALID_COMPOUNDS:
        for _ in range(samples_per_compound):
            lap_age = random.uniform(0, MAX_LAP_AGE)
            track_temp = random.uniform(25, 50)
            weather_penalty = random.choice([0.0, 0.1, 0.3, 0.6, 1.0])
            base_deg = get_degradation(compound, lap_age)
            weather_factor = 1.0 + 0.01 * (track_temp - 40.0) + 0.08 * weather_penalty
            target = base_deg * weather_factor
            noise = random.gauss(0.0, 0.01)
            dataset.append((_feature_vector(compound, lap_age, track_temp, weather_penalty), target + noise))
    random.shuffle(dataset)
    return dataset


def _train_linear_model(
    dataset: Sequence[Tuple[Sequence[float], float]],
    learning_rate: float = 0.08,
    epochs: int = 800,
) -> LinearModel:
    if not dataset:
        raise ValueError("Dataset must contain samples.")
    feature_dim = len(dataset[0][0])
    weights = [random.uniform(-0.1, 0.1) for _ in range(feature_dim)]
    bias = 0.0

    for _ in range(epochs):
        grad_w = [0.0] * feature_dim
        grad_b = 0.0
        for features, target in dataset:
            prediction = sum(w * x for w, x in zip(weights, features)) + bias
            error = prediction - target
            for idx in range(feature_dim):
                grad_w[idx] += error * features[idx]
            grad_b += error
        n = float(len(dataset))
        for idx in range(feature_dim):
            weights[idx] -= learning_rate * grad_w[idx] / n
        bias -= learning_rate * grad_b / n
    return LinearModel(weights=weights, bias=bias)


_DATASET = _generate_dataset()
_MODEL = _train_linear_model(_DATASET)


def predict_degradation(
    compound: str,
    lap_age: float,
    track_temp: float,
    weather_penalty: float,
) -> float:
    """Return the ML-estimated degradation contribution in seconds."""
    features = _feature_vector(compound, lap_age, track_temp, weather_penalty)
    prediction = _MODEL.predict(features)
    return max(prediction, 0.0)


__all__ = ["predict_degradation"]

