"""Dependency-free artifact types used only when scientific packages are unavailable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class RuleBasedOccupancyModel:
    """Portable baseline classifier persisted for artifact portability checks.

    The normal project workflow replaces this object with the tuned scikit-learn
    pipeline in ``run_project.py``. It is intentionally simple but executable.
    """

    high_threshold: float = 85.0
    moderate_threshold: float = 70.0
    feature_names_in_: tuple[str, ...] = ("Admissions", "Staff_On_Duty", "Emergency_Cases", "Doctors", "Nurses")

    def predict(self, rows: Iterable[Any]) -> list[str]:
        """Classify rows using a transparent staffing-demand pressure rule."""
        predictions: list[str] = []
        for row in rows:
            if hasattr(row, "to_dict"):
                row = row.to_dict()
            admissions = float(row.get("Admissions", 20))
            emergency = float(row.get("Emergency_Cases", 20))
            staff = max(float(row.get("Staff_On_Duty", 35)), 1.0)
            nurses = float(row.get("Nurses", 20))
            doctors = max(float(row.get("Doctors", 8)), 1.0)
            score = 52 + 0.65 * admissions + 0.38 * emergency - 0.30 * staff - 0.18 * nurses + 1.2 * doctors
            predictions.append("High" if score >= self.high_threshold else "Moderate" if score >= self.moderate_threshold else "Low")
        return predictions


@dataclass
class PortableCentroidClassifier:
    """A dependency-free trained nearest-centroid occupancy classifier.

    It is used solely for the pre-generated artifact when the local workspace
    cannot install scientific wheels. The normal project pipeline replaces it
    with the cross-validated scikit-learn model.
    """

    feature_names_in_: tuple[str, ...]
    centroids: dict[str, dict[str, float]]
    scales: dict[str, float]
    classes_: tuple[str, ...] = ("High", "Low", "Moderate")

    @classmethod
    def fit(cls, rows: Iterable[dict[str, Any]], features: tuple[str, ...]) -> "PortableCentroidClassifier":
        """Learn class centroids from labelled hospital census records."""
        buckets: dict[str, list[dict[str, Any]]] = {"Low": [], "Moderate": [], "High": []}
        materialized = list(rows)
        for row in materialized:
            rate = float(row["Bed_Occupancy_Rate"])
            label = "High" if rate >= 85 else "Moderate" if rate >= 70 else "Low"
            buckets[label].append(row)
        means = {
            label: {feature: sum(float(row[feature]) for row in bucket) / len(bucket) for feature in features}
            for label, bucket in buckets.items()
        }
        scales = {}
        for feature in features:
            values = [float(row[feature]) for row in materialized]
            average = sum(values) / len(values)
            scales[feature] = (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5 or 1.0
        return cls(features, means, scales)

    def _rows(self, rows: Iterable[Any]) -> list[dict[str, Any]]:
        if hasattr(rows, "to_dict"):
            return list(rows.to_dict(orient="records"))
        return [row.to_dict() if hasattr(row, "to_dict") else row for row in rows]

    def predict(self, rows: Iterable[Any]) -> list[str]:
        """Return the class whose trained centroid has minimum scaled distance."""
        predictions: list[str] = []
        for row in self._rows(rows):
            distances = {
                label: sum(((float(row[feature]) - center[feature]) / self.scales[feature]) ** 2 for feature in self.feature_names_in_)
                for label, center in self.centroids.items()
            }
            predictions.append(min(distances, key=distances.get))
        return predictions

    def predict_proba(self, rows: Iterable[Any]) -> list[list[float]]:
        """Produce normalized inverse-distance class scores for desktop previews."""
        probabilities: list[list[float]] = []
        for row in self._rows(rows):
            scores = []
            for label in self.classes_:
                distance = sum(((float(row[feature]) - self.centroids[label][feature]) / self.scales[feature]) ** 2 for feature in self.feature_names_in_)
                scores.append(1.0 / (1.0 + distance))
            total = sum(scores)
            probabilities.append([score / total for score in scores])
        return probabilities


@dataclass
class PortableRobustScaler:
    """Small serializable scaler metadata object for artifact portability."""

    medians: dict[str, float]
    iqr: dict[str, float]

    def transform_one(self, row: dict[str, float]) -> dict[str, float]:
        """Apply robust scaling to a mapping of numeric inputs."""
        return {key: (float(value) - self.medians[key]) / (self.iqr[key] or 1.0) for key, value in row.items() if key in self.medians}
