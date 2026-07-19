"""Model comparison, leakage-safe tuning, training, and artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from .config import ProjectConfig
from .feature_engineering import HospitalFeatureEngineer
from .preprocessing import DataPreprocessor
from .utils import occupancy_class


class EncodedXGBoostClassifier(BaseEstimator, ClassifierMixin):
    """Adapt XGBoost's integer-label contract to human-readable occupancy classes."""

    def __init__(
        self, n_estimators: int = 150, max_depth: int = 5, min_samples_leaf: int = 1,
        learning_rate: float = 0.07, subsample: float = 0.85, colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def fit(self, x: object, y: pd.Series | np.ndarray) -> "EncodedXGBoostClassifier":
        """Fit XGBoost using deterministic integer encodings for class labels."""
        from xgboost import XGBClassifier  # type: ignore
        labels = np.asarray(y, dtype=str)
        self.classes_ = np.unique(labels)
        encoded = np.searchsorted(self.classes_, labels)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_child_weight=float(self.min_samples_leaf), learning_rate=self.learning_rate,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            random_state=self.random_state, n_jobs=-1, eval_metric="mlogloss",
        )
        self.model_.fit(x, encoded)
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, x: object) -> np.ndarray:
        """Decode class predictions back to the original occupancy labels."""
        return self.classes_[self.model_.predict(x).astype(int)]

    def predict_proba(self, x: object) -> np.ndarray:
        """Return class probabilities in the same order as ``classes_``."""
        return self.model_.predict_proba(x)


@dataclass
class TrainingResult:
    """A compact, serializable representation of model development outputs."""

    model: Pipeline
    model_name: str
    numeric_features: list[str]
    categorical_features: list[str]
    comparison: pd.DataFrame
    test_metrics: dict[str, float]
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class OccupancyModelTrainer:
    """Compare several classifiers and tune the best practical candidate."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def train(self, engineered: pd.DataFrame) -> TrainingResult:
        """Train models with a temporal holdout and CV only on the training period."""
        frame = engineered.copy().sort_values("Date").reset_index(drop=True)
        frame["Occupancy_Class"] = frame["Bed_Occupancy_Rate"].map(occupancy_class)
        numeric, categorical = HospitalFeatureEngineer.model_feature_columns(frame)
        features = numeric + categorical
        split_index = int(len(frame) * (1 - self.config.test_size))
        train_frame, test_frame = frame.iloc[:split_index], frame.iloc[split_index:]
        x_train, y_train = train_frame[features], train_frame["Occupancy_Class"]
        x_test, y_test = test_frame[features], test_frame["Occupancy_Class"]
        cv_frame = train_frame.sample(min(self.config.train_rows_for_cv, len(train_frame)), random_state=self.config.random_state)
        x_cv, y_cv = cv_frame[features], cv_frame["Occupancy_Class"]
        preprocessor = DataPreprocessor.build_feature_preprocessor(numeric, categorical)
        candidates = self._candidate_models()
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
        rows: list[dict[str, float | str]] = []
        for name, estimator in candidates.items():
            pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
            scores = cross_validate(
                pipeline, x_cv, y_cv, cv=cv,
                scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"}, n_jobs=-1,
            )
            rows.append({
                "model": name,
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
                "cv_accuracy_std": float(scores["test_accuracy"].std()),
                "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
            })
        comparison = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False).reset_index(drop=True)
        best_name = str(comparison.iloc[0]["model"])
        tuned = self._tune(best_name, preprocessor, candidates[best_name], x_cv, y_cv, cv)
        tuned.fit(x_train, y_train)
        predictions = tuned.predict(x_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        }
        return TrainingResult(tuned, best_name, numeric, categorical, comparison, metrics, x_train, x_test, y_train, y_test)

    def save_artifacts(self, result: TrainingResult, models_dir: Path) -> None:
        """Persist the fitted full pipeline and a standalone numeric scaler for deployment."""
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.model, models_dir / "best_model.pkl")
        standalone_scaler = RobustScaler().fit(result.x_train[result.numeric_features])
        joblib.dump(standalone_scaler, models_dir / "scaler.pkl")

    def _candidate_models(self) -> dict[str, object]:
        models: dict[str, object] = {
            "Logistic Regression": LogisticRegression(max_iter=800, C=1.0),
            "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=4, random_state=self.config.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=16, min_samples_leaf=2, random_state=self.config.random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.07, max_depth=3, random_state=self.config.random_state),
            "Extra Trees": ExtraTreesClassifier(n_estimators=220, max_depth=18, min_samples_leaf=2, random_state=self.config.random_state, n_jobs=-1),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.6, random_state=self.config.random_state),
            "Support Vector Machine": SVC(C=1.2, gamma="scale", probability=True, random_state=self.config.random_state),
        }
        try:
            import xgboost  # type: ignore  # noqa: F401
            models["XGBoost"] = EncodedXGBoostClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.07, subsample=0.85,
                colsample_bytree=0.8, random_state=self.config.random_state,
            )
        except ImportError:
            pass
        return models

    def _tune(self, name: str, preprocessor: object, estimator: object, x: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> Pipeline:
        """Use randomized and grid search according to the selected estimator family."""
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        if name in {"Random Forest", "Extra Trees", "XGBoost"}:
            distribution = {
                "model__n_estimators": [120, 180, 260],
                "model__max_depth": [8, 14, 20, None],
                "model__min_samples_leaf": [1, 2, 4],
            }
            search = RandomizedSearchCV(
                pipeline, distribution, n_iter=5, scoring="f1_macro", cv=cv,
                random_state=self.config.random_state, n_jobs=-1,
            )
        else:
            parameters = {"model__C": [0.3, 1.0, 2.0]} if name == "Logistic Regression" else {"model__max_depth": [3, 6, 10]} if name == "Decision Tree" else {}
            if not parameters:
                return pipeline
            search = GridSearchCV(pipeline, parameters, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(x, y)
        return search.best_estimator_
