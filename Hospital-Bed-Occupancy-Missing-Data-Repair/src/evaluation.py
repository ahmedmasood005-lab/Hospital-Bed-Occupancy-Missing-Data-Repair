"""Model performance tables, prediction exports, and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, classification_report
from sklearn.model_selection import learning_curve, validation_curve

from .modeling import TrainingResult


class ModelEvaluator:
    """Create deployment-oriented classification diagnostics."""

    def evaluate(self, result: TrainingResult, figure_dir: Path, predictions_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate holdout predictions and write all requested visual diagnostics."""
        figure_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        model = result.model
        predictions = model.predict(result.x_test)
        probabilities = model.predict_proba(result.x_test) if hasattr(model, "predict_proba") else None
        report = pd.DataFrame(classification_report(result.y_test, predictions, output_dict=True, zero_division=0)).transpose().reset_index(names="label")
        prediction_export = result.x_test.copy()
        prediction_export["Actual_Occupancy_Class"] = result.y_test.to_numpy()
        prediction_export["Predicted_Occupancy_Class"] = predictions
        prediction_export.to_csv(predictions_dir / "holdout_predictions.csv", index=False)
        self._confusion(result.y_test, predictions, figure_dir)
        if probabilities is not None:
            self._roc_pr_calibration(result, probabilities, figure_dir)
        self._learning_and_validation(result, figure_dir)
        importance = self._importance(result, figure_dir)
        return report, importance

    @staticmethod
    def _confusion(y_true: pd.Series, predictions: np.ndarray, directory: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(y_true, predictions, cmap="Blues", ax=ax)
        ax.set_title("Occupancy Risk Confusion Matrix")
        fig.tight_layout(); fig.savefig(directory / "confusion_matrix.png", dpi=180); plt.close(fig)

    @staticmethod
    def _roc_pr_calibration(result: TrainingResult, probabilities: np.ndarray, directory: Path) -> None:
        classes = list(result.model.classes_)
        high_index = classes.index("High") if "High" in classes else int(np.argmax(classes))
        binary = result.y_test.eq("High").astype(int)
        high_probability = probabilities[:, high_index]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        RocCurveDisplay.from_predictions(binary, high_probability, ax=axes[0], name="High Occupancy")
        axes[0].set_title("ROC Curve")
        PrecisionRecallDisplay.from_predictions(binary, high_probability, ax=axes[1], name="High Occupancy")
        axes[1].set_title("Precision-Recall Curve")
        observed, predicted = calibration_curve(binary, high_probability, n_bins=8, strategy="quantile")
        axes[2].plot(predicted, observed, marker="o", label="Model")
        axes[2].plot([0, 1], [0, 1], "--", label="Perfect calibration")
        axes[2].set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="Calibration Curve")
        axes[2].legend()
        fig.tight_layout(); fig.savefig(directory / "roc_pr_calibration.png", dpi=180); plt.close(fig)

    @staticmethod
    def _learning_and_validation(result: TrainingResult, directory: Path) -> None:
        x = result.x_train.sample(min(3_500, len(result.x_train)), random_state=42)
        y = result.y_train.loc[x.index]
        train_sizes, train_scores, validation_scores = learning_curve(
            result.model, x, y, cv=3, scoring="f1_macro", train_sizes=np.linspace(0.25, 1.0, 4), n_jobs=-1
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train")
        axes[0].plot(train_sizes, validation_scores.mean(axis=1), marker="o", label="Validation")
        axes[0].set(title="Learning Curve", xlabel="Training examples", ylabel="Macro F1")
        axes[0].legend()
        model_step = result.model.named_steps["model"]
        if hasattr(model_step, "max_depth"):
            parameter = "model__max_depth"
            range_values = [4, 8, 12, 16]
            scores_train, scores_validation = validation_curve(result.model, x, y, param_name=parameter, param_range=range_values, cv=3, scoring="f1_macro", n_jobs=-1)
            axes[1].plot(range_values, scores_train.mean(axis=1), marker="o", label="Train")
            axes[1].plot(range_values, scores_validation.mean(axis=1), marker="o", label="Validation")
            axes[1].set(title="Validation Curve", xlabel="Max depth", ylabel="Macro F1")
            axes[1].legend()
        else:
            axes[1].text(0.1, 0.5, "Validation curve not applicable\nto selected estimator.", fontsize=12)
            axes[1].set_axis_off()
        fig.tight_layout(); fig.savefig(directory / "learning_validation_curves.png", dpi=180); plt.close(fig)

    @staticmethod
    def _importance(result: TrainingResult, directory: Path) -> pd.DataFrame:
        sample = result.x_test.sample(min(1_500, len(result.x_test)), random_state=42)
        labels = result.y_test.loc[sample.index]
        values = permutation_importance(result.model, sample, labels, scoring="f1_macro", n_repeats=5, random_state=42, n_jobs=-1)
        importance = pd.DataFrame({"feature": sample.columns, "importance_mean": values.importances_mean, "importance_std": values.importances_std}).sort_values("importance_mean", ascending=False)
        top = importance.head(15).sort_values("importance_mean")
        plt.figure(figsize=(9, 6))
        sns.barplot(data=top, x="importance_mean", y="feature", color="#2874A6")
        plt.title("Permutation Importance")
        plt.xlabel("Decrease in macro F1 after permutation")
        plt.tight_layout(); plt.savefig(directory / "permutation_importance.png", dpi=180); plt.close()
        return importance
