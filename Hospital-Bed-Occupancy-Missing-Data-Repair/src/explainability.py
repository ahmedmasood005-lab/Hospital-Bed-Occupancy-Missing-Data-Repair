"""Feature attribution and partial dependence explanations with safe fallbacks."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

from .modeling import TrainingResult


class ExplainabilitySuite:
    """Create global tree importance, partial dependence, and SHAP artifacts when available."""

    def generate(self, result: TrainingResult, figure_dir: Path) -> dict[str, str]:
        """Produce explainability plots, returning paths and graceful availability notes."""
        figure_dir.mkdir(parents=True, exist_ok=True)
        messages: dict[str, str] = {}
        model = result.model
        numeric = result.numeric_features
        if numeric:
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                PartialDependenceDisplay.from_estimator(model, result.x_test.sample(min(1_000, len(result.x_test)), random_state=42), [numeric[0], numeric[min(1, len(numeric) - 1)]], target="High", ax=ax)
                fig.tight_layout(); path = figure_dir / "partial_dependence.png"; fig.savefig(path, dpi=180); plt.close(fig)
                messages["partial_dependence"] = str(path)
            except (ValueError, TypeError):
                messages["partial_dependence"] = "Unavailable for the selected estimator."
        try:
            import shap  # type: ignore
            transformed = model.named_steps["preprocess"].transform(result.x_test.sample(min(500, len(result.x_test)), random_state=42))
            estimator = model.named_steps["model"]
            if not hasattr(estimator, "feature_importances_"):
                raise TypeError("Selected estimator is not tree-based.")
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(transformed)
            values = shap_values[0] if isinstance(shap_values, list) else shap_values
            plt.figure(figsize=(10, 6))
            shap.summary_plot(values, transformed, show=False, max_display=15)
            path = figure_dir / "shap_summary.png"; plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
            messages["shap"] = str(path)
        except (ImportError, TypeError, ValueError, AttributeError):
            messages["shap"] = "SHAP optional dependency or compatible tree estimator not available."
        return messages
