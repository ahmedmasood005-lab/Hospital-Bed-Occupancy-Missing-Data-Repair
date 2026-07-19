"""Professional, reproducible EDA and evaluation visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class HospitalVisualizer:
    """Write a concise portfolio-quality collection of analytical figures."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="deep")

    def generate_eda(self, raw: pd.DataFrame, clean: pd.DataFrame) -> list[Path]:
        """Generate missingness, distributions, operational comparisons, and time trends."""
        figures: list[Path] = []
        missing = raw.isna().mean().sort_values(ascending=False).to_frame("Missing Rate")
        path = self.output_dir / "missing_value_heatmap.png"
        plt.figure(figsize=(10, 6))
        sns.heatmap(raw.isna().sample(min(400, len(raw)), random_state=42).T, cbar=False, cmap=["#EAF2F8", "#C0392B"])
        plt.title("Missing Value Heatmap (400-record sample)")
        plt.xlabel("Record index")
        plt.ylabel("Feature")
        plt.tight_layout(); plt.savefig(path, dpi=170); plt.close(); figures.append(path)

        numeric = clean.select_dtypes(include="number").columns.tolist()
        correlation_columns = [c for c in numeric if c not in {"Month", "Quarter", "Day_of_Week"}][:16]
        path = self.output_dir / "correlation_heatmap.png"
        plt.figure(figsize=(14, 10))
        sns.heatmap(clean[correlation_columns].corr(), cmap="vlag", center=0, square=False)
        plt.title("Operational Feature Correlation Heatmap")
        plt.tight_layout(); plt.savefig(path, dpi=170); plt.close(); figures.append(path)

        path = self.output_dir / "occupancy_distribution.png"
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(clean["Bed_Occupancy_Rate"], kde=True, bins=35, ax=axes[0], color="#2874A6")
        axes[0].set_title("Occupancy Distribution")
        sns.boxplot(data=clean, x="Department", y="Bed_Occupancy_Rate", ax=axes[1])
        axes[1].set_title("Occupancy by Department")
        fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig); figures.append(path)

        path = self.output_dir / "department_ward_analysis.png"
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.violinplot(data=clean, x="Department", y="Bed_Occupancy_Rate", inner="quartile", ax=axes[0])
        axes[0].set_title("Department Occupancy Profile")
        sns.barplot(data=clean, x="Ward", y="Bed_Occupancy_Rate", estimator="mean", errorbar=None, ax=axes[1])
        axes[1].tick_params(axis="x", rotation=18)
        axes[1].set_title("Ward Mean Occupancy")
        fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig); figures.append(path)

        path = self.output_dir / "monthly_seasonal_holiday_trends.png"
        trend = clean.groupby("Date", as_index=False)["Bed_Occupancy_Rate"].mean()
        monthly = clean.groupby("Month", as_index=False)["Bed_Occupancy_Rate"].mean()
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        sns.lineplot(data=trend, x="Date", y="Bed_Occupancy_Rate", ax=axes[0], color="#1F618D")
        axes[0].set_title("Daily Census Trend")
        sns.lineplot(data=monthly, x="Month", y="Bed_Occupancy_Rate", marker="o", ax=axes[1])
        axes[1].set_title("Monthly Trend")
        sns.barplot(data=clean, x="Season", y="Bed_Occupancy_Rate", hue="Holiday", errorbar=None, ax=axes[2])
        axes[2].set_title("Seasonal and Holiday Trend")
        fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig); figures.append(path)

        path = self.output_dir / "staffing_scatter.png"
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=clean.sample(min(2500, len(clean)), random_state=42), x="Staff_On_Duty", y="Bed_Occupancy_Rate", hue="Department", alpha=0.55)
        plt.title("Staffing versus Occupancy")
        plt.tight_layout(); plt.savefig(path, dpi=170); plt.close(); figures.append(path)

        path = self.output_dir / "pairplot_sample.png"
        pair = clean[["Bed_Occupancy_Rate", "Admissions", "Nurses", "Emergency_Cases", "Average_Length_of_Stay"]].sample(min(500, len(clean)), random_state=42)
        grid = sns.pairplot(pair, corner=True, diag_kind="kde", plot_kws={"alpha": 0.45, "s": 12})
        grid.fig.suptitle("Pair Plot: Core Census Drivers", y=1.02)
        grid.fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(grid.fig); figures.append(path)

        missing.to_csv(self.output_dir / "missingness_summary.csv")
        return figures
