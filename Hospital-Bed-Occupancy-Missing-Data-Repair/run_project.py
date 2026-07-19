"""Reproduce every dataset, model, figure, metric, and report deliverable."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.data_loader import SyntheticHospitalDataGenerator
from src.evaluation import ModelEvaluator
from src.explainability import ExplainabilitySuite
from src.feature_engineering import HospitalFeatureEngineer
from src.imputation import ImputationComparator
from src.modeling import OccupancyModelTrainer
from src.preprocessing import DataPreprocessor
from src.reporting import ProjectReportBuilder
from src.statistical_analysis import StatisticalAnalyzer
from src.utils import configure_logging, save_json, set_random_seed
from src.visualization import HospitalVisualizer


def run(skip_slow: bool = False) -> dict[str, object]:
    """Execute the end-to-end capstone workflow and return final metadata."""
    config = ProjectConfig()
    config.create_directories()
    logger = configure_logging(config.logs_dir)
    set_random_seed(config.random_state)
    logger.info("Generating complete and missing hospital census datasets.")
    generator = SyntheticHospitalDataGenerator(config)
    complete, raw_missing, missingness = generator.create_and_save()
    preprocessor = DataPreprocessor(config.random_state)
    raw_cleaned = preprocessor.clean_types_and_duplicates(raw_missing)
    logger.info("Benchmarking twelve imputation strategies on complete reference data.")
    comparator = ImputationComparator(config.random_state)
    benchmark_rows = 800 if skip_slow else config.benchmark_rows
    imputation_metrics, recommendations, method_summary = comparator.benchmark(complete, benchmark_rows=benchmark_rows)
    repaired = comparator.repair_with_recommendations(raw_cleaned, recommendations)
    repaired.to_csv(config.processed_dir / "hospital_census_clean.csv", index=False)
    imputation_metrics.to_csv(config.reports_dir / "imputation_metrics.csv", index=False)
    recommendations.to_csv(config.reports_dir / "imputation_recommendations.csv", index=False)
    method_summary.to_csv(config.reports_dir / "imputation_method_summary.csv", index=False)
    logger.info("Detecting anomalies, creating engineered data, and rendering EDA.")
    anomaly_columns = ["Admissions", "Discharges", "Nurses", "Staff_On_Duty", "Emergency_Cases", "Bed_Occupancy_Rate"]
    iqr = preprocessor.iqr_outlier_summary(repaired, anomaly_columns)
    repaired["IsolationForest_Outlier"] = preprocessor.isolation_forest_flags(repaired, anomaly_columns)
    engineered = HospitalFeatureEngineer().transform(repaired)
    engineered = preprocessor.normalize_for_export(engineered, anomaly_columns)
    engineered.to_csv(config.engineered_dir / "hospital_census_engineered.csv", index=False)
    iqr.to_csv(config.reports_dir / "outlier_iqr_summary.csv", index=False)
    stats = StatisticalAnalyzer.analyze(engineered)
    stats.to_csv(config.reports_dir / "statistical_results.csv", index=False)
    visualizer = HospitalVisualizer(config.figures_dir)
    visualizer.generate_eda(raw_missing, engineered)
    logger.info("Training and evaluating occupancy-risk model candidates.")
    trainer = OccupancyModelTrainer(config)
    result = trainer.train(engineered)
    trainer.save_artifacts(result, config.models_dir)
    result.comparison.to_csv(config.reports_dir / "model_comparison.csv", index=False)
    evaluator = ModelEvaluator()
    classification_report, importance = evaluator.evaluate(result, config.figures_dir, config.predictions_dir)
    classification_report.to_csv(config.reports_dir / "classification_report.csv", index=False)
    importance.to_csv(config.reports_dir / "permutation_importance.csv", index=False)
    explainability = ExplainabilitySuite().generate(result, config.figures_dir)
    best_imputation = method_summary.sort_values("rmse").iloc[0]["method"]
    payload: dict[str, object] = {
        "records": int(len(engineered)),
        "duplicates_removed": int(preprocessor.duplicate_count_),
        "best_imputation_method": str(best_imputation),
        "best_model": result.model_name,
        "accuracy": result.test_metrics["accuracy"],
        "f1_macro": result.test_metrics["f1_macro"],
        "recommendations": recommendations.to_dict(orient="records"),
        "missingness": missingness,
        "explainability": explainability,
        "test_metrics": result.test_metrics,
    }
    save_json(payload, config.reports_dir / "project_summary.json")
    report = ProjectReportBuilder(config.reports_dir, config.figures_dir)
    report.build(payload)
    logger.info("Pipeline complete. %s records repaired and modeled.", len(engineered))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hospital Bed Occupancy Missing-Data Repair project.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller benchmark sample for smoke tests.")
    args = parser.parse_args()
    run(skip_slow=args.quick)
