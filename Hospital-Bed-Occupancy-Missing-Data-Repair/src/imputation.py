"""Auditable comparison of advanced missing-data repair strategies."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from .utils import occupancy_class


METHODS = [
    "Mean",
    "Median",
    "Mode",
    "Forward Fill",
    "Backward Fill",
    "Linear Interpolation",
    "Polynomial Interpolation",
    "Spline Interpolation",
    "KNN Imputer",
    "Iterative Imputer (MICE)",
    "Regression Imputation",
    "Random Forest Imputation",
]


@dataclass
class ImputationComparator:
    """Compare missing-value repair methods against complete reference data."""

    random_state: int = 42

    group_columns: list[str] = field(
        default_factory=lambda: [
            "Hospital_ID",
            "Department",
        ]
    )

    strategy_by_feature_: dict[str, str] = field(
        default_factory=dict,
        init=False,
    )

    def _numeric_columns(
        self,
        frame: pd.DataFrame,
    ) -> list[str]:
        """Return configured numeric columns available in the dataframe."""
        return [
            column
            for column in NUMERIC_COLUMNS
            if column in frame.columns
        ]

    def _available_group_columns(
        self,
        frame: pd.DataFrame,
    ) -> list[str]:
        """Return grouping columns available in the dataframe."""
        return [
            column
            for column in self.group_columns
            if column in frame.columns
        ]

    @staticmethod
    def _fill_categories(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fill missing categorical values using each feature's mode."""
        result = frame.copy()

        categorical_columns = [
            column
            for column in CATEGORICAL_COLUMNS
            if column in result.columns
        ]

        for column in categorical_columns:
            mode_values = result[column].dropna().mode()

            fill_value = (
                mode_values.iloc[0]
                if not mode_values.empty
                else "Unknown"
            )

            result[column] = result[column].fillna(fill_value)

        return result

    @staticmethod
    def _safe_numeric_fallback(
        values: pd.DataFrame,
    ) -> pd.Series:
        """Create safe fallback values for numeric features."""
        medians = values.median(numeric_only=True)

        return medians.fillna(0.0)

    def impute(
        self,
        dataframe: pd.DataFrame,
        method: str,
    ) -> pd.DataFrame:
        """
        Repair numeric features using the selected imputation strategy.

        Categorical features are repaired using mode imputation.
        """
        if method not in METHODS:
            raise ValueError(
                f"Unsupported imputation method: {method}"
            )

        if dataframe.empty:
            return dataframe.copy()

        frame = dataframe.copy()

        sort_columns = self._available_group_columns(frame)

        if "Date" in frame.columns:
            sort_columns.append("Date")

        if sort_columns:
            frame = frame.sort_values(
                sort_columns
            ).reset_index(drop=True)
        else:
            frame = frame.reset_index(drop=True)

        numeric_columns = self._numeric_columns(frame)

        if not numeric_columns:
            return self._fill_categories(frame)

        values = frame[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        fallback_values = self._safe_numeric_fallback(values)

        if not values.isna().any().any():
            frame[numeric_columns] = values
            return self._fill_categories(frame)

        if method == "Mean":
            means = values.mean().fillna(fallback_values)

            frame[numeric_columns] = values.fillna(means)

        elif method == "Median":
            frame[numeric_columns] = values.fillna(
                fallback_values
            )

        elif method == "Mode":
            for column in numeric_columns:
                mode_values = values[column].dropna().mode()

                fill_value = (
                    float(mode_values.iloc[0])
                    if not mode_values.empty
                    else float(fallback_values[column])
                )

                frame[column] = values[column].fillna(
                    fill_value
                )

        elif method in {
            "Forward Fill",
            "Backward Fill",
        }:
            group_columns = self._available_group_columns(frame)

            if group_columns:
                if method == "Forward Fill":
                    repaired = frame.groupby(
                        group_columns,
                        observed=True,
                        dropna=False,
                    )[numeric_columns].transform(
                        lambda group: group.ffill()
                    )
                else:
                    repaired = frame.groupby(
                        group_columns,
                        observed=True,
                        dropna=False,
                    )[numeric_columns].transform(
                        lambda group: group.bfill()
                    )
            else:
                repaired = (
                    values.ffill()
                    if method == "Forward Fill"
                    else values.bfill()
                )

            frame[numeric_columns] = repaired.fillna(
                fallback_values
            )

        elif method in {
            "Linear Interpolation",
            "Polynomial Interpolation",
            "Spline Interpolation",
        }:
            interpolation_method = {
                "Linear Interpolation": "linear",
                "Polynomial Interpolation": "polynomial",
                "Spline Interpolation": "spline",
            }[method]

            repaired = values.copy()

            group_columns = self._available_group_columns(frame)

            if group_columns:
                grouped_positions = frame.groupby(
                    group_columns,
                    observed=True,
                    dropna=False,
                ).groups.values()
            else:
                grouped_positions = [frame.index]

            for positions in grouped_positions:
                group_index = list(positions)
                subset = repaired.loc[group_index]

                try:
                    interpolation_arguments = {}

                    if interpolation_method in {
                        "polynomial",
                        "spline",
                    }:
                        interpolation_arguments["order"] = 2

                    repaired.loc[group_index] = subset.interpolate(
                        method=interpolation_method,
                        limit_direction="both",
                        **interpolation_arguments,
                    )

                except (
                    ValueError,
                    ImportError,
                    TypeError,
                ):
                    repaired.loc[group_index] = subset.interpolate(
                        method="linear",
                        limit_direction="both",
                    )

            frame[numeric_columns] = repaired.fillna(
                fallback_values
            )

        elif method == "KNN Imputer":
            imputer = KNNImputer(
                n_neighbors=5,
                weights="distance",
            )

            transformed_values = imputer.fit_transform(values)

            frame[numeric_columns] = pd.DataFrame(
                transformed_values,
                columns=numeric_columns,
                index=frame.index,
            )

        elif method == "Iterative Imputer (MICE)":
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=10,
                random_state=self.random_state,
                initial_strategy="median",
                skip_complete=True,
            )

            transformed_values = imputer.fit_transform(values)

            frame[numeric_columns] = pd.DataFrame(
                transformed_values,
                columns=numeric_columns,
                index=frame.index,
            )

        elif method == "Regression Imputation":
            frame[numeric_columns] = self._regression_impute(
                values
            )

        elif method == "Random Forest Imputation":
            frame[numeric_columns] = self._forest_impute(
                values
            )

        frame[numeric_columns] = frame[
            numeric_columns
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        frame[numeric_columns] = frame[
            numeric_columns
        ].fillna(
            fallback_values
        )

        return self._fill_categories(frame)

    def _regression_impute(
        self,
        values: pd.DataFrame,
    ) -> pd.DataFrame:
        """Repair missing values through feature-wise ridge regression."""
        repaired = values.copy()

        fallback_values = self._safe_numeric_fallback(values)

        base = values.fillna(
            fallback_values
        )

        for column in values.columns:
            missing_mask = values[column].isna()

            if not missing_mask.any():
                continue

            predictor_columns = [
                predictor
                for predictor in values.columns
                if predictor != column
            ]

            observed_mask = ~missing_mask

            if (
                observed_mask.sum() < 10
                or not predictor_columns
            ):
                repaired.loc[
                    missing_mask,
                    column,
                ] = fallback_values[column]

                continue

            model = Pipeline(
                steps=[
                    (
                        "scale",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        Ridge(alpha=1.0),
                    ),
                ]
            )

            model.fit(
                base.loc[
                    observed_mask,
                    predictor_columns,
                ],
                values.loc[
                    observed_mask,
                    column,
                ],
            )

            repaired.loc[
                missing_mask,
                column,
            ] = model.predict(
                base.loc[
                    missing_mask,
                    predictor_columns,
                ]
            )

        return repaired.fillna(
            fallback_values
        )

    def _forest_impute(
        self,
        values: pd.DataFrame,
    ) -> pd.DataFrame:
        """Repair missing values using feature-wise random forest models."""
        repaired = values.copy()

        fallback_values = self._safe_numeric_fallback(values)

        base = values.fillna(
            fallback_values
        )

        for offset, column in enumerate(values.columns):
            missing_mask = values[column].isna()

            if not missing_mask.any():
                continue

            predictor_columns = [
                predictor
                for predictor in values.columns
                if predictor != column
            ]

            observed_mask = ~missing_mask

            if (
                observed_mask.sum() < 25
                or not predictor_columns
            ):
                repaired.loc[
                    missing_mask,
                    column,
                ] = fallback_values[column]

                continue

            model = RandomForestRegressor(
                n_estimators=70,
                max_depth=10,
                min_samples_leaf=2,
                random_state=self.random_state + offset,
                n_jobs=-1,
            )

            model.fit(
                base.loc[
                    observed_mask,
                    predictor_columns,
                ],
                values.loc[
                    observed_mask,
                    column,
                ],
            )

            repaired.loc[
                missing_mask,
                column,
            ] = model.predict(
                base.loc[
                    missing_mask,
                    predictor_columns,
                ]
            )

        return repaired.fillna(
            fallback_values
        )

    def benchmark(
        self,
        complete_data: pd.DataFrame,
        benchmark_rows: int = 1_600,
        mask_rate: float = 0.12,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Mask known values and compare all imputation strategies.

        Returns:
            Detailed metrics, feature recommendations and method summary.
        """
        if complete_data.empty:
            raise ValueError(
                "Cannot benchmark an empty dataframe."
            )

        if not 0 < mask_rate < 1:
            raise ValueError(
                "mask_rate must be between 0 and 1."
            )

        random_generator = np.random.default_rng(
            self.random_state
        )

        sample_size = min(
            benchmark_rows,
            len(complete_data),
        )

        base = complete_data.sample(
            n=sample_size,
            random_state=self.random_state,
        ).copy()

        sort_columns = self._available_group_columns(base)

        if "Date" in base.columns:
            sort_columns.append("Date")

        if sort_columns:
            base = base.sort_values(
                sort_columns
            )

        base = base.reset_index(drop=True)

        numeric_columns = self._numeric_columns(base)

        if not numeric_columns:
            raise ValueError(
                "No numeric columns are available for benchmarking."
            )

        base[numeric_columns] = base[
            numeric_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        complete_numeric_columns = [
            column
            for column in numeric_columns
            if base[column].notna().sum() >= 2
        ]

        if not complete_numeric_columns:
            raise ValueError(
                "No numeric features contain enough known values "
                "for benchmarking."
            )

        masked = base.copy()

        masks: dict[str, np.ndarray] = {}

        for column in complete_numeric_columns:
            available_positions = np.flatnonzero(
                base[column].notna().to_numpy()
            )

            mask_count = max(
                1,
                int(len(available_positions) * mask_rate),
            )

            mask_count = min(
                mask_count,
                len(available_positions),
            )

            selected_positions = random_generator.choice(
                available_positions,
                size=mask_count,
                replace=False,
            )

            feature_mask = np.zeros(
                len(base),
                dtype=bool,
            )

            feature_mask[selected_positions] = True

            masks[column] = feature_mask

            masked.loc[
                feature_mask,
                column,
            ] = np.nan

        metric_rows: list[dict[str, object]] = []

        for method in METHODS:
            started_time = time.perf_counter()

            repaired = self.impute(
                masked,
                method,
            )

            elapsed_time = (
                time.perf_counter()
                - started_time
            )

            downstream_accuracy = (
                self._downstream_accuracy(repaired)
            )

            for column in complete_numeric_columns:
                feature_mask = masks[column]

                actual = base.loc[
                    feature_mask,
                    column,
                ].to_numpy(dtype=float)

                predicted = repaired.loc[
                    feature_mask,
                    column,
                ].to_numpy(dtype=float)

                valid_mask = (
                    np.isfinite(actual)
                    & np.isfinite(predicted)
                )

                actual = actual[valid_mask]
                predicted = predicted[valid_mask]

                if len(actual) == 0:
                    continue

                errors = predicted - actual

                metric_rows.append(
                    {
                        "feature": column,
                        "method": method,
                        "rmse": float(
                            np.sqrt(
                                mean_squared_error(
                                    actual,
                                    predicted,
                                )
                            )
                        ),
                        "mae": float(
                            mean_absolute_error(
                                actual,
                                predicted,
                            )
                        ),
                        "distribution_drift": float(
                            wasserstein_distance(
                                actual,
                                predicted,
                            )
                        ),
                        "bias": float(
                            np.mean(errors)
                        ),
                        "absolute_bias": float(
                            abs(np.mean(errors))
                        ),
                        "variance": float(
                            np.var(predicted)
                        ),
                        "execution_time_seconds": float(
                            elapsed_time
                        ),
                        "downstream_model_accuracy": float(
                            downstream_accuracy
                        ),
                        "masked_values": int(
                            feature_mask.sum()
                        ),
                    }
                )

        metrics = pd.DataFrame(metric_rows)

        if metrics.empty:
            raise RuntimeError(
                "No imputation metrics were generated."
            )

        ranked = metrics.copy()

        lower_is_better_columns = [
            "rmse",
            "mae",
            "distribution_drift",
            "absolute_bias",
            "execution_time_seconds",
        ]

        for column in lower_is_better_columns:
            ranked[f"{column}_rank"] = ranked.groupby(
                "feature"
            )[column].rank(
                method="average",
                pct=True,
                ascending=True,
            )

        ranked["accuracy_rank"] = ranked.groupby(
            "feature"
        )["downstream_model_accuracy"].rank(
            method="average",
            pct=True,
            ascending=False,
        )

        ranked["composite_score"] = (
            0.34 * ranked["rmse_rank"]
            + 0.22 * ranked["mae_rank"]
            + 0.16 * ranked["distribution_drift_rank"]
            + 0.10 * ranked["absolute_bias_rank"]
            + 0.08 * ranked[
                "execution_time_seconds_rank"
            ]
            + 0.10 * ranked["accuracy_rank"]
        )

        best_indices = ranked.groupby(
            "feature"
        )["composite_score"].idxmin()

        recommendations = ranked.loc[
            best_indices,
            [
                "feature",
                "method",
                "composite_score",
                "rmse",
                "mae",
                "distribution_drift",
                "bias",
                "absolute_bias",
                "variance",
                "execution_time_seconds",
                "downstream_model_accuracy",
            ],
        ].sort_values(
            "feature"
        ).reset_index(drop=True)

        self.strategy_by_feature_ = dict(
            zip(
                recommendations["feature"],
                recommendations["method"],
            )
        )

        method_summary = (
            metrics.groupby(
                "method",
                as_index=False,
            )
            .agg(
                rmse=("rmse", "mean"),
                mae=("mae", "mean"),
                distribution_drift=(
                    "distribution_drift",
                    "mean",
                ),
                bias=("bias", "mean"),
                absolute_bias=(
                    "absolute_bias",
                    "mean",
                ),
                variance=("variance", "mean"),
                mean_execution_time_seconds=(
                    "execution_time_seconds",
                    "mean",
                ),
                downstream_model_accuracy=(
                    "downstream_model_accuracy",
                    "mean",
                ),
            )
            .sort_values(
                [
                    "rmse",
                    "mae",
                ]
            )
            .reset_index(drop=True)
        )

        return (
            metrics,
            recommendations,
            method_summary,
        )

    def repair_with_recommendations(
        self,
        data: pd.DataFrame,
        recommendations: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Repair every feature using its benchmark-selected strategy.

        This avoids applying one imputation method to every feature.
        """
        required_columns = {
            "feature",
            "method",
        }

        if not required_columns.issubset(
            recommendations.columns
        ):
            raise ValueError(
                "Recommendations must contain "
                "'feature' and 'method' columns."
            )

        frame = data.copy()

        for method in sorted(
            recommendations["method"].dropna().unique()
        ):
            feature_columns = recommendations.loc[
                recommendations["method"].eq(method),
                "feature",
            ].tolist()

            feature_columns = [
                column
                for column in feature_columns
                if column in frame.columns
            ]

            if not feature_columns:
                continue

            repaired = self.impute(
                frame,
                method,
            )

            frame[feature_columns] = repaired[
                feature_columns
            ]

        return self._fill_categories(frame)

    def _downstream_accuracy(
        self,
        repaired: pd.DataFrame,
    ) -> float:
        """
        Measure occupancy-risk classification performance after repair.

        The target is converted into occupancy risk classes and evaluated
        with a preprocessing and logistic-regression pipeline.
        """
        frame = repaired.copy()

        if "Bed_Occupancy_Rate" not in frame.columns:
            return 0.0

        frame["Occupancy_Class"] = frame[
            "Bed_Occupancy_Rate"
        ].map(
            occupancy_class
        )

        candidate_numeric = [
            "ICU_Beds",
            "General_Beds",
            "Emergency_Beds",
            "Admissions",
            "Discharges",
            "Doctors",
            "Nurses",
            "Staff_On_Duty",
            "Average_Length_of_Stay",
            "Emergency_Cases",
            "Covid_Cases",
        ]

        candidate_categorical = [
            "City",
            "Department",
            "Weather",
            "Holiday",
            "Season",
        ]

        numeric_columns = [
            column
            for column in candidate_numeric
            if column in frame.columns
        ]

        categorical_columns = [
            column
            for column in candidate_categorical
            if column in frame.columns
        ]

        feature_columns = (
            numeric_columns
            + categorical_columns
        )

        if not feature_columns:
            return 0.0

        frame = frame[
            feature_columns
            + ["Occupancy_Class"]
        ].copy()

        frame = frame.dropna(
            subset=["Occupancy_Class"]
        )

        if frame.empty:
            return 0.0

        for column in numeric_columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            median_value = frame[column].median()

            if pd.isna(median_value):
                median_value = 0.0

            frame[column] = frame[column].fillna(
                median_value
            )

        for column in categorical_columns:
            mode_values = frame[column].dropna().mode()

            fill_value = (
                mode_values.iloc[0]
                if not mode_values.empty
                else "Unknown"
            )

            frame[column] = frame[column].fillna(
                fill_value
            ).astype(str)

        class_counts = frame[
            "Occupancy_Class"
        ].value_counts()

        if len(class_counts) < 2:
            return 0.0

        minimum_test_rows = len(class_counts)

        test_size = max(
            0.25,
            minimum_test_rows / len(frame),
        )

        test_size = min(
            test_size,
            0.40,
        )

        use_stratification = (
            class_counts.min() >= 2
            and int(len(frame) * test_size)
            >= len(class_counts)
        )

        transformers = []

        if numeric_columns:
            transformers.append(
                (
                    "num",
                    StandardScaler(),
                    numeric_columns,
                )
            )

        if categorical_columns:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                    categorical_columns,
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        features = frame[feature_columns]
        target = frame["Occupancy_Class"]

        try:
            (
                x_train,
                x_test,
                y_train,
                y_test,
            ) = train_test_split(
                features,
                target,
                test_size=test_size,
                random_state=self.random_state,
                stratify=(
                    target
                    if use_stratification
                    else None
                ),
            )
        except ValueError:
            return 0.0

        if y_train.nunique() < 2:
            return 0.0

        model = Pipeline(
            steps=[
                (
                    "prepare",
                    preprocessor,
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        solver="lbfgs",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        try:
            model.fit(
                x_train,
                y_train,
            )

            predictions = model.predict(
                x_test
            )

            return float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            )

        except (
            ValueError,
            TypeError,
            RuntimeError,
        ):
            return 0.0