"""Data quality checks, leakage-safe transformations, and model preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


@dataclass
class DataPreprocessor:
    """Clean and diagnose hospital census data before feature engineering."""

    random_state: int = 42
    duplicate_count_: int = field(default=0, init=False)

    def clean_types_and_duplicates(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove duplicate records and safely coerce date, numeric,
        and categorical columns.
        """
        if dataframe.empty:
            raise ValueError("The input dataframe is empty.")

        frame = dataframe.copy()

        before_count = len(frame)

        frame = frame.drop_duplicates().reset_index(drop=True)

        self.duplicate_count_ = before_count - len(frame)

        if "Date" not in frame.columns:
            raise KeyError("Required column 'Date' is missing.")

        frame["Date"] = pd.to_datetime(
            frame["Date"],
            errors="coerce",
        )

        if frame["Date"].isna().any():
            invalid_count = int(frame["Date"].isna().sum())

            raise ValueError(
                f"{invalid_count} invalid date values were found "
                "during preprocessing."
            )

        numeric_columns = list(
            dict.fromkeys(
                NUMERIC_COLUMNS
                + ["Bed_Occupancy_Rate"]
            )
        )

        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        for column in CATEGORICAL_COLUMNS:
            if column in frame.columns:
                frame[column] = frame[column].astype(
                    "string"
                )

        sort_columns = [
            column
            for column in [
                "Hospital_ID",
                "Department",
                "Date",
            ]
            if column in frame.columns
        ]

        if sort_columns:
            frame = frame.sort_values(
                sort_columns
            )

        return frame.reset_index(drop=True)

    @staticmethod
    def iqr_outlier_summary(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """
        Calculate IQR-based outlier bounds, counts, and rates.

        The method reports outliers but does not automatically remove them.
        """
        rows: list[
            dict[str, float | str | int]
        ] = []

        available_columns = [
            column
            for column in columns
            if column in dataframe.columns
        ]

        for column in available_columns:
            series = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            if series.empty:
                rows.append(
                    {
                        "feature": column,
                        "lower_bound": np.nan,
                        "upper_bound": np.nan,
                        "outlier_count": 0,
                        "outlier_rate_pct": 0.0,
                    }
                )
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (
                (series < lower_bound)
                | (series > upper_bound)
            )

            rows.append(
                {
                    "feature": column,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_count": int(
                        outlier_mask.sum()
                    ),
                    "outlier_rate_pct": float(
                        100 * outlier_mask.mean()
                    ),
                }
            )

        return pd.DataFrame(rows)

    def isolation_forest_flags(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
        contamination: float = 0.02,
    ) -> pd.Series:
        """
        Flag multivariate anomalies using Isolation Forest.

        Missing values are median-imputed before anomaly detection.

        Args:
            dataframe: Input hospital dataframe.
            columns: Numeric columns used for anomaly detection.
            contamination: Expected proportion of anomalies.

        Returns:
            Boolean pandas Series where True indicates an outlier.
        """
        if not 0 < contamination <= 0.5:
            raise ValueError(
                "contamination must be greater than 0 "
                "and less than or equal to 0.5."
            )

        available_columns = [
            column
            for column in columns
            if column in dataframe.columns
        ]

        if not available_columns:
            return pd.Series(
                False,
                index=dataframe.index,
                name="IsolationForest_Outlier",
                dtype=bool,
            )

        features = dataframe[
            available_columns
        ].copy()

        features = features.apply(
            pd.to_numeric,
            errors="coerce",
        )

        features = features.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid_columns = [
            column
            for column in features.columns
            if features[column].notna().any()
        ]

        if not valid_columns:
            return pd.Series(
                False,
                index=dataframe.index,
                name="IsolationForest_Outlier",
                dtype=bool,
            )

        features = features[valid_columns]

        imputer = SimpleImputer(
            strategy="median",
        )

        imputed_features = imputer.fit_transform(
            features
        )

        if (
            imputed_features.shape[0] < 2
            or imputed_features.shape[1] == 0
        ):
            return pd.Series(
                False,
                index=dataframe.index,
                name="IsolationForest_Outlier",
                dtype=bool,
            )

        model = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_estimators=150,
            n_jobs=-1,
        )

        predictions = model.fit_predict(
            imputed_features
        )

        return pd.Series(
            predictions == -1,
            index=dataframe.index,
            name="IsolationForest_Outlier",
            dtype=bool,
        )

    @staticmethod
    def build_feature_preprocessor(
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> ColumnTransformer:
        """
        Create leakage-safe numerical and categorical preprocessing pipelines.

        Numeric features:
            Median imputation followed by robust scaling.

        Categorical features:
            Most-frequent imputation followed by one-hot encoding.
        """
        transformers: list[tuple] = []

        if numeric_features:
            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        RobustScaler(),
                    ),
                ]
            )

            transformers.append(
                (
                    "numeric",
                    numerical_pipeline,
                    numeric_features,
                )
            )

        if categorical_features:
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            )

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features,
                )
            )

        if not transformers:
            raise ValueError(
                "At least one numeric or categorical feature "
                "must be supplied."
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    @staticmethod
    def normalize_for_export(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """
        Add min-max normalized columns without replacing source values.

        New columns use the suffix `_Normalized`.
        """
        output = dataframe.copy()

        available_columns = [
            column
            for column in columns
            if column in output.columns
        ]

        for column in available_columns:
            values = pd.to_numeric(
                output[column],
                errors="coerce",
            ).astype(float)

            minimum = values.min()
            maximum = values.max()

            denominator = maximum - minimum

            normalized_column = (
                f"{column}_Normalized"
            )

            if (
                pd.isna(denominator)
                or denominator == 0
            ):
                output[normalized_column] = 0.0
            else:
                output[normalized_column] = (
                    values - minimum
                ) / denominator

        return output