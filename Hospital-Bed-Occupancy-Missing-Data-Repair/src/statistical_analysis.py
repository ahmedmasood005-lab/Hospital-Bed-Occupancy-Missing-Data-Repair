"""Statistical analysis utilities for hospital occupancy data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


class StatisticalAnalyzer:
    """Run inferential and distributional statistical tests."""

    @staticmethod
    def analyze(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Run statistical tests on hospital occupancy data.

        Returns:
            DataFrame containing test names, comparisons, statistics,
            p-values, and effect sizes where applicable.
        """
        if dataframe.empty:
            raise ValueError("Cannot analyze an empty dataframe.")

        results: list[dict[str, object]] = []

        occupancy_column = StatisticalAnalyzer._find_occupancy_column(
            dataframe
        )

        if occupancy_column is None:
            raise KeyError(
                "No occupancy column was found. Expected one of: "
                "'Bed_Occupancy_Rate', 'Occupancy_Percent', "
                "or 'Bed_Utilization_Percent'."
            )

        occupancy = pd.to_numeric(
            dataframe[occupancy_column],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if occupancy.empty:
            raise ValueError(
                f"Column '{occupancy_column}' contains no valid numeric values."
            )

        sample = occupancy.sample(
            n=min(5000, len(occupancy)),
            random_state=42,
        ).to_numpy(dtype=float)

        # Shapiro-Wilk works best with at most 5,000 observations.
        shapiro_sample = sample[:5000]

        if len(shapiro_sample) >= 3:
            statistic, p_value = stats.shapiro(shapiro_sample)

            results.append(
                {
                    "test": "Shapiro-Wilk",
                    "comparison": f"Normality of {occupancy_column}",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "effect_size": np.nan,
                }
            )

        if len(sample) >= 2 and np.std(sample, ddof=1) > 0:
            standardized_sample = (
                sample - np.mean(sample)
            ) / np.std(sample, ddof=1)

            statistic, p_value = stats.kstest(
                standardized_sample,
                "norm",
            )

            results.append(
                {
                    "test": "Kolmogorov-Smirnov",
                    "comparison": f"Normality of {occupancy_column}",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "effect_size": np.nan,
                }
            )

        if len(sample) >= 3:
            # SciPy 1.17+ emits a FutureWarning for default p-value behavior.
            # Anderson's current result provides statistic and critical values.
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    category=FutureWarning,
                )

                anderson_result = stats.anderson(
                    sample,
                    dist="norm",
                )

            results.append(
                {
                    "test": "Anderson-Darling",
                    "comparison": f"Normality of {occupancy_column}",
                    "statistic": float(anderson_result.statistic),
                    "p_value": np.nan,
                    "effect_size": np.nan,
                }
            )

        StatisticalAnalyzer._append_correlations(
            dataframe=dataframe,
            occupancy_column=occupancy_column,
            results=results,
        )

        StatisticalAnalyzer._append_department_tests(
            dataframe=dataframe,
            occupancy_column=occupancy_column,
            results=results,
        )

        StatisticalAnalyzer._append_holiday_tests(
            dataframe=dataframe,
            occupancy_column=occupancy_column,
            results=results,
        )

        StatisticalAnalyzer._append_chi_square_test(
            dataframe=dataframe,
            results=results,
        )

        return pd.DataFrame(results)

    @staticmethod
    def _append_correlations(
        dataframe: pd.DataFrame,
        occupancy_column: str,
        results: list[dict[str, object]],
    ) -> None:
        """Append Pearson and Spearman correlation tests."""
        candidate_columns = [
            "Admissions",
            "Discharges",
            "Emergency_Cases",
            "Covid_Cases",
            "Staff_On_Duty",
            "Available_Beds",
            "Occupied_Beds",
            "Average_Length_of_Stay",
        ]

        for column in candidate_columns:
            if column not in dataframe.columns:
                continue

            pair = dataframe[
                [column, occupancy_column]
            ].copy()

            pair[column] = pd.to_numeric(
                pair[column],
                errors="coerce",
            )

            pair[occupancy_column] = pd.to_numeric(
                pair[occupancy_column],
                errors="coerce",
            )

            pair = pair.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            if len(pair) < 3:
                continue

            if (
                pair[column].nunique() < 2
                or pair[occupancy_column].nunique() < 2
            ):
                continue

            pearson_statistic, pearson_p = stats.pearsonr(
                pair[column],
                pair[occupancy_column],
            )

            results.append(
                {
                    "test": "Pearson Correlation",
                    "comparison": (
                        f"{column} vs {occupancy_column}"
                    ),
                    "statistic": float(pearson_statistic),
                    "p_value": float(pearson_p),
                    "effect_size": float(
                        pearson_statistic ** 2
                    ),
                }
            )

            spearman_result = stats.spearmanr(
                pair[column],
                pair[occupancy_column],
                nan_policy="omit",
            )

            results.append(
                {
                    "test": "Spearman Correlation",
                    "comparison": (
                        f"{column} vs {occupancy_column}"
                    ),
                    "statistic": float(
                        spearman_result.statistic
                    ),
                    "p_value": float(
                        spearman_result.pvalue
                    ),
                    "effect_size": float(
                        spearman_result.statistic ** 2
                    ),
                }
            )

    @staticmethod
    def _append_department_tests(
        dataframe: pd.DataFrame,
        occupancy_column: str,
        results: list[dict[str, object]],
    ) -> None:
        """Append ANOVA and effect-size results across departments."""
        if "Department" not in dataframe.columns:
            return

        department_data = dataframe[
            ["Department", occupancy_column]
        ].copy()

        department_data[occupancy_column] = pd.to_numeric(
            department_data[occupancy_column],
            errors="coerce",
        )

        department_data = department_data.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        groups = [
            group[occupancy_column].to_numpy(dtype=float)
            for _, group in department_data.groupby(
                "Department",
                observed=True,
            )
            if len(group) >= 2
        ]

        if len(groups) < 2:
            return

        f_statistic, p_value = stats.f_oneway(
            *groups
        )

        results.append(
            {
                "test": "ANOVA",
                "comparison": "Occupancy across Departments",
                "statistic": float(f_statistic),
                "p_value": float(p_value),
                "effect_size": StatisticalAnalyzer._eta_squared(
                    groups
                ),
            }
        )

    @staticmethod
    def _append_holiday_tests(
        dataframe: pd.DataFrame,
        occupancy_column: str,
        results: list[dict[str, object]],
    ) -> None:
        """Append independent t-test and Mann-Whitney U test."""
        holiday_column = None

        for candidate in [
            "Holiday",
            "Holiday_Indicator",
        ]:
            if candidate in dataframe.columns:
                holiday_column = candidate
                break

        if holiday_column is None:
            return

        comparison_data = dataframe[
            [holiday_column, occupancy_column]
        ].copy()

        comparison_data[occupancy_column] = pd.to_numeric(
            comparison_data[occupancy_column],
            errors="coerce",
        )

        comparison_data = comparison_data.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        holiday_mask = StatisticalAnalyzer._holiday_mask(
            comparison_data[holiday_column]
        )

        holiday_values = comparison_data.loc[
            holiday_mask,
            occupancy_column,
        ].to_numpy(dtype=float)

        non_holiday_values = comparison_data.loc[
            ~holiday_mask,
            occupancy_column,
        ].to_numpy(dtype=float)

        if (
            len(holiday_values) < 2
            or len(non_holiday_values) < 2
        ):
            return

        t_statistic, t_p_value = stats.ttest_ind(
            holiday_values,
            non_holiday_values,
            equal_var=False,
            nan_policy="omit",
        )

        results.append(
            {
                "test": "Independent t-test",
                "comparison": (
                    "Holiday vs Non-Holiday Occupancy"
                ),
                "statistic": float(t_statistic),
                "p_value": float(t_p_value),
                "effect_size": StatisticalAnalyzer._cohens_d(
                    holiday_values,
                    non_holiday_values,
                ),
            }
        )

        mann_statistic, mann_p_value = stats.mannwhitneyu(
            holiday_values,
            non_holiday_values,
            alternative="two-sided",
        )

        results.append(
            {
                "test": "Mann-Whitney U",
                "comparison": (
                    "Holiday vs Non-Holiday Occupancy"
                ),
                "statistic": float(mann_statistic),
                "p_value": float(mann_p_value),
                "effect_size": StatisticalAnalyzer._rank_biserial(
                    mann_statistic,
                    len(holiday_values),
                    len(non_holiday_values),
                ),
            }
        )

    @staticmethod
    def _append_chi_square_test(
        dataframe: pd.DataFrame,
        results: list[dict[str, object]],
    ) -> None:
        """Append Chi-square association test for categorical variables."""
        if (
            "Department" not in dataframe.columns
            or "Occupancy_Class" not in dataframe.columns
        ):
            return

        contingency = pd.crosstab(
            dataframe["Department"],
            dataframe["Occupancy_Class"],
        )

        if (
            contingency.shape[0] < 2
            or contingency.shape[1] < 2
        ):
            return

        statistic, p_value, _, _ = stats.chi2_contingency(
            contingency
        )

        sample_size = contingency.to_numpy().sum()
        minimum_dimension = min(
            contingency.shape[0] - 1,
            contingency.shape[1] - 1,
        )

        if (
            sample_size > 0
            and minimum_dimension > 0
        ):
            cramers_v = np.sqrt(
                statistic
                / (
                    sample_size
                    * minimum_dimension
                )
            )
        else:
            cramers_v = np.nan

        results.append(
            {
                "test": "Chi-Square",
                "comparison": (
                    "Department vs Occupancy Class"
                ),
                "statistic": float(statistic),
                "p_value": float(p_value),
                "effect_size": float(cramers_v),
            }
        )

    @staticmethod
    def _eta_squared(
        groups: list[np.ndarray],
    ) -> float:
        """Calculate eta-squared effect size for one-way ANOVA."""
        valid_groups = [
            np.asarray(group, dtype=float)
            for group in groups
            if len(group) > 0
        ]

        if len(valid_groups) < 2:
            return float("nan")

        all_values = np.concatenate(
            valid_groups
        )

        if len(all_values) == 0:
            return float("nan")

        grand_mean = np.mean(all_values)

        between_group_sum = sum(
            len(group)
            * (
                np.mean(group) - grand_mean
            ) ** 2
            for group in valid_groups
        )

        total_sum = np.sum(
            (
                all_values - grand_mean
            ) ** 2
        )

        if total_sum == 0:
            return 0.0

        return float(
            between_group_sum / total_sum
        )

    @staticmethod
    def _cohens_d(
        first_group: np.ndarray,
        second_group: np.ndarray,
    ) -> float:
        """Calculate Cohen's d using pooled standard deviation."""
        first_group = np.asarray(
            first_group,
            dtype=float,
        )

        second_group = np.asarray(
            second_group,
            dtype=float,
        )

        first_size = len(first_group)
        second_size = len(second_group)

        if first_size < 2 or second_size < 2:
            return float("nan")

        pooled_variance = (
            (
                first_size - 1
            )
            * np.var(
                first_group,
                ddof=1,
            )
            + (
                second_size - 1
            )
            * np.var(
                second_group,
                ddof=1,
            )
        ) / (
            first_size
            + second_size
            - 2
        )

        if pooled_variance <= 0:
            return 0.0

        return float(
            (
                np.mean(first_group)
                - np.mean(second_group)
            )
            / np.sqrt(pooled_variance)
        )

    @staticmethod
    def _rank_biserial(
        u_statistic: float,
        first_size: int,
        second_size: int,
    ) -> float:
        """Calculate rank-biserial correlation from Mann-Whitney U."""
        denominator = (
            first_size * second_size
        )

        if denominator == 0:
            return float("nan")

        return float(
            1.0
            - (
                2.0
                * u_statistic
                / denominator
            )
        )

    @staticmethod
    def _holiday_mask(
        series: pd.Series,
    ) -> pd.Series:
        """Convert mixed holiday labels into a Boolean mask."""
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(
                series,
                errors="coerce",
            ).fillna(0)

            return numeric.astype(float) > 0

        normalized = (
            series.astype("string")
            .str.strip()
            .str.lower()
        )

        return normalized.isin(
            {
                "yes",
                "true",
                "1",
                "holiday",
                "public holiday",
            }
        )

    @staticmethod
    def _find_occupancy_column(
        dataframe: pd.DataFrame,
    ) -> str | None:
        """Return the first available occupancy measure."""
        candidates = [
            "Bed_Occupancy_Rate",
            "Occupancy_Percent",
            "Bed_Utilization_Percent",
            "Bed_Utilization_%",
            "Occupancy_%",
        ]

        for column in candidates:
            if column in dataframe.columns:
                return column

        return None