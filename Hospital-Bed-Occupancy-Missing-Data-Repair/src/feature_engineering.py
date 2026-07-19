"""Leakage-aware temporal and operational hospital feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


class HospitalFeatureEngineer:
    """Create interpretable features used by the occupancy risk classifier."""

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Engineer ratios, calendar signals, interactions, and lagged census features."""
        frame = dataframe.copy()
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.sort_values(["Hospital_ID", "Department", "Date"]).reset_index(drop=True)
        total_beds = frame[["ICU_Beds", "General_Beds", "Emergency_Beds"]].sum(axis=1).clip(lower=1)
        frame["Total_Beds"] = total_beds
        frame["Bed_Utilization_Pct"] = 100 * frame["Occupied_Beds"] / total_beds
        frame["Occupancy_Pct"] = frame["Bed_Utilization_Pct"]
        frame["Staff_per_Bed"] = frame["Staff_On_Duty"] / total_beds
        frame["Doctor_Ratio"] = frame["Doctors"] / total_beds
        frame["Nurse_Ratio"] = frame["Nurses"] / total_beds
        frame["Admissions_per_Bed"] = frame["Admissions"] / total_beds
        frame["Emergency_Load_Interaction"] = frame["Emergency_Cases"] * frame["Admissions"]
        frame["Staffing_Pressure_Interaction"] = frame["Occupied_Beds"] / frame["Staff_On_Duty"].clip(lower=1)
        frame["Admissions_Squared"] = frame["Admissions"] ** 2
        frame["Emergency_Cases_Squared"] = frame["Emergency_Cases"] ** 2
        frame["Month"] = frame["Date"].dt.month
        frame["Quarter"] = frame["Date"].dt.quarter
        frame["Day_of_Week"] = frame["Date"].dt.dayofweek
        frame["Weekend_Indicator"] = frame["Day_of_Week"].isin([5, 6]).astype(int)
        frame["Holiday_Indicator"] = frame["Holiday"].eq("Yes").astype(int)

        group_keys = ["Hospital_ID", "Department"]
        grouped_admissions = frame.groupby(group_keys, observed=True)["Admissions"]
        frame["Admissions_Lag_1"] = grouped_admissions.shift(1)
        frame["Admissions_Growth"] = grouped_admissions.pct_change().replace([np.inf, -np.inf], np.nan)
        frame["Admissions_Rolling_Mean_7"] = grouped_admissions.transform(
            lambda series: series.shift(1).rolling(7, min_periods=2).mean()
        )
        frame["Admissions_Rolling_Median_7"] = grouped_admissions.transform(
            lambda series: series.shift(1).rolling(7, min_periods=2).median()
        )
        frame["Admissions_Moving_Average_14"] = grouped_admissions.transform(
            lambda series: series.shift(1).rolling(14, min_periods=3).mean()
        )
        # Fill only temporal warm-up rows, retaining no future information.
        temporal_columns = [
            "Admissions_Lag_1",
            "Admissions_Growth",
            "Admissions_Rolling_Mean_7",
            "Admissions_Rolling_Median_7",
            "Admissions_Moving_Average_14",
        ]
        for column in temporal_columns:
            frame[column] = frame.groupby(group_keys, observed=True)[column].transform(
                lambda series: series.fillna(series.median())
            )
            frame[column] = frame[column].fillna(frame[column].median())
        return frame

    @staticmethod
    def model_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Return predictors that avoid direct target leakage from bed occupancy fields."""
        excluded = {
            "Date", "Bed_Occupancy_Rate", "Occupancy_Class", "Occupied_Beds",
            "Available_Beds", "Bed_Utilization_Pct", "Occupancy_Pct", "Total_Beds",
        }
        categorical = [
            column for column in ["Hospital_ID", "City", "Department", "Ward", "Weather", "Holiday", "Season"]
            if column in frame.columns
        ]
        numerical = [
            column for column in frame.select_dtypes(include=[np.number]).columns
            if column not in excluded and column not in categorical
        ]
        return numerical, categorical
