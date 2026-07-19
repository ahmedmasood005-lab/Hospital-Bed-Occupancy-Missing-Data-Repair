"""Synthetic hospital census generation and robust CSV loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .utils import save_json


@dataclass
class SyntheticHospitalDataGenerator:
    """Generate realistic, reproducible daily hospital census observations."""

    config: ProjectConfig

    def generate_complete(self) -> pd.DataFrame:
        """Return 17,520 complete daily department census records."""
        rng = np.random.default_rng(self.config.random_state)
        hospitals = [f"HOSP-{index:03d}" for index in range(1, 17)]
        cities = [
            "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Peshawar"
        ]
        departments = ["Emergency", "Medical", "Surgical"]
        ward_lookup = {
            "Emergency": "Emergency Ward",
            "Medical": "General Medicine",
            "Surgical": "Surgical Ward",
        }
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        records: list[dict[str, object]] = []

        for hospital_index, hospital in enumerate(hospitals):
            city = cities[hospital_index % len(cities)]
            hospital_factor = rng.normal(0, 0.05)
            for department in departments:
                department_factor = {"Emergency": 0.12, "Medical": 0.03, "Surgical": 0.06}[department]
                for day_index, date in enumerate(dates):
                    seasonal_wave = 0.09 * np.sin(2 * np.pi * (day_index - 20) / 365)
                    weekday_effect = 0.045 if date.dayofweek < 5 else -0.03
                    is_holiday = date.strftime("%m-%d") in {"01-01", "03-23", "05-01", "08-14", "12-25"}
                    season = self._season(date.month)
                    weather = self._weather(season, rng)
                    weather_effect = {"Stormy": 0.07, "Rainy": 0.035, "Foggy": 0.025, "Clear": -0.01, "Hot": 0.02}[weather]
                    covid_wave = max(0, np.sin(2 * np.pi * (day_index - 65) / 170))
                    icu_beds = int(rng.integers(16, 42))
                    general_beds = int(rng.integers(80, 185))
                    emergency_beds = int(rng.integers(18, 55))
                    if department == "Emergency":
                        general_beds = int(general_beds * 0.45)
                    elif department == "Surgical":
                        general_beds = int(general_beds * 0.70)
                    total_beds = icu_beds + general_beds + emergency_beds
                    occupancy_prop = np.clip(
                        0.62 + hospital_factor + department_factor + seasonal_wave
                        + weekday_effect + weather_effect + (0.04 if is_holiday else 0)
                        + 0.07 * covid_wave + rng.normal(0, 0.045),
                        0.38,
                        0.98,
                    )
                    occupied = int(np.clip(np.round(total_beds * occupancy_prop), 1, total_beds))
                    available = total_beds - occupied
                    emergency_cases = int(max(3, rng.normal(26 + 34 * department_factor + 10 * weather_effect, 7)))
                    covid_cases = int(max(0, rng.poisson(2 + 12 * covid_wave)))
                    admissions = int(max(2, rng.normal(occupied * 0.14 + emergency_cases * 0.32, 8)))
                    discharges = int(max(1, rng.normal(admissions * 0.91, 7)))
                    doctors = int(max(4, round(total_beds / rng.uniform(13, 18))))
                    nurses = int(max(11, round(total_beds / rng.uniform(3.5, 5.2))))
                    staff = doctors + nurses + int(max(5, rng.normal(total_beds / 8, 4)))
                    alos = round(np.clip(rng.normal(4.2 + 0.8 * department_factor + 0.25 * covid_wave, 1.1), 1.2, 11.0), 2)
                    records.append(
                        {
                            "Hospital_ID": hospital,
                            "Date": date,
                            "City": city,
                            "Department": department,
                            "Ward": ward_lookup[department],
                            "ICU_Beds": icu_beds,
                            "General_Beds": general_beds,
                            "Emergency_Beds": emergency_beds,
                            "Occupied_Beds": occupied,
                            "Available_Beds": available,
                            "Admissions": admissions,
                            "Discharges": discharges,
                            "Doctors": doctors,
                            "Nurses": nurses,
                            "Staff_On_Duty": staff,
                            "Average_Length_of_Stay": alos,
                            "Emergency_Cases": emergency_cases,
                            "Covid_Cases": covid_cases,
                            "Weather": weather,
                            "Holiday": "Yes" if is_holiday else "No",
                            "Season": season,
                            "Bed_Occupancy_Rate": round(100 * occupied / total_beds, 2),
                        }
                    )
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _season(month: int) -> str:
        if month in (12, 1, 2):
            return "Winter"
        if month in (3, 4, 5):
            return "Spring"
        if month in (6, 7, 8):
            return "Summer"
        return "Autumn"

    @staticmethod
    def _weather(season: str, rng: np.random.Generator) -> str:
        probabilities = {
            "Winter": (["Clear", "Foggy", "Rainy", "Stormy", "Hot"], [0.28, 0.30, 0.28, 0.10, 0.04]),
            "Spring": (["Clear", "Foggy", "Rainy", "Stormy", "Hot"], [0.42, 0.08, 0.27, 0.11, 0.12]),
            "Summer": (["Clear", "Foggy", "Rainy", "Stormy", "Hot"], [0.24, 0.01, 0.19, 0.08, 0.48]),
            "Autumn": (["Clear", "Foggy", "Rainy", "Stormy", "Hot"], [0.49, 0.13, 0.22, 0.08, 0.08]),
        }
        choices, p = probabilities[season]
        return str(rng.choice(choices, p=p))

    def introduce_missingness(self, complete: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
        """Inject documented MCAR, MAR, and MNAR patterns without masking target labels."""
        rng = np.random.default_rng(self.config.random_state + 17)
        corrupted = complete.copy()
        patterns: dict[str, dict[str, object]] = {}

        def inject(column: str, eligible: pd.Series, rate: float, mechanism: str, rule: str) -> None:
            indices = corrupted.index[eligible]
            selected = rng.choice(indices, size=max(1, int(len(indices) * rate)), replace=False)
            corrupted.loc[selected, column] = np.nan
            patterns[column] = {
                "mechanism": mechanism,
                "target_missing_percentage": rate * 100,
                "realized_missing_rows": int(len(selected)),
                "rule": rule,
            }

        # MCAR: selection is independent of observed variables.
        inject("Admissions", pd.Series(True, index=corrupted.index), 0.07, "MCAR", "Uniform random 7% mask.")
        inject("Emergency_Cases", pd.Series(True, index=corrupted.index), 0.05, "MCAR", "Uniform random 5% mask.")
        inject("Weather", pd.Series(True, index=corrupted.index), 0.04, "MCAR", "Uniform random 4% mask.")
        # MAR: probability depends on observed ward/season.
        inject("Average_Length_of_Stay", corrupted["Department"].eq("Surgical"), 0.18, "MAR", "18% among Surgical records; 6% overall-scale MAR pattern.")
        inject("Doctors", corrupted["City"].isin(["Karachi", "Lahore"]), 0.10, "MAR", "10% in two high-volume cities.")
        inject("City", corrupted["Weather"].isin(["Rainy", "Stormy"]), 0.08, "MAR", "8% during wet-weather observations.")
        # MNAR: probability depends on the unobserved value's own magnitude.
        inject("Nurses", complete["Nurses"].ge(complete["Nurses"].quantile(0.72)), 0.22, "MNAR", "22% of high-nurse-count observations.")
        inject("Staff_On_Duty", complete["Staff_On_Duty"].ge(complete["Staff_On_Duty"].quantile(0.78)), 0.20, "MNAR", "20% of high staffing observations.")
        inject("Covid_Cases", complete["Covid_Cases"].ge(complete["Covid_Cases"].quantile(0.80)), 0.18, "MNAR", "18% of high COVID census observations.")

        # Add a small number of duplicates to validate duplicate removal in the pipeline.
        duplicates = corrupted.sample(n=24, random_state=self.config.random_state)
        corrupted = pd.concat([corrupted, duplicates], ignore_index=True)
        documentation: dict[str, object] = {
            "base_complete_records": int(len(complete)),
            "raw_records_including_duplicates": int(len(corrupted)),
            "duplicate_records_injected": int(len(duplicates)),
            "patterns": patterns,
            "target_protected": "Bed_Occupancy_Rate is never masked; it remains an outcome label.",
        }
        return corrupted, documentation

    def create_and_save(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        """Generate complete and corrupted variants and persist their provenance."""
        self.config.create_directories()
        complete = self.generate_complete()
        corrupted, documentation = self.introduce_missingness(complete)
        complete.to_csv(self.config.raw_dir / "hospital_census_complete.csv", index=False)
        corrupted.to_csv(self.config.raw_dir / "hospital_census_missing.csv", index=False)
        save_json(documentation, self.config.raw_dir / "missingness_documentation.json")
        return complete, corrupted, documentation


def load_hospital_data(path: str | Path) -> pd.DataFrame:
    """Load a hospital CSV and parse dates with clear failure messages."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset was not found: {source}")
    dataframe = pd.read_csv(source)
    if "Date" not in dataframe.columns:
        raise ValueError("Dataset must contain a Date column.")
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    if dataframe["Date"].isna().any():
        raise ValueError("Date conversion failed for one or more records.")
    return dataframe
