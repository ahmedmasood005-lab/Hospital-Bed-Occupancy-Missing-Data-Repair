"""Central configuration for the hospital census project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Paths, seed, and modeling settings used throughout the application."""

    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    random_state: int = 42
    benchmark_rows: int = 1_600
    train_rows_for_cv: int = 5_000
    test_size: float = 0.25
    high_occupancy_threshold: float = 85.0

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root_dir / "data" / "processed"

    @property
    def engineered_dir(self) -> Path:
        return self.root_dir / "data" / "engineered"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.root_dir / "outputs" / "figures"

    @property
    def predictions_dir(self) -> Path:
        return self.root_dir / "outputs" / "predictions"

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / "outputs" / "logs"

    def create_directories(self) -> None:
        """Create all runtime output locations."""
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.engineered_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
            self.predictions_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


NUMERIC_COLUMNS = [
    "ICU_Beds",
    "General_Beds",
    "Emergency_Beds",
    "Occupied_Beds",
    "Available_Beds",
    "Admissions",
    "Discharges",
    "Doctors",
    "Nurses",
    "Staff_On_Duty",
    "Average_Length_of_Stay",
    "Emergency_Cases",
    "Covid_Cases",
]

CATEGORICAL_COLUMNS = [
    "Hospital_ID", "City", "Department", "Ward", "Weather", "Holiday", "Season"
]

TARGET_COLUMN = "Bed_Occupancy_Rate"
