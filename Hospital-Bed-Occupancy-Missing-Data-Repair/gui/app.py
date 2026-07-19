"""Desktop application entry point."""

from __future__ import annotations

from .dashboard import HospitalDashboard


def main() -> None:
    """Launch the hospital missing-data repair desktop application."""
    app = HospitalDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
