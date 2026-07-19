"""Modern Tkinter dashboard for data repair, training, and local prediction."""

from __future__ import annotations

import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import joblib
import pandas as pd
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_hospital_data
from src.imputation import ImputationComparator
from src.utils import occupancy_class
from run_project import run
from .widgets import LogConsole, StatusBar


class HospitalDashboard(tk.Tk):
    """A complete desktop workflow for hospital data inspection and repair."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Hospital Bed Occupancy | Missing-Data Repair")
        self.geometry("1280x800")
        self.minsize(1040, 680)
        self.root_dir = PROJECT_ROOT
        self.data: pd.DataFrame | None = None
        self.current_path: Path | None = None
        self.theme_dark = True
        self.log_queue: queue.Queue[str] = queue.Queue()
        self._configure_style()
        self._build_layout()
        self.after(150, self._drain_logs)
        self._load_default_data()

    def _configure_style(self) -> None:
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self._apply_theme()

    def _apply_theme(self) -> None:
        background = "#111827" if self.theme_dark else "#F4F6F8"
        panel = "#1F2937" if self.theme_dark else "#FFFFFF"
        foreground = "#F9FAFB" if self.theme_dark else "#17202A"
        accent = "#2563EB"
        self.configure(bg=background)
        self.style.configure("TFrame", background=background)
        self.style.configure("Panel.TFrame", background=panel)
        self.style.configure("TLabel", background=background, foreground=foreground, font=("Segoe UI", 10))
        self.style.configure("Panel.TLabel", background=panel, foreground=foreground)
        self.style.configure("Title.TLabel", background=panel, foreground=foreground, font=("Segoe UI Semibold", 18))
        self.style.configure("TButton", font=("Segoe UI", 10), padding=7)
        self.style.configure("Accent.TButton", background=accent, foreground="white", font=("Segoe UI Semibold", 10), padding=8)
        self.style.map("Accent.TButton", background=[("active", "#1D4ED8")])
        self.style.configure("Treeview", background=panel, foreground=foreground, fieldbackground=panel, rowheight=26)
        self.style.configure("Treeview.Heading", background="#334155", foreground="white", font=("Segoe UI Semibold", 9))
        self.style.configure("TNotebook", background=background)
        self.style.configure("TNotebook.Tab", background=panel, foreground=foreground, padding=(12, 7))

    def _build_layout(self) -> None:
        sidebar = ttk.Frame(self, style="Panel.TFrame", width=225)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="Hospital Analytics", style="Title.TLabel").pack(anchor="w", padx=16, pady=(22, 3))
        ttk.Label(sidebar, text="Missing-data repair studio", style="Panel.TLabel").pack(anchor="w", padx=16, pady=(0, 20))
        actions = [
            ("Upload dataset", self.upload_dataset), ("Dataset preview", self.show_preview),
            ("Missing value viewer", self.show_missing), ("Visual dashboard", self.show_visuals),
            ("Imputation selection", self.show_imputation), ("Train model", self.train_model),
            ("Predict risk", self.predict_risk), ("Export clean dataset", self.export_clean_dataset),
            ("Export model", self.export_model), ("Generate report", self.generate_report),
        ]
        for text, command in actions:
            ttk.Button(sidebar, text=text, command=command).pack(fill="x", padx=12, pady=3)
        ttk.Separator(sidebar).pack(fill="x", padx=12, pady=12)
        ttk.Button(sidebar, text="Toggle dark / light", command=self.toggle_theme).pack(fill="x", padx=12, pady=3)
        ttk.Label(sidebar, text="Synthetic data only\nNot for clinical decision use", style="Panel.TLabel", justify="left").pack(side="bottom", anchor="w", padx=16, pady=16)
        main = ttk.Frame(self)
        main.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(main, style="Panel.TFrame")
        header.pack(fill="x")
        self.page_title = tk.StringVar(value="Overview")
        ttk.Label(header, textvariable=self.page_title, style="Title.TLabel").pack(side="left", padx=20, pady=16)
        self.progress = ttk.Progressbar(header, mode="indeterminate", length=180)
        self.progress.pack(side="right", padx=18)
        self.content = ttk.Frame(main)
        self.content.pack(fill="both", expand=True, padx=16, pady=14)
        self.console = LogConsole(main)
        self.console.pack(fill="x", padx=16, pady=(0, 8))
        self.status = StatusBar(main, style="Panel.TFrame")
        self.status.pack(fill="x")
        self.show_overview()

    def _clear_content(self, title: str) -> None:
        self.page_title.set(title)
        for child in self.content.winfo_children(): child.destroy()

    def show_overview(self) -> None:
        self._clear_content("Overview")
        card = ttk.Frame(self.content, style="Panel.TFrame")
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="Hospital Bed Occupancy Missing-Data Repair", style="Title.TLabel").pack(anchor="w", padx=28, pady=(28, 8))
        message = ("Upload an operational census CSV or use the generated dataset. Explore missingness, review the feature-specific imputation policy, run the complete reproducible pipeline, and export ready-to-share artifacts.")
        ttk.Label(card, text=message, style="Panel.TLabel", wraplength=780, justify="left").pack(anchor="w", padx=28, pady=8)
        button_bar = ttk.Frame(card, style="Panel.TFrame"); button_bar.pack(anchor="w", padx=28, pady=20)
        ttk.Button(button_bar, text="Run complete pipeline", style="Accent.TButton", command=self.train_model).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Open dataset preview", command=self.show_preview).pack(side="left")

    def _load_default_data(self) -> None:
        default = self.root_dir / "data" / "raw" / "hospital_census_missing.csv"
        if default.exists():
            self._set_data(default)
        else:
            self.log_queue.put("No generated dataset yet. Run the complete pipeline to create it.")

    def _set_data(self, path: Path) -> None:
        try:
            self.data = load_hospital_data(path)
            self.current_path = path
            self.status.set(f"Loaded {len(self.data):,} records from {path.name}")
            self.log_queue.put(f"Dataset loaded: {path}")
        except (OSError, ValueError) as error:
            messagebox.showerror("Dataset error", str(error))

    def upload_dataset(self) -> None:
        filename = filedialog.askopenfilename(title="Choose hospital census CSV", filetypes=[("CSV files", "*.csv")])
        if filename: self._set_data(Path(filename))

    def show_preview(self) -> None:
        self._clear_content("Dataset Preview")
        if self.data is None:
            ttk.Label(self.content, text="No data is loaded. Upload a CSV or run the pipeline.").pack(pady=30); return
        tree_frame = ttk.Frame(self.content); tree_frame.pack(fill="both", expand=True)
        columns = list(self.data.columns[:12])
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column); tree.column(column, width=130, stretch=False)
        for _, row in self.data.head(80).iterrows(): tree.insert("", "end", values=[str(row[column])[:28] for column in columns])
        ybar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview); xbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        tree.grid(row=0, column=0, sticky="nsew"); ybar.grid(row=0, column=1, sticky="ns"); xbar.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1); tree_frame.columnconfigure(0, weight=1)
        ttk.Label(self.content, text=f"Showing 80 rows and 12 columns of {len(self.data):,} records.").pack(anchor="w", pady=8)

    def show_missing(self) -> None:
        self._clear_content("Missing Value Viewer")
        if self.data is None: return
        summary = self.data.isna().sum().sort_values(ascending=False)
        for column, count in summary.items():
            row = ttk.Frame(self.content, style="Panel.TFrame"); row.pack(fill="x", pady=2)
            ttk.Label(row, text=column, style="Panel.TLabel", width=26).pack(side="left", padx=10, pady=7)
            ttk.Label(row, text=f"{count:,} missing ({100 * count / len(self.data):.2f}%)", style="Panel.TLabel").pack(side="left")
        ttk.Button(self.content, text="Open feature-specific recommendation", style="Accent.TButton", command=self.show_imputation).pack(anchor="w", pady=12)

    def show_visuals(self) -> None:
        self._clear_content("Visualization Dashboard")
        figures = sorted((self.root_dir / "outputs" / "figures").glob("*.png"))
        if not figures:
            ttk.Label(self.content, text="No figures yet. Run the pipeline first.").pack(pady=30); return
        canvas = tk.Canvas(self.content, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.content, orient="vertical", command=canvas.yview)
        container = ttk.Frame(canvas)
        container.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        self._images: list[ImageTk.PhotoImage] = []
        for source in figures[:8]:
            image = Image.open(source); image.thumbnail((720, 380))
            photo = ImageTk.PhotoImage(image); self._images.append(photo)
            ttk.Label(container, text=source.stem.replace("_", " ").title()).pack(anchor="w", padx=8, pady=(14, 2))
            ttk.Label(container, image=photo).pack(anchor="w", padx=8)

    def show_imputation(self) -> None:
        self._clear_content("Imputation Selection")
        recommendation_path = self.root_dir / "reports" / "imputation_recommendations.csv"
        if not recommendation_path.exists():
            ttk.Label(self.content, text="Recommendations are generated by the complete pipeline.").pack(pady=30); return
        table = pd.read_csv(recommendation_path)
        tree = ttk.Treeview(self.content, columns=list(table.columns), show="headings", height=16)
        for column in table.columns:
            tree.heading(column, text=column.replace("_", " ").title()); tree.column(column, width=150)
        for _, row in table.iterrows(): tree.insert("", "end", values=[f"{value:.4f}" if isinstance(value, float) else value for value in row])
        tree.pack(fill="both", expand=True)
        ttk.Label(self.content, text="Automatic best method: lowest composite reconstruction score, with downstream accuracy included.").pack(anchor="w", pady=8)

    def train_model(self) -> None:
        if self.progress["mode"] == "indeterminate" and self.progress.instate(["!disabled"]):
            # The check is harmless; worker lifecycle is stored independently.
            pass
        self.progress.start(12); self.status.set("Pipeline running: generating, repairing, modeling, reporting...")
        self.log_queue.put("Starting complete pipeline in a worker thread.")
        def worker() -> None:
            try:
                payload = run()
                self.log_queue.put(f"Pipeline succeeded. Model: {payload['best_model']} | accuracy: {payload['accuracy']:.3f}")
                self.after(0, lambda: self._pipeline_finished(True))
            except Exception as error:  # UI boundary intentionally catches and displays pipeline errors.
                self.log_queue.put(f"Pipeline failed: {error}")
                self.after(0, lambda: self._pipeline_finished(False))
        threading.Thread(target=worker, daemon=True).start()

    def _pipeline_finished(self, success: bool) -> None:
        self.progress.stop()
        if success:
            self._load_default_data(); self.status.set("Pipeline complete. Artifacts, report, and model are ready.")
            messagebox.showinfo("Pipeline complete", "Datasets, model, figures, metrics, and report were generated successfully.")
        else:
            self.status.set("Pipeline failed. Inspect the log console for details.")
            messagebox.showerror("Pipeline failed", "The pipeline did not finish. Read the logging window for the failure detail.")

    def predict_risk(self) -> None:
        model_path = self.root_dir / "models" / "best_model.pkl"
        data_path = self.root_dir / "data" / "engineered" / "hospital_census_engineered.csv"
        if not model_path.exists() or not data_path.exists():
            messagebox.showwarning("Prediction unavailable", "Run the complete pipeline before requesting a prediction."); return
        try:
            model = joblib.load(model_path)
            frame = pd.read_csv(data_path)
            features = list(model.feature_names_in_)
            record = frame.loc[[0], features]
            outcome = model.predict(record)[0]
            probability = float(model.predict_proba(record).max()) if hasattr(model, "predict_proba") else float("nan")
            messagebox.showinfo("Occupancy prediction", f"Preview record prediction: {outcome} occupancy risk\nModel confidence: {probability:.1%}\n\nUse an uploaded dataset and retrain to score its records with its own operational context.")
            self.log_queue.put(f"Predicted {outcome} occupancy risk for preview record.")
        except (OSError, ValueError, KeyError) as error:
            messagebox.showerror("Prediction error", str(error))

    def export_clean_dataset(self) -> None:
        source = self.root_dir / "data" / "processed" / "hospital_census_clean.csv"
        self._export_file(source, "Export clean repaired dataset")

    def export_model(self) -> None:
        source = self.root_dir / "models" / "best_model.pkl"
        self._export_file(source, "Export trained model")

    def generate_report(self) -> None:
        source = self.root_dir / "reports" / "Project_Report.pdf"
        self._export_file(source, "Export project report")

    def _export_file(self, source: Path, title: str) -> None:
        if not source.exists():
            messagebox.showwarning("Export unavailable", "Run the complete pipeline before exporting this artifact."); return
        destination = filedialog.asksaveasfilename(title=title, initialfile=source.name, defaultextension=source.suffix)
        if destination:
            Path(destination).write_bytes(source.read_bytes())
            self.status.set(f"Exported {source.name}")
            self.log_queue.put(f"Exported {source.name} to {destination}")

    def toggle_theme(self) -> None:
        self.theme_dark = not self.theme_dark; self._apply_theme(); self.status.set("Theme updated")

    def _drain_logs(self) -> None:
        try:
            while True: self.console.write(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(150, self._drain_logs)
