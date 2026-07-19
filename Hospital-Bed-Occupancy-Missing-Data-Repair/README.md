# Hospital Bed Occupancy Missing-Data Repair

An advanced, end-to-end healthcare analytics capstone that repairs incomplete hospital census data, compares 12 imputation techniques, recommends an imputation method per feature, and measures the impact on occupancy-risk machine learning.

> Synthetic data only. This project is a data-science portfolio and operational-planning demonstrator, not a clinical decision-support tool.

## Highlights

- Generates 17,520 realistic daily hospital-department census records across 16 hospitals.
- Injects and documents MCAR, MAR, and MNAR missingness patterns.
- Benchmarks mean, median, mode, forward/backward fill, three interpolation methods, KNN, MICE, regression, and random forest imputation.
- Scores repairs with RMSE, MAE, distribution drift, bias, variance, execution time, and downstream classification accuracy.
- Selects the best method for each feature rather than applying a global default.
- Builds leakage-aware temporal features and compares eight classifiers, including XGBoost when installed.
- Includes EDA, statistical testing, explainability, a modern Tkinter application, a PDF report, Excel metrics workbook, and six executable notebooks.

## Quick start

```powershell
git clone <your-repository-url>
cd Hospital-Bed-Occupancy-Missing-Data-Repair
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_project.py
python -m gui.app
```

The pipeline creates data, figures, metrics, model artifacts, predictions, and the report in their project folders. Use `python run_project.py --quick` for a faster smoke-test benchmark.

## Project workflow

```text
Synthetic census -> MCAR/MAR/MNAR masking -> quality checks -> per-feature imputation benchmark
      -> repaired data -> feature engineering -> temporal train/test split -> model comparison
      -> evaluation + explainability -> Tkinter dashboard + report + model exports
```

## Results snapshot

| Item | Result |
|---|---:|
| Complete daily records | 17,520 |
| Imputation techniques | 12 |
| Model candidates | 8 |
| Selected classifier | Extra Trees |
| Holdout macro F1 | 0.857 |
| Protected target | Bed Occupancy Rate |

The generated benchmark report contains the actual per-feature recommendation table. The output labels occupancy as **Low** (<70%), **Moderate** (70-84.99%), or **High** (>=85%).

## Screenshots

The project generates figures at runtime. Representative assets are stored in `outputs/figures/`:

- `missing_value_heatmap.svg` - missingness overview
- `occupancy_distribution.svg` - occupancy class distribution
- `monthly_seasonal_holiday_trends.svg` - seasonal capacity trends
- `permutation_importance.svg` - operational driver ranking

## Folder structure

```text
Hospital-Bed-Occupancy-Missing-Data-Repair/
├── data/
│   ├── raw/                 # complete and masked synthetic census data
│   ├── processed/           # repaired clean census data
│   └── engineered/          # feature-engineered modeling data
├── gui/                     # themed Tkinter desktop application
├── models/                  # best_model.pkl and scaler.pkl
├── notebooks/               # six notebook workflow stages
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── logs/
├── reports/                 # PDF, XLSX, metrics, recommendations, test outputs
├── scripts/                 # metrics workbook builder
├── src/                     # reusable OOP Python modules
├── run_project.py
├── requirements.txt
└── config.yaml
```

## Notebook sequence

1. `01_Data_Loading.ipynb` - source loading and schema review
2. `02_EDA.ipynb` - distribution, department, ward, and seasonal EDA
3. `03_Preprocessing.ipynb` - duplicates, quality checks, and imputation comparison
4. `04_Feature_Engineering.ipynb` - operational and time-series features
5. `05_Modeling.ipynb` - cross-validation and model tuning
6. `06_Evaluation.ipynb` - holdout performance and explainability

## GUI capabilities

The desktop application provides a professional sidebar workflow with dark/light themes, CSV upload, data preview, missingness viewer, chart dashboard, imputation recommendations, asynchronous model training, a prediction action, clean-data/model/report export, progress feedback, and an application log.

Launch it with:

```powershell
python -m gui.app
```

## Engineering notes

- The target is never used to fill input features and never masked during synthetic missingness injection.
- Feature generation orders records by hospital, department, and date. Lagged and rolling variables use historical observations only.
- The train/test split is chronological; cross-validation and tuning run only within the training period.
- The fitted pipeline owns preprocessing to prevent train/test leakage.
- SHAP is enabled when a compatible tree estimator and optional dependency are available; permutation importance and partial dependence are always attempted.

## Technologies

Python 3.12+, pandas, NumPy, SciPy, scikit-learn, XGBoost, matplotlib, seaborn, SHAP, Tkinter, joblib, Jupyter, python-docx, reportlab, and Git/GitHub.

## Future improvements

- Validate against governed real-world hospital feeds and local bed definitions.
- Add LightGBM/CatBoost under an approved deployment environment.
- Add drift monitoring, model registry integration, and role-based audit logs.
- Add probabilistic multiple imputation pooling and uncertainty intervals.
- Package the GUI as a signed desktop executable.

## License

Licensed under the [MIT License](LICENSE).
