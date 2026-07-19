# 🏥 Hospital Bed Occupancy Missing-Data Repair Using Advanced Machine Learning

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Healthcare](https://img.shields.io/badge/Domain-Healthcare-success)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 📌 Project Overview

Hospital data frequently contains incomplete records due to manual entry errors, system failures, delayed reporting, or data integration issues. Missing values reduce the quality of predictive analytics and negatively affect healthcare decision-making.

This project presents a complete machine learning pipeline for repairing missing hospital bed occupancy data using multiple imputation techniques and evaluating their impact on predictive model performance.

The system automatically compares different imputation methods, recommends the most suitable technique for each feature, engineers meaningful healthcare features, trains multiple machine learning models, evaluates their performance, explains predictions using SHAP, and provides an interactive desktop application for end users.

The project follows modern Data Science and Machine Learning practices while maintaining a modular and reusable architecture.

---

# 🎯 Objectives

- Generate realistic synthetic hospital occupancy data
- Simulate MCAR, MAR, and MNAR missing-data mechanisms
- Compare multiple missing-data imputation techniques
- Automatically recommend the best imputation strategy
- Perform statistical analysis and exploratory data analysis
- Engineer advanced healthcare features
- Train and compare multiple machine learning models
- Explain predictions using SHAP
- Export reports and trained models
- Provide an easy-to-use desktop GUI

---

# 🚀 Key Features

## Data Generation

- Synthetic Hospital Dataset
- More than 17,000 hospital records
- Multiple hospital departments
- Daily hospital occupancy information
- Healthcare resource utilization

---

## Missing Data Simulation

Supports:

- MCAR (Missing Completely At Random)
- MAR (Missing At Random)
- MNAR (Missing Not At Random)

---

## Missing Data Imputation

The project compares 12 advanced imputation methods.

- Mean Imputation
- Median Imputation
- Mode Imputation
- Forward Fill
- Backward Fill
- Linear Interpolation
- Polynomial Interpolation
- Spline Interpolation
- KNN Imputation
- MICE (Iterative Imputer)
- Regression Imputation
- Random Forest Imputation

The system automatically recommends the best method for every feature.

---

## Feature Engineering

Engineered healthcare features include:

- Total Beds
- Available Beds
- Occupancy Rate
- Bed Utilization
- Staff per Bed
- Doctor Ratio
- Nurse Ratio
- Admission Growth
- Rolling Average
- Moving Average
- Lag Features
- Seasonal Features
- Weekend Indicator
- Holiday Indicator

---

## Exploratory Data Analysis

The project automatically generates:

- Missing value heatmaps
- Correlation matrix
- Distribution plots
- Histograms
- Boxplots
- Scatter plots
- Pair plots
- Occupancy trends
- Department comparisons

---

## Statistical Analysis

Implemented statistical techniques include:

- Pearson Correlation
- Spearman Correlation
- ANOVA
- Independent T-Test
- Mann-Whitney U Test
- Chi-Square Test
- Shapiro-Wilk Test
- Kolmogorov-Smirnov Test
- Anderson-Darling Test

---

## Outlier Detection

- IQR Method
- Isolation Forest

---

## Machine Learning Models

The project compares multiple ML algorithms.

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Extra Trees
- Support Vector Machine
- XGBoost (Optional)
- LightGBM (Optional)
- CatBoost (Optional)

---

## Model Evaluation

Performance metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- Cross Validation

---

## Explainable AI

The project provides model explainability using:

- SHAP
- Feature Importance
- Permutation Importance

---

## Desktop Application

Interactive Tkinter GUI includes:

- Dataset Upload
- Missing Value Analysis
- Automatic Data Repair
- Feature Engineering
- Model Training
- Prediction
- Report Generation
- Export Results

---

# 📂 Project Structure

```
Hospital-Bed-Occupancy-Missing-Data-Repair/
│
├── data/
├── src/
├── gui/
├── models/
├── notebooks/
├── reports/
├── outputs/
├── requirements.txt
├── config.yaml
├── run_project.py
└── README.md
```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- SciPy
- Matplotlib
- Seaborn
- SHAP
- Joblib
- Tkinter
- Jupyter Notebook
- ReportLab
- OpenPyXL

---

# ▶ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Hospital-Bed-Occupancy-Missing-Data-Repair.git
```

Move into the project

```bash
cd Hospital-Bed-Occupancy-Missing-Data-Repair
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run Project

Quick Mode

```bash
python run_project.py --quick
```

Full Pipeline

```bash
python run_project.py
```

Run GUI

```bash
python -m gui.app
```

or

```bash
python run_gui.py
```

---

# 📊 Generated Outputs

The project automatically generates:

- Processed Dataset
- Imputed Dataset
- Feature Engineered Dataset
- Visualizations
- Statistical Reports
- PDF Reports
- Excel Reports
- Trained Models
- Prediction Results

---

# 📈 Applications

- Hospital Resource Planning
- Bed Occupancy Forecasting
- Healthcare Analytics
- Data Cleaning
- Medical Research
- Healthcare Business Intelligence
- Predictive Analytics

---

# 📚 Future Improvements

- FastAPI Deployment
- MLflow Integration
- Docker Support
- Drift Monitoring
- Optuna Hyperparameter Optimization
- Cloud Deployment
- CI/CD Pipeline

---

# 👨‍💻 Author

## **Ahmed Masood**

**Data Science & Machine Learning Enthusiast**

- Python Developer
- Healthcare Data Analytics
- Machine Learning
- Artificial Intelligence
- Data Visualization

GitHub: https://github.com/YOUR_USERNAME

LinkedIn: https://linkedin.com/in/YOUR_LINKEDIN

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---

# 📜 License

This project is developed for educational, research, and portfolio purposes.

MIT License © 2026 Ahmed Masood
