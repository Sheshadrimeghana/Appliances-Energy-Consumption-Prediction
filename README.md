# Household Energy Consumption ML Project

## 1. Project Overview
This project predicts `total_energy_use` (Wh) using household lighting usage, indoor temperature/humidity sensors, outdoor weather variables, and timestamp-derived features.

## 2. Dataset
- Rows: 19,735
- Original columns: 42
- Completely empty columns removed: 13
- Target: `total_energy_use`
- Problem type: Regression

## 3. ML Workflow
1. Load the supplied CSV.
2. Remove completely empty columns.
3. Check missing values and duplicates.
4. Parse `recorded_timestamp`.
5. Create hour/day/month/weekend and cyclic time features.
6. Apply `log1p` transformation to lighting energy.
7. Handle missing values with imputation.
8. One-hot encode `day_of_week`.
9. Standardize numerical features.
10. Detect outliers using Isolation Forest.
11. Treat extreme numerical predictor values with training-set 1st/99th percentile clipping.
12. Select the 15 strongest transformed features with SelectKBest.
13. Train/test split: 80/20, random_state=42.
14. Train Linear Regression, Random Forest, and Gradient Boosting.
15. Compare MAE, RMSE, and R².
16. Save the best model.

## 4. Results
| Model | MAE (Wh) | RMSE (Wh) | R² |
|---|---:|---:|---:|
| Random Forest | 32.57 | 67.36 | 0.547 |
| Gradient Boosting | 45.91 | 84.35 | 0.289 |
| Linear Regression | 52.66 | 90.26 | 0.186 |

The lowest test RMSE in this experiment was achieved by **Random Forest**.

## 5. Project Structure
```text
ML_Energy_Consumption_Project/
├── data/
│   ├── energy_consumption_dataset.csv
│   └── energy_consumption_cleaned.csv
├── notebook/
│   └── ML_Energy_Consumption_Project.ipynb
├── artifacts/
│   └── energy_model.joblib
├── outputs/
│   ├── model_metrics.csv
│   ├── feature_importance.csv
│   ├── target_distribution.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   └── feature_importance.png
├── src/
│   ├── predict.py
│   └── app.py
├── ML_Energy_Consumption_Summary_Report.pdf
├── requirements.txt
└── README.md
```

## 6. How to Run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook notebook/ML_Energy_Consumption_Project.ipynb
```

Run all notebook cells from top to bottom.

## 7. API
The optional Flask API is in `src/app.py`. After installing dependencies:

```bash
python src/app.py
```

The endpoint is:
`POST /predict`

A sample request is documented in `src/predict.py`.

## 8. Important Note
The reported metrics use a random 80/20 split. For a production forecasting system, use time-based validation so future observations are not mixed with past observations during evaluation.

## 9. Future Extensions
- LSTM/Transformer time-series forecasting
- Real-time Flask/FastAPI prediction service
- Energy monitoring dashboard
