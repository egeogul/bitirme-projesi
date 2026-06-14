# Supply Chain Decision Support System

AI-powered supply chain decision support system built with LightGBM forecasting, inventory optimization, and an interactive Streamlit dashboard.

> Bitirme Projesi — 2026

---

## Overview

This project implements a complete demand forecasting and inventory optimization pipeline on the [Kaggle Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) dataset (913,000 daily sales records across 10 stores and 50 items, 2013–2017).

### Pipeline

```
Raw Data → EDA → Feature Engineering → LightGBM Training → Inventory Optimization → Dashboard
   01           02                    03                  04                        05
```

### Key Results

| Metric | Value |
|--------|-------|
| Model  | LightGBM |
| MAE    | 5.668 |
| RMSE   | 7.359 |
| MAPE   | 11.69% |
| SMAPE  | 11.27% |
| Train period | 2013–2016 |
| Test period  | 2017 |

---

## Project Structure

```
bitirme-projesi/
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # 41 lag/rolling/seasonal features
│   ├── 03_model_training.ipynb       # XGBoost & LightGBM training + evaluation
│   └── 04_stock_optimization.ipynb   # Safety stock, ROP, EOQ optimization
├── 05_dashboard.py                   # Streamlit dashboard (5 tabs)
├── data/
│   ├── raw/
│   │   ├── train.csv                 # 913K rows: date, store, item, sales
│   │   └── test.csv                  # Kaggle test set
│   └── processed/
│       ├── train_features.parquet    # Feature matrix (44 columns)
│       ├── test_predictions.parquet  # 2017 predictions + actuals
│       ├── feature_importance.csv    # Top features (gain)
│       ├── inventory_recommendations.parquet
│       ├── best_model.joblib         # Saved LightGBM model
│       └── model_meta.json           # Metrics, params, feature list
├── figures/                          # Exported thesis figures (PNG)
├── src/
│   ├── models/
│   ├── services/
│   ├── api/
│   └── utils/
├── config/
│   └── settings.py
├── scripts/
│   └── export_figures.py             # Regenerate thesis figures
├── tests/
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd bitirme-projesi
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add raw data

Download the dataset from Kaggle and place the files under `data/raw/`:

```
data/raw/train.csv
data/raw/test.csv
```

---

## Running the Notebooks

Run notebooks in order to reproduce the full pipeline:

```bash
jupyter notebook
```

| Notebook | Output |
|----------|--------|
| `01_eda.ipynb` | Exploratory plots, seasonality analysis |
| `02_feature_engineering.ipynb` | `data/processed/train_features.parquet` |
| `03_model_training.ipynb` | `best_model.joblib`, `test_predictions.parquet`, `feature_importance.csv`, `model_meta.json` |
| `04_stock_optimization.ipynb` | `inventory_recommendations.parquet` |

---

## Running the Dashboard

```bash
streamlit run 05_dashboard.py
```

The dashboard opens at `http://localhost:8501` and provides five tabs:

| Tab | Content |
|-----|---------|
| **Overview** | KPI cards, action distribution pie chart, store × item heatmap |
| **Forecast Charts** | Actual vs predicted (2017), residual distribution, monthly MAE/RMSE |
| **Inventory and Actions** | Stock gauge, action table sorted by priority, stock vs ROP scatter |
| **Risk Analysis** | Stockout probability, risk matrix, service-level sensitivity |
| **Model Information** | Test metrics, feature importance top-20, model parameters |

---

## Regenerating Thesis Figures

```bash
python scripts/export_figures.py
```

Saves 6 PNG files to `figures/`:

| File | Description |
|------|-------------|
| `01_daily_sales_trend.png` | Daily total sales + 30-day moving average (2013–2017) |
| `02_monthly_seasonality.png` | Monthly average sales bar chart |
| `03_acf_pacf.png` | ACF / PACF with lag 7/14/30 highlighted |
| `04_rolling_lag_features.png` | Rolling mean + lag features for Store 1 / Item 1 |
| `05_actual_vs_predicted.png` | Actual vs LightGBM forecast + residuals (2017) |
| `06_feature_importance_top20.png` | Top-20 features colored by category |

---

## Environment Variables

Create a `.env` file in the project root if needed:

```env
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
DATABASE_URL=sqlite:///./supply_chain.db
```
