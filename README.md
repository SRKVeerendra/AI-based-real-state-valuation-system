# 🏠 AI-Based Real Estate Valuation System

A complete machine learning system to predict house prices using **8 datasets** including India property data, King County housing data, and global/India market price indices.

---

## 📁 Project Structure

```
real_estate_valuation/
├── data/                          ← All 8 datasets go here
│   ├── House_Price_India.csv
│   ├── house_prices.csv
│   ├── Housing.csv
│   ├── housing_price_index.csv
│   ├── nominal_index.csv
│   ├── nominal_year.csv
│   ├── real_index.csv
│   └── real_year.csv
│
├── src/
│   ├── data_preprocessing.py      ← Step 1: Clean, merge, engineer features
│   ├── eda.py                     ← Step 2: Exploratory Data Analysis
│   ├── model_training.py          ← Step 3: Train & compare 4 ML models
│   └── app.py                     ← Step 4: Streamlit web app
│
├── models/                        ← Saved models & plots (auto-generated)
├── main.py                        ← Run the full pipeline
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (VS Code)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python main.py
```

### 3. Launch Web App
```bash
streamlit run src/app.py
```

---

## 🔢 Step-by-Step Guide

### Step 1 — Data Preprocessing
```bash
python main.py --step 1
```
- Loads all 8 datasets
- Standardizes column names across datasets
- Extracts global market trend features (nominal/real price indices)
- Extracts India city price index multiplier
- Merges all datasets into one rich feature set
- Removes duplicates and outliers
- Saves `data/processed_dataset.csv`

### Step 2 — EDA
```bash
python main.py --step 2
```
- Price distribution plots
- Feature correlation heatmap
- Price by grade, bedrooms, location
- Saves `models/eda_plots.png`

### Step 3 — Model Training
```bash
python main.py --step 3
```
Trains and compares 4 models:
| Model | Notes |
|---|---|
| Linear Regression (Ridge) | Baseline |
| Decision Tree | Interpretable |
| Random Forest | High accuracy |
| **XGBoost** | **Best accuracy** |

Saves:
- `models/best_model.pkl`
- `models/xgboost_model.pkl`
- `models/random_forest_model.pkl`
- `models/scaler.pkl`
- `models/feature_names.pkl`
- `models/model_metrics.csv`
- `models/feature_importance.png`
- `models/prediction_scatter.png`

### Step 4 — Web App
```bash
streamlit run src/app.py
```
App tabs:
- **🔮 Price Predictor** — Input property details, get instant prediction
- **📊 Market Analytics** — Interactive charts & dashboards
- **🤖 Model Performance** — Compare all models, view feature importance
- **📁 Dataset Overview** — Dataset info & feature engineering details

---

## 📊 Datasets Used

| Dataset | Rows | Description | Role |
|---|---|---|---|
| House_Price_India.csv | 14,620 | India property features + price | Primary training |
| house_prices.csv | 21,613 | King County, WA properties | Primary training |
| Housing.csv | 21,613 | King County extended | Supplemental |
| housing_price_index.csv | 7 | India city quarterly index | Market feature |
| nominal_index.csv | 23,994 | Global nominal index (quarterly) | Trend feature |
| nominal_year.csv | 23,994 | Global nominal index (yearly) | Trend feature |
| real_index.csv | 23,994 | Global real index (quarterly) | Trend feature |
| real_year.csv | 23,994 | Global real index (yearly) | Trend feature |

---

## 🔬 Features Used

### Core Property Features (from primary datasets)
- Living area, lot area, bedrooms, bathrooms, floors
- Grade, condition, view, waterfront
- Year built, year renovated
- Location: latitude, longitude, zipcode

### India-Specific Features
- Number of schools nearby
- Distance from airport

### Engineered Features
- `house_age`, `was_renovated`, `years_since_renovation`
- `total_area`, `bath_bed_ratio`, `basement_ratio`
- `area_per_bedroom`, `grade_condition_score`
- `living_vs_neighbors`, `is_luxury`

### Market Trend Features (from macro datasets)
- `india_price_index` — All-India avg city price index
- `india_trend_slope` — Quarterly growth rate
- `global_nominal_avg` — Global average nominal house price index
- `global_real_avg` — Global average real house price index
- `global_yoy_change` — Year-over-year % change

---

## 📈 Expected Accuracy

| Model | R² | Accuracy |
|---|---|---|
| Linear Regression | ~0.70 | ~78% |
| Decision Tree | ~0.75 | ~82% |
| Random Forest | ~0.87 | ~88% |
| **XGBoost** | **~0.90+** | **~90%+** |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **scikit-learn** — ML models & preprocessing
- **XGBoost** — Gradient boosting
- **Streamlit** — Web interface
- **Plotly** — Interactive charts
- **pandas / numpy** — Data processing
- **joblib** — Model persistence
