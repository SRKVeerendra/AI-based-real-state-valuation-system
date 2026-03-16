"""
STEP 3: Model Training, Development & Evaluation
AI-Based Real Estate Valuation System

Models trained:
  1. Linear Regression   (baseline)
  2. Decision Tree
  3. Random Forest
  4. XGBoost             (best accuracy)

Best model saved to models/best_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection   import train_test_split, cross_val_score, KFold
from sklearn.preprocessing     import StandardScaler, RobustScaler
from sklearn.linear_model      import LinearRegression, Ridge
from sklearn.tree              import DecisionTreeRegressor
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics           import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline          import Pipeline
import xgboost as xgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR   = "data/"
MODELS_DIR = "models/"

# ─────────────────────────────────────────────
# FEATURE SET  (chosen from EDA correlations)
# ─────────────────────────────────────────────
FEATURES = [
    # Core property features
    'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement',
    'bedrooms', 'bathrooms', 'floors',
    'grade', 'condition', 'view', 'waterfront',
    'yr_built', 'yr_renovated',
    'sqft_living15', 'sqft_lot15',
    'lat', 'long', 'zipcode',

    # India-specific features
    'schools_nearby', 'airport_distance',

    # Engineered features
    'house_age', 'was_renovated', 'years_since_renovation',
    'total_area', 'bath_bed_ratio', 'basement_ratio',
    'living_lot_ratio', 'area_per_bedroom', 'area_per_bathroom',
    'grade_condition_score', 'living_vs_neighbors', 'lot_vs_neighbors',
    'is_luxury',

    # Market trend features (from macro datasets)
    'india_price_index', 'india_trend_slope',
    'global_nominal_avg', 'global_real_avg', 'global_yoy_change',
    'is_india_dataset', 'has_school_dist_data',
    'sale_year',
]

TARGET = 'price'


def load_data():
    df = pd.read_csv(DATA_DIR + "processed_dataset.csv")
    # Encode any remaining string columns
    condition_map = {'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Excellent': 5}
    for col in df.select_dtypes(include='object').columns:
        if col in condition_map or df[col].isin(condition_map.keys()).any():
            df[col] = df[col].map(condition_map).fillna(3)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].fillna(0)
    y = np.log1p(df[TARGET])          # Log-transform for better accuracy
    print(f"  Features used   : {len(available)}")
    print(f"  Training samples: {len(X)}")
    return X, y, available


def evaluate(model, X_test, y_test, name):
    y_pred_log = model.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100

    print(f"\n  ── {name} ──")
    print(f"     R²    : {r2:.4f}  ({r2*100:.2f}%)")
    print(f"     RMSE  : {rmse:,.0f}")
    print(f"     MAE   : {mae:,.0f}")
    print(f"     MAPE  : {mape:.2f}%")
    print(f"     Accuracy: {max(0, (1 - mape/100))*100:.2f}%")

    return {"model": name, "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape,
            "Accuracy": max(0, (1 - mape/100))*100}


def train_all_models():
    print("=" * 60)
    print("STEP 3: Model Training & Evaluation")
    print("=" * 60)

    X, y, feature_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train: {X_train.shape}  |  Test: {X_test.shape}")

    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = []

    # ── 1. Linear Regression (baseline) ──────────────────────
    print("\n  Training Linear Regression...")
    lr = Ridge(alpha=10)
    lr.fit(X_train_sc, y_train)
    results.append(evaluate(lr, X_test_sc, y_test, "Linear Regression (Ridge)"))

    # ── 2. Decision Tree ─────────────────────────────────────
    print("\n  Training Decision Tree...")
    dt = DecisionTreeRegressor(max_depth=12, min_samples_split=20,
                               min_samples_leaf=10, random_state=42)
    dt.fit(X_train, y_train)
    results.append(evaluate(dt, X_test, y_test, "Decision Tree"))

    # ── 3. Random Forest ─────────────────────────────────────
    print("\n  Training Random Forest (this may take ~1 min)...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10,
                               min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate(rf, X_test, y_test, "Random Forest"))

    # ── 4. XGBoost ───────────────────────────────────────────
    print("\n  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    results.append(evaluate(xgb_model, X_test, y_test, "XGBoost"))

    # ── Pick best model ───────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by R²):")
    print("=" * 60)
    print(results_df[['model', 'R2', 'RMSE', 'MAE', 'Accuracy']].to_string(index=False))

    best_name = results_df.iloc[0]['model']
    print(f"\n  🏆 Best Model: {best_name}")
    print(f"     R²       : {results_df.iloc[0]['R2']:.4f}")
    print(f"     Accuracy : {results_df.iloc[0]['Accuracy']:.2f}%")

    # ── Save best model ───────────────────────────────────────
    model_map = {
        "Linear Regression (Ridge)": lr,
        "Decision Tree"            : dt,
        "Random Forest"            : rf,
        "XGBoost"                  : xgb_model,
    }
    best_model  = model_map[best_name]
    best_scaler = scaler if "Linear" in best_name else None

    joblib.dump(best_model,      MODELS_DIR + "best_model.pkl")
    joblib.dump(scaler,          MODELS_DIR + "scaler.pkl")
    joblib.dump(xgb_model,       MODELS_DIR + "xgboost_model.pkl")
    joblib.dump(rf,              MODELS_DIR + "random_forest_model.pkl")
    joblib.dump(feature_names,   MODELS_DIR + "feature_names.pkl")

    # Save metrics
    results_df.to_csv(MODELS_DIR + "model_metrics.csv", index=False)

    meta = {
        "best_model"      : best_name,
        "best_r2"         : float(results_df.iloc[0]['R2']),
        "best_accuracy"   : float(results_df.iloc[0]['Accuracy']),
        "best_rmse"       : float(results_df.iloc[0]['RMSE']),
        "best_mae"        : float(results_df.iloc[0]['MAE']),
        "features_used"   : feature_names,
        "n_features"      : len(feature_names),
        "train_samples"   : len(X_train),
        "test_samples"    : len(X_test),
    }
    with open(MODELS_DIR + "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n  ✅ Models saved to models/")

    # ── Feature Importance Plot ───────────────────────────────
    plot_feature_importance(xgb_model, feature_names)
    plot_predictions(xgb_model, X_test, y_test)

    return results_df


def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feat_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feat_df = feat_df.sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(10, 7))
    sns.barplot(x='importance', y='feature', data=feat_df, palette='viridis')
    plt.title("XGBoost – Top 20 Feature Importances", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(MODELS_DIR + "feature_importance.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  ✅ Feature importance plot saved.")


def plot_predictions(model, X_test, y_test):
    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect prediction')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices (XGBoost)", fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODELS_DIR + "prediction_scatter.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("  ✅ Prediction scatter plot saved.")


if __name__ == "__main__":
    train_all_models()
