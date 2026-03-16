"""
STEP 4: Streamlit Web Application
AI-Based Real Estate Valuation System
Run with: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Real Estate Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a3c5e 0%, #2d7dd2 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f7ff;
        border-left: 5px solid #2d7dd2;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .price-display {
        background: linear-gradient(135deg, #1a3c5e, #2d7dd2);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1a3c5e, #2d7dd2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
    }
    .info-box {
        background: #e8f4fd;
        border: 1px solid #b8d9f7;
        padding: 12px 15px;
        border-radius: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model & metadata ─────────────────────────────────────
@st.cache_resource
def load_model():
    models_dir = "models/"
    model     = joblib.load(models_dir + "best_model.pkl")
    scaler    = joblib.load(models_dir + "scaler.pkl")
    features  = joblib.load(models_dir + "feature_names.pkl")
    with open(models_dir + "model_meta.json") as f:
        meta = json.load(f)
    metrics_df = pd.read_csv(models_dir + "model_metrics.csv")
    return model, scaler, features, meta, metrics_df


@st.cache_data
def load_dataset():
    return pd.read_csv("data/processed_dataset.csv")


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏠 AI-Based Real Estate Valuation System</h1>
    <p style="margin:0; opacity:0.85;">Powered by XGBoost & Random Forest | Trained on 8 Datasets | India + Global Market Data</p>
</div>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────
try:
    model, scaler, feature_names, meta, metrics_df = load_model()
    df = load_dataset()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning(f"⚠️ Model not trained yet. Please run `python src/model_training.py` first.\n\n{e}")

# ── Navigation ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Price Predictor",
    "📊 Market Analytics",
    "🤖 Model Performance",
    "📁 Dataset Overview"
])


# ══════════════════════════════════════════════════════════════
# TAB 1: PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Property Details")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 🏗️ Property Basics")
        bedrooms    = st.slider("Bedrooms",           1, 10, 3)
        bathrooms   = st.slider("Bathrooms",          1.0, 8.0, 2.0, step=0.25)
        floors      = st.slider("Floors",             1.0, 4.0, 1.0, step=0.5)
        sqft_living = st.number_input("Living Area (sqft)", 300, 15000, 1800, step=100)
        sqft_lot    = st.number_input("Lot Area (sqft)",    500, 200000, 6000, step=500)
        sqft_above  = st.number_input("Area Above Basement (sqft)", 300, 15000,
                                      min(sqft_living, 1800), step=100)
        sqft_basement = max(0, sqft_living - sqft_above)
        st.info(f"🏚️ Basement Area (auto): **{sqft_basement} sqft**")

        st.markdown("#### 📅 Age & Renovation")
        yr_built      = st.slider("Year Built",      1900, 2024, 1990)
        yr_renovated  = st.slider("Year Renovated (0 = never)", 0, 2024, 0)
        sale_year     = st.slider("Sale Year",       2010, 2026, 2020)

    with col_right:
        st.markdown("#### ⭐ Quality & Features")
        grade       = st.slider("Grade (1–13)",       1, 13, 7,
                                help="Building quality score")
        condition   = st.slider("Condition (1–5)",    1, 5, 3,
                                help="1=Poor, 3=Average, 5=Excellent")
        view        = st.slider("View Score (0–4)",   0, 4, 0)
        waterfront  = st.selectbox("Waterfront Property?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        st.markdown("#### 📍 Location")
        sqft_living15 = st.number_input("Avg Neighbor Living Area (sqft)", 500, 10000, 1800, step=100)
        sqft_lot15    = st.number_input("Avg Neighbor Lot Area (sqft)",    500, 200000, 6000, step=500)
        lat           = st.number_input("Latitude",   20.0, 60.0, 47.5, format="%.4f")
        long          = st.number_input("Longitude", -130.0, -60.0, -122.2, format="%.4f")
        zipcode       = st.number_input("ZIP / Postal Code", 10000, 999999, 98178)

        st.markdown("#### 🌏 India-Specific (optional)")
        is_india       = st.checkbox("India Property?")
        schools_nearby = st.slider("Schools Nearby",       0, 10, 2) if is_india else 0
        airport_dist   = st.number_input("Distance from Airport (km)", 0, 200, 30) if is_india else 0

    # ── Predict ─────────────────────────────────────────────
    st.markdown("---")
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_btn = st.button("🔮 Predict House Price")

    if predict_btn and model_loaded:
        # Build feature vector
        house_age               = max(0, sale_year - yr_built)
        was_renovated           = 1 if yr_renovated > 0 else 0
        years_since_renovation  = (sale_year - yr_renovated) if yr_renovated > 0 else house_age
        total_area              = sqft_above + sqft_basement
        bath_bed_ratio          = bathrooms / max(bedrooms, 1)
        basement_ratio          = sqft_basement / max(total_area, 1)
        living_lot_ratio        = sqft_living / max(sqft_lot, 1)
        area_per_bedroom        = sqft_living / max(bedrooms, 1)
        area_per_bathroom       = sqft_living / max(bathrooms, 1)
        grade_condition_score   = grade * condition
        living_vs_neighbors     = sqft_living / max(sqft_living15, 1)
        lot_vs_neighbors        = sqft_lot    / max(sqft_lot15, 1)
        is_luxury               = 1 if (grade >= 10 and sqft_living >= 3000) else 0

        # Market trend defaults (using dataset medians)
        india_price_index   = 150.0 if is_india else 0
        india_trend_slope   = 0.8   if is_india else 0
        global_nominal_avg  = 0.95
        global_real_avg     = 0.85
        global_yoy_change   = 2.5

        raw = {
            'sqft_living': sqft_living, 'sqft_lot': sqft_lot,
            'sqft_above': sqft_above, 'sqft_basement': sqft_basement,
            'bedrooms': bedrooms, 'bathrooms': bathrooms, 'floors': floors,
            'grade': grade, 'condition': condition, 'view': view, 'waterfront': waterfront,
            'yr_built': yr_built, 'yr_renovated': yr_renovated,
            'sqft_living15': sqft_living15, 'sqft_lot15': sqft_lot15,
            'lat': lat, 'long': long, 'zipcode': zipcode,
            'schools_nearby': schools_nearby, 'airport_distance': airport_dist,
            'house_age': house_age, 'was_renovated': was_renovated,
            'years_since_renovation': years_since_renovation,
            'total_area': total_area, 'bath_bed_ratio': bath_bed_ratio,
            'basement_ratio': basement_ratio, 'living_lot_ratio': living_lot_ratio,
            'area_per_bedroom': area_per_bedroom, 'area_per_bathroom': area_per_bathroom,
            'grade_condition_score': grade_condition_score,
            'living_vs_neighbors': living_vs_neighbors, 'lot_vs_neighbors': lot_vs_neighbors,
            'is_luxury': is_luxury,
            'india_price_index': india_price_index, 'india_trend_slope': india_trend_slope,
            'global_nominal_avg': global_nominal_avg, 'global_real_avg': global_real_avg,
            'global_yoy_change': global_yoy_change,
            'is_india_dataset': int(is_india), 'has_school_dist_data': int(is_india),
            'sale_year': sale_year,
        }

        X_input = pd.DataFrame([raw])[feature_names].fillna(0)

        pred_log = model.predict(X_input)[0]
        pred_price = np.expm1(pred_log)

        # Confidence range (±10%)
        low  = pred_price * 0.90
        high = pred_price * 1.10

        st.markdown(f"""
        <div class="price-display">
            🏠 Estimated Price: ${pred_price:,.0f}
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Predicted Price",  f"${pred_price:,.0f}")
        c2.metric("📉 Lower Estimate",   f"${low:,.0f}")
        c3.metric("📈 Upper Estimate",   f"${high:,.0f}")

        # Price gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_price,
            title={'text': "Predicted House Price (USD)"},
            gauge={
                'axis': {'range': [0, max(df['price'].quantile(0.99), pred_price * 1.2)]},
                'bar': {'color': "#2d7dd2"},
                'steps': [
                    {'range': [0, df['price'].quantile(0.33)],  'color': "#e8f4fd"},
                    {'range': [df['price'].quantile(0.33), df['price'].quantile(0.66)], 'color': "#b8d9f7"},
                    {'range': [df['price'].quantile(0.66), df['price'].quantile(0.99)], 'color': "#6aaee0"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': df['price'].median()
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
        📌 <b>Market Context:</b> Median price in dataset = <b>${df['price'].median():,.0f}</b> |
        Your property is in the <b>{int(np.searchsorted(np.sort(df['price']), pred_price) / len(df) * 100)}th percentile</b>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2: MARKET ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Market Analytics Dashboard")

    if 'df' in dir() and df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Properties",  f"{len(df):,}")
        c2.metric("Avg Price",         f"${df['price'].mean():,.0f}")
        c3.metric("Median Price",      f"${df['price'].median():,.0f}")
        c4.metric("Max Price",         f"${df['price'].max():,.0f}")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.histogram(df, x='price', nbins=80,
                                title="Price Distribution",
                                color_discrete_sequence=['#2d7dd2'])
            fig1.update_layout(bargap=0.05)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            grade_stats = df.groupby('grade')['price'].median().reset_index()
            fig2 = px.bar(grade_stats, x='grade', y='price',
                          title="Median Price by Grade",
                          color='price', color_continuous_scale='Blues')
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig3 = px.scatter(df.sample(min(3000, len(df))),
                              x='sqft_living', y='price',
                              color='grade', size='bedrooms',
                              title="Price vs Living Area (colored by Grade)",
                              opacity=0.6, color_continuous_scale='Viridis')
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            yr_stats = df.groupby('sale_year')['price'].median().reset_index()
            fig4 = px.line(yr_stats, x='sale_year', y='price',
                           title="Median Price by Sale Year",
                           markers=True, line_shape='spline',
                           color_discrete_sequence=['#1a3c5e'])
            st.plotly_chart(fig4, use_container_width=True)

        # Correlation heatmap
        num_cols = ['price', 'sqft_living', 'grade', 'bathrooms', 'bedrooms',
                    'sqft_above', 'total_area', 'house_age', 'condition', 'view']
        available_num = [c for c in num_cols if c in df.columns]
        corr = df[available_num].corr()
        fig5 = px.imshow(corr, text_auto=".2f",
                         title="Feature Correlation Matrix",
                         color_continuous_scale='RdBu_r', aspect='auto')
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Model Performance Comparison")

    if model_loaded:
        # Best model KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Best Model",   meta['best_model'])
        c2.metric("📈 R² Score",     f"{meta['best_r2']:.4f}")
        c3.metric("✅ Accuracy",     f"{meta['best_accuracy']:.2f}%")
        c4.metric("📉 RMSE",         f"${meta['best_rmse']:,.0f}")

        # Model comparison table
        st.markdown("#### All Models Comparison")
        display_df = metrics_df[['model', 'R2', 'RMSE', 'MAE', 'Accuracy']].copy()
        display_df['R2']       = display_df['R2'].map("{:.4f}".format)
        display_df['RMSE']     = display_df['RMSE'].map("${:,.0f}".format)
        display_df['MAE']      = display_df['MAE'].map("${:,.0f}".format)
        display_df['Accuracy'] = display_df['Accuracy'].map("{:.2f}%".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Model bar chart
        fig_bar = px.bar(metrics_df.sort_values('R2', ascending=False),
                         x='model', y='R2',
                         title="Model R² Score Comparison",
                         color='R2', color_continuous_scale='Blues',
                         text=metrics_df.sort_values('R2', ascending=False)['R2'].map('{:.4f}'.format))
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)

        # Feature importance image
        if os.path.exists("models/feature_importance.png"):
            st.markdown("#### XGBoost Feature Importance")
            st.image("models/feature_importance.png", use_container_width=True)

        if os.path.exists("models/prediction_scatter.png"):
            st.markdown("#### Actual vs Predicted Prices")
            st.image("models/prediction_scatter.png", use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
        📐 <b>Features used:</b> {meta['n_features']} |
        🧪 <b>Training samples:</b> {meta['train_samples']:,} |
        🔬 <b>Test samples:</b> {meta['test_samples']:,}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📁 Dataset Overview")

    dataset_info = pd.DataFrame({
        "Dataset"     : ["House_Price_India.csv", "house_prices.csv", "Housing.csv",
                         "housing_price_index.csv", "nominal_index.csv",
                         "nominal_year.csv", "real_index.csv", "real_year.csv"],
        "Rows"        : [14620, 21613, 21613, 7, 23994, 23994, 23994, 23994],
        "Columns"     : [23, 21, 21, 11, 4, 4, 4, 4],
        "Type"        : ["Primary – India Properties", "Primary – King County",
                         "Primary – King County Extended", "India City Index",
                         "Global Nominal Index (Quarterly)", "Global Nominal Index (Yearly)",
                         "Global Real Index (Quarterly)", "Global Real Index (Yearly)"],
        "Role"        : ["Main training data", "Main training data", "Supplemental training",
                         "Market multiplier feature", "Trend feature", "Trend feature",
                         "Trend feature", "Trend feature"],
    })
    st.dataframe(dataset_info, use_container_width=True, hide_index=True)

    st.markdown("#### 🔍 Sample Data (Processed)")
    if 'df' in dir():
        st.dataframe(df.head(50), use_container_width=True)

    st.markdown("""
    #### 📐 Feature Engineering Summary
    | Feature | Description |
    |---|---|
    | `house_age` | Years since built at time of sale |
    | `was_renovated` | Binary flag: renovated or not |
    | `years_since_renovation` | Age since last renovation |
    | `total_area` | sqft_above + sqft_basement |
    | `bath_bed_ratio` | Bathrooms ÷ Bedrooms |
    | `basement_ratio` | Basement ÷ Total area |
    | `living_lot_ratio` | Living area ÷ Lot area |
    | `area_per_bedroom` | Sqft living ÷ Bedrooms |
    | `grade_condition_score` | Grade × Condition |
    | `living_vs_neighbors` | Your area ÷ Neighbor avg |
    | `is_luxury` | Grade ≥ 10 AND sqft_living ≥ 3000 |
    | `india_price_index` | All-India city avg price index |
    | `global_nominal_avg` | Global avg nominal house price index |
    | `global_real_avg` | Global avg real house price index |
    | `global_yoy_change` | Year-over-year % change in global index |
    """)


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:0.85rem;">
    🏠 AI-Based Real Estate Valuation System &nbsp;|&nbsp;
    Built with Streamlit, XGBoost, Random Forest &nbsp;|&nbsp;
    8 Datasets | India + Global Market Intelligence
</div>
""", unsafe_allow_html=True)
