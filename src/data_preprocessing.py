"""
STEP 1: Data Collection, Preprocessing & Feature Engineering
AI-Based Real Estate Valuation System
Uses all 8 datasets to build a rich, merged feature set.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data/"

# ─────────────────────────────────────────────
# 1. LOAD ALL 8 DATASETS
# ─────────────────────────────────────────────

def load_all_datasets():
    print("=" * 60)
    print("STEP 1: Loading all 8 datasets...")
    print("=" * 60)

    # --- Primary property datasets ---
    india = pd.read_csv(DATA_DIR + "House_Price_India.csv")
    house_prices = pd.read_csv(DATA_DIR + "house_prices.csv")
    housing = pd.read_csv(DATA_DIR + "Housing.csv")

    # --- India price index (city-level quarterly index) ---
    price_index = pd.read_csv(DATA_DIR + "housing_price_index.csv")

    # --- Global price indices (nominal & real) ---
    nominal_index = pd.read_csv(DATA_DIR + "nominal_index.csv", encoding='latin-1')
    nominal_year  = pd.read_csv(DATA_DIR + "nominal_year.csv",  encoding='latin-1')
    real_index    = pd.read_csv(DATA_DIR + "real_index.csv",    encoding='latin-1')
    real_year     = pd.read_csv(DATA_DIR + "real_year.csv",     encoding='latin-1')

    print(f"  House_Price_India  : {india.shape}")
    print(f"  house_prices       : {house_prices.shape}")
    print(f"  Housing            : {housing.shape}")
    print(f"  housing_price_index: {price_index.shape}")
    print(f"  nominal_index      : {nominal_index.shape}")
    print(f"  nominal_year       : {nominal_year.shape}")
    print(f"  real_index         : {real_index.shape}")
    print(f"  real_year          : {real_year.shape}")

    return india, house_prices, housing, price_index, nominal_index, nominal_year, real_index, real_year


# ─────────────────────────────────────────────
# 2. BUILD GLOBAL MARKET TREND FEATURES
#    from nominal_index, nominal_year, real_index, real_year
# ─────────────────────────────────────────────

def build_global_trends(nominal_index, nominal_year, real_index, real_year):
    """
    Derive market-level trend features:
      - global_nominal_avg : average nominal price index across countries (quarterly)
      - global_real_avg    : average real price index across countries (quarterly)
      - global_yoy_change  : year-over-year % change in nominal index
    These are merged onto the main dataset by year.
    """
    for df in [nominal_index, nominal_year, real_index, real_year]:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year

    # Aggregate global averages per year
    nom_yr = nominal_year.dropna(subset=['price']).groupby('year')['price'].mean().reset_index()
    nom_yr.columns = ['year', 'global_nominal_avg']

    real_yr = real_year.dropna(subset=['price']).groupby('year')['price'].mean().reset_index()
    real_yr.columns = ['year', 'global_real_avg']

    # Year-over-year % change
    nom_yr['global_yoy_change'] = nom_yr['global_nominal_avg'].pct_change() * 100

    global_trends = nom_yr.merge(real_yr, on='year', how='outer').sort_values('year')
    global_trends['global_yoy_change'] = global_trends['global_yoy_change'].fillna(0)

    print(f"\n  Global trend features built for {len(global_trends)} years.")
    return global_trends


# ─────────────────────────────────────────────
# 3. BUILD INDIA CITY INDEX FEATURE
#    from housing_price_index (All India row)
# ─────────────────────────────────────────────

def get_india_market_multiplier(price_index):
    """
    Extract the All-India average price index as a single market multiplier.
    Used to scale/adjust predictions based on market conditions.
    """
    all_india = price_index[price_index['Particulars'] == 'All India']
    cols = [c for c in all_india.columns if c != 'Particulars']
    vals = all_india[cols].values.flatten().astype(float)
    india_avg_index = float(np.mean(vals))
    india_trend_slope = float(np.polyfit(range(len(vals)), vals, 1)[0])  # quarterly growth rate
    print(f"  India avg price index : {india_avg_index:.2f}")
    print(f"  India trend slope     : {india_trend_slope:.4f} per quarter")
    return india_avg_index, india_trend_slope


# ─────────────────────────────────────────────
# 4. PREPROCESS & MERGE PRIMARY DATASETS
# ─────────────────────────────────────────────

def preprocess_india(df, india_avg_index, india_trend_slope):
    df = df.copy()

    # Rename columns to standardized names
    df.rename(columns={
        'number of bedrooms'               : 'bedrooms',
        'number of bathrooms'              : 'bathrooms',
        'living area'                      : 'sqft_living',
        'lot area'                         : 'sqft_lot',
        'number of floors'                 : 'floors',
        'waterfront present'               : 'waterfront',
        'number of views'                  : 'view',
        'condition of the house'           : 'condition',
        'grade of the house'               : 'grade',
        'Area of the house(excluding basement)': 'sqft_above',
        'Area of the basement'             : 'sqft_basement',
        'Built Year'                       : 'yr_built',
        'Renovation Year'                  : 'yr_renovated',
        'Postal Code'                      : 'zipcode',
        'Lattitude'                        : 'lat',
        'Longitude'                        : 'long',
        'living_area_renov'                : 'sqft_living15',
        'lot_area_renov'                   : 'sqft_lot15',
        'Number of schools nearby'         : 'schools_nearby',
        'Distance from the airport'        : 'airport_distance',
        'Price'                            : 'price',
    }, inplace=True)

    # Parse year from Date column
    df['sale_year'] = pd.to_numeric(df['Date'], errors='coerce')
    df['sale_year'] = (df['sale_year'] / 365 + 1900).astype(int)
    df.drop(columns=['id', 'Date'], inplace=True, errors='ignore')

    # Inject India market features
    df['india_price_index']    = india_avg_index
    df['india_trend_slope']    = india_trend_slope
    df['is_india_dataset']     = 1

    # India-specific features present
    df['has_school_dist_data'] = 1

    return df


def preprocess_housing(df, global_trends):
    df = df.copy()
    df.rename(columns={'sqft_living15': 'sqft_living15', 'sqft_lot15': 'sqft_lot15'}, inplace=True)

    # Parse year from date string
    df['sale_year'] = pd.to_datetime(df['date'].astype(str).str[:8], format='%Y%m%d', errors='coerce').dt.year
    df.drop(columns=['id', 'date'], inplace=True, errors='ignore')

    # No India-specific columns
    df['schools_nearby']    = 0
    df['airport_distance']  = 0
    df['india_price_index'] = 0
    df['india_trend_slope'] = 0
    df['is_india_dataset']  = 0
    df['has_school_dist_data'] = 0

    # Merge global trends
    df = df.merge(global_trends[['year', 'global_nominal_avg', 'global_real_avg', 'global_yoy_change']],
                  left_on='sale_year', right_on='year', how='left')
    df.drop(columns=['year'], inplace=True, errors='ignore')

    # Convert waterfront: 'N'/'Y' -> 0/1 if string
    if df['waterfront'].dtype == object:
        df['waterfront'] = df['waterfront'].map({'N': 0, 'Y': 1, '0': 0, '1': 1}).fillna(0).astype(int)

    # Convert condition: 'Average'/etc -> numeric if string
    condition_map = {'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Excellent': 5}
    if df['condition'].dtype == object:
        df['condition'] = df['condition'].map(condition_map).fillna(3).astype(int)

    return df


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df):
    df = df.copy()

    # House age at time of sale
    df['house_age'] = df['sale_year'] - df['yr_built']
    df['house_age'] = df['house_age'].clip(lower=0)

    # Was it renovated?
    df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
    df['years_since_renovation'] = np.where(
        df['yr_renovated'] > 0,
        df['sale_year'] - df['yr_renovated'],
        df['house_age']
    )

    # Total area
    df['total_area'] = df['sqft_above'] + df['sqft_basement']

    # Price-driving ratio features
    df['bath_bed_ratio']    = df['bathrooms'] / (df['bedrooms'].replace(0, 1))
    df['basement_ratio']    = df['sqft_basement'] / (df['total_area'].replace(0, 1))
    df['living_lot_ratio']  = df['sqft_living']   / (df['sqft_lot'].replace(0, 1))
    df['area_per_bedroom']  = df['sqft_living']   / (df['bedrooms'].replace(0, 1))
    df['area_per_bathroom'] = df['sqft_living']   / (df['bathrooms'].replace(0, 1))

    # Renovation quality boost
    df['grade_condition_score'] = df['grade'] * df['condition']

    # Neighbour area comparison
    df['living_vs_neighbors'] = df['sqft_living'] / (df['sqft_living15'].replace(0, 1))
    df['lot_vs_neighbors']    = df['sqft_lot']    / (df['sqft_lot15'].replace(0, 1))

    # Luxury indicator
    df['is_luxury'] = ((df['grade'] >= 10) & (df['sqft_living'] >= 3000)).astype(int)

    # Fill any remaining NaNs
    df.fillna(0, inplace=True)

    return df


# ─────────────────────────────────────────────
# 6. MERGE & FINALIZE
# ─────────────────────────────────────────────

def build_final_dataset():
    print("\n" + "=" * 60)
    print("Building Final Merged Dataset")
    print("=" * 60)

    (india, house_prices, housing,
     price_index, nominal_index, nominal_year,
     real_index, real_year) = load_all_datasets()

    # Build auxiliary features from macro datasets
    global_trends = build_global_trends(nominal_index, nominal_year, real_index, real_year)
    india_avg_index, india_trend_slope = get_india_market_multiplier(price_index)

    # Preprocess each primary dataset
    print("\n  Preprocessing House_Price_India...")
    df_india = preprocess_india(india, india_avg_index, india_trend_slope)
    df_india = df_india.merge(
        global_trends[['year', 'global_nominal_avg', 'global_real_avg', 'global_yoy_change']],
        left_on='sale_year', right_on='year', how='left'
    ).drop(columns=['year'], errors='ignore')
    df_india.fillna(0, inplace=True)

    print("  Preprocessing house_prices (King County)...")
    df_houses = preprocess_housing(house_prices, global_trends)

    print("  Preprocessing Housing (King County extended)...")
    df_housing = preprocess_housing(housing, global_trends)

    # Combine all three datasets
    all_cols = sorted(set(df_india.columns) | set(df_houses.columns) | set(df_housing.columns))

    for df in [df_india, df_houses, df_housing]:
        for col in all_cols:
            if col not in df.columns:
                df[col] = 0

    combined = pd.concat([df_india, df_houses, df_housing], ignore_index=True)

    # Remove duplicates (house_prices & Housing overlap)
    combined.drop_duplicates(subset=['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'price'],
                             inplace=True)

    # Feature Engineering
    print(f"\n  Combined shape before FE: {combined.shape}")
    combined = feature_engineering(combined)
    print(f"  Combined shape after  FE: {combined.shape}")

    # Remove obvious outliers (price)
    q_low  = combined['price'].quantile(0.005)
    q_high = combined['price'].quantile(0.995)
    combined = combined[(combined['price'] >= q_low) & (combined['price'] <= q_high)]
    print(f"  After outlier removal    : {combined.shape}")

    # Save processed dataset
    combined.to_csv(DATA_DIR + "processed_dataset.csv", index=False)
    print("\n  ✅ Saved: data/processed_dataset.csv")
    print(f"  Total samples  : {len(combined)}")
    print(f"  Total features : {combined.shape[1] - 1}")

    return combined


if __name__ == "__main__":
    df = build_final_dataset()
    print("\nFeature list:")
    for c in sorted(df.columns):
        print(f"  {c}")
