"""
STEP 2: Exploratory Data Analysis (EDA)
AI-Based Real Estate Valuation System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATA_DIR  = "data/"
PLOTS_DIR = "models/"


def run_eda():
    print("=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)

    df = pd.read_csv(DATA_DIR + "processed_dataset.csv")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nBasic statistics:\n{df['price'].describe()}")

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Real Estate EDA – Price Distribution & Key Features", fontsize=16, fontweight='bold')

    # 1. Price distribution
    sns.histplot(df['price'], bins=60, kde=True, ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title("Price Distribution")
    axes[0, 0].set_xlabel("Price")

    # 2. Log-price distribution
    sns.histplot(np.log1p(df['price']), bins=60, kde=True, ax=axes[0, 1], color='darkorange')
    axes[0, 1].set_title("Log(Price) Distribution")
    axes[0, 1].set_xlabel("log(Price)")

    # 3. Price vs sqft_living
    axes[0, 2].scatter(df['sqft_living'], df['price'], alpha=0.2, s=5, color='teal')
    axes[0, 2].set_title("Price vs Living Area")
    axes[0, 2].set_xlabel("sqft_living")
    axes[0, 2].set_ylabel("Price")

    # 4. Price vs grade
    sns.boxplot(x='grade', y='price', data=df, ax=axes[1, 0], palette='coolwarm')
    axes[1, 0].set_title("Price by Grade")

    # 5. Correlation heatmap (top 12 features)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()['price'].abs().sort_values(ascending=False)
    top_feats = corr.index[1:13].tolist()
    corr_matrix = df[top_feats + ['price']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdYlGn',
                ax=axes[1, 1], linewidths=0.5, annot_kws={"size": 7})
    axes[1, 1].set_title("Top Feature Correlations")

    # 6. Price by bedrooms
    bed_price = df.groupby('bedrooms')['price'].median().reset_index()
    bed_price = bed_price[bed_price['bedrooms'] <= 8]
    axes[1, 2].bar(bed_price['bedrooms'], bed_price['price'], color='mediumpurple')
    axes[1, 2].set_title("Median Price by Bedrooms")
    axes[1, 2].set_xlabel("Bedrooms")
    axes[1, 2].set_ylabel("Median Price")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR + "eda_plots.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  ✅ EDA plots saved: models/eda_plots.png")

    # Print top 15 correlations
    print("\nTop 15 features correlated with Price:")
    print(corr.head(16).to_string())


if __name__ == "__main__":
    run_eda()
