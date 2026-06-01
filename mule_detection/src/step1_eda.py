import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Configuration
DATA_PATH = "mule_detection/data/raw/DataSet.csv"
PLOTS_DIR = "mule_detection/reports/plots/"
HINT_FEATURES = [
    'F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2122', 'F2582', 'F2678', 
    'F2737', 'F2956', 'F3043', 'F3836', 'F3887', 'F3889', 'F3891', 'F3894'
]
TARGET = 'F3924'

def perform_eda():
    print("Starting Step 1: Exploratory Data Analysis (EDA)")
    
    # 1. LOAD DATA
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Dataset not found at {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    
    # Handle non-numeric hint features
    if 'F3889' in df.columns:
        duration_map = {
            'L7D': 7, 'L14D': 14, 'L31D': 31, 'L90D': 90, 
            'L180D': 180, 'L365D': 365, 'G365D': 366
        }
        df['F3889_encoded'] = df['F3889'].map(duration_map)
        # For statistical analysis, use the encoded version
        HINT_FEATURES_NUM = [f if f != 'F3889' else 'F3889_encoded' for f in HINT_FEATURES]
    else:
        HINT_FEATURES_NUM = HINT_FEATURES

    if 'F3891' in df.columns:
        df['F3891_encoded'], _ = pd.factorize(df['F3891'])
        HINT_FEATURES_NUM = [f if f != 'F3891' else 'F3891_encoded' for f in HINT_FEATURES_NUM]

    print(f"\n--- Data Overview ---")
    print(f"Shape: {df.shape}")
    print(f"Dtypes Summary:\n{df.dtypes.value_counts()}")
    print("\nFirst 5 rows:")
    print(df.head())

    # 2. BASIC CHECKS
    print(f"\n--- Basic Checks ---")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print("Dropped duplicate rows.")
    
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 3. TARGET ANALYSIS
    print(f"\n--- Target Analysis ({TARGET}) ---")
    target_counts = df[TARGET].value_counts()
    target_pct = df[TARGET].value_counts(normalize=True) * 100
    print(f"Class distribution:\n{target_counts}")
    print(f"Percentage:\n{target_pct}")

    plt.figure(figsize=(8, 6))
    sns.countplot(x=TARGET, data=df, hue=TARGET, palette='viridis', legend=False)
    plt.title('Distribution of Suspicious vs Legitimate Accounts')
    plt.xlabel('Account Type (0=Legit, 1=Mule)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'))
    plt.close()

    # 4. MISSING VALUE ANALYSIS
    print(f"\n--- Missing Value Analysis ---")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    cols_with_nulls = null_counts[null_counts > 0].sort_values(ascending=False)
    print(f"Columns with missing values: {len(cols_with_nulls)}")
    if len(cols_with_nulls) > 0:
        print(cols_with_nulls.head(10))

    plt.figure(figsize=(12, 8))
    sns.heatmap(df[HINT_FEATURES].isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap (Hint Features)')
    plt.savefig(os.path.join(PLOTS_DIR, 'missing_values_heatmap.png'))
    plt.close()

    # 5. STATISTICAL ANALYSIS OF HINT FEATURES
    print(f"\n--- Statistical Analysis of Hint Features ---")
    hint_df = df[HINT_FEATURES_NUM + [TARGET]]
    stats = hint_df[HINT_FEATURES_NUM].agg(['mean', 'median', 'std', 'skew', 'kurtosis']).T
    print(stats)

    # Distribution plots for hint features
    fig, axes = plt.subplots(6, 3, figsize=(18, 24))
    axes = axes.flatten()
    for i, col in enumerate(HINT_FEATURES):
        if df[col].dtype == 'object' or df[col].nunique() < 15:
            sns.countplot(data=df, x=col, hue=TARGET, ax=axes[i], palette='Set1')
            axes[i].tick_params(axis='x', rotation=45)
        else:
            sns.histplot(data=df, x=col, hue=TARGET, kde=False, ax=axes[i], palette='Set1', bins=30, element="step")
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'hint_feature_distributions.png'))
    plt.close()

    # 6. CORRELATION ANALYSIS
    print(f"\n--- Correlation Analysis ---")
    corr_matrix = df[HINT_FEATURES_NUM + [TARGET]].corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap (Hint Features & Target)')
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()

    top_corr = corr_matrix[TARGET].abs().sort_values(ascending=False)
    print(f"Top 5 features most correlated with {TARGET}:")
    print(top_corr.iloc[1:6])

    # 7. OUTLIER DETECTION
    print(f"\n--- Outlier Detection ---")
    fig, axes = plt.subplots(6, 3, figsize=(18, 24))
    axes = axes.flatten()
    for i, col in enumerate(HINT_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'boxplots.png'))
    plt.close()

    # 8. PRINT EDA SUMMARY REPORT
    print(f"\n" + "="*40)
    print("EDA SUMMARY REPORT")
    print("="*40)
    print(f"Dataset Size: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Class Imbalance Ratio: 1:{target_counts[0]/target_counts[1]:.2f} (Legit:Mule)")
    print(f"Top 3 Correlated Features: {', '.join(top_corr.index[1:4])}")
    
    high_null_cols = null_pct[null_pct > 30].index.tolist()
    print(f"Features with >30% nulls: {len(high_null_cols)}")
    
    print("\nKey Observations for Modeling:")
    print("1. Handle class imbalance (SMOTE/Weights)")
    print("2. Scale features due to varying ranges")
    print("3. Log transform highly skewed features")
    print("4. Treat outliers in hint features")
    print("="*40)

    print("\nSTEP 1 COMPLETE - EDA finished and all plots saved.")
    print("Type 'continue to next step' to begin Feature Engineering.")

if __name__ == "__main__":
    perform_eda()
