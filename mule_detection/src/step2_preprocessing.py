import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from scipy.stats import skew
import json

# Configuration
DATA_PATH = "mule_detection/data/raw/DataSet.csv"
PROCESSED_DIR = "mule_detection/data/processed/"
REPORTS_DIR = "mule_detection/reports/"
HINT_FEATURES = [
    'F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2122', 'F2582', 'F2678', 
    'F2737', 'F2956', 'F3043', 'F3836', 'F3887', 'F3889', 'F3891', 'F3894'
]
TARGET = 'F3924'

def preprocess_and_engineer():
    print("Starting Step 2: Data Preprocessing & Feature Engineering")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Pre-handle categorical hint features for feature engineering
    if 'F3889' in df.columns:
        duration_map = {
            'L7D': 7, 'L14D': 14, 'L31D': 31, 'L90D': 90, 
            'L180D': 180, 'L365D': 365, 'G365D': 366
        }
        df['F3889_num'] = df['F3889'].map(duration_map)
    
    if 'F3891' in df.columns:
        df['F3891_num'], _ = pd.factorize(df['F3891'])
    
    # Update HINT_FEATURES list for numeric operations
    HINT_FEATURES_NUM = [f if f not in ['F3889', 'F3891'] else f + '_num' for f in HINT_FEATURES]

    # 1. MISSING VALUE TREATMENT
    print("\n1. Treating Missing Values...")
    # Drop columns with > 50% missing values
    missing_threshold = 0.5
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > missing_threshold].index.tolist()
    # Ensure target and hint features are not dropped
    cols_to_drop = [c for c in cols_to_drop if c not in HINT_FEATURES and c not in HINT_FEATURES_NUM and c != TARGET]
    
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns with >50% missing values.")
    
    # Impute missing values (median for numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Handle any remaining (categorical) missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    # 2. OUTLIER TREATMENT (Winsorization on hint features)
    print("\n2. Treating Outliers...")
    for col in HINT_FEATURES_NUM:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    # 3. FEATURE ENGINEERING
    print("\n3. Engineering Features...")
    new_features = {}
    
    # a) Ratio features
    new_features['debit_credit_ratio'] = df['F527'] / (df['F531'] + 1e-5)
    new_features['high_value_ratio'] = df['F2082'] / (df['F2122'] + 1e-5)
    new_features['freq_amount_ratio'] = df['F670'] / (df['F1692'] + 1e-5)
    new_features['transfer_ratio'] = df['F3887'] / (df['F3889_num'] + 1e-5)

    # b) Aggregation features
    new_features['hint_sum'] = df[HINT_FEATURES_NUM].sum(axis=1)
    new_features['hint_mean'] = df[HINT_FEATURES_NUM].mean(axis=1)
    new_features['hint_std'] = df[HINT_FEATURES_NUM].std(axis=1)
    new_features['hint_max'] = df[HINT_FEATURES_NUM].max(axis=1)
    new_features['hint_min'] = df[HINT_FEATURES_NUM].min(axis=1)
    new_features['hint_range'] = new_features['hint_max'] - new_features['hint_min']
    new_features['hint_skew'] = df[HINT_FEATURES_NUM].apply(lambda x: skew(x), axis=1)

    # c) Interaction features
    new_features['F115_x_F321'] = df['F115'] * df['F321']
    new_features['F527_x_F670'] = df['F527'] * df['F670']
    new_features['F2082_x_F2678'] = df['F2082'] * df['F2678']
    new_features['F3836_x_F3887'] = df['F3836'] * df['F3887']

    # d) Log transforms
    for col in HINT_FEATURES_NUM:
        if df[col].skew() > 1:
            new_features[f'{col}_log'] = np.log1p(df[col])

    # e) Percentile rank features
    for col in HINT_FEATURES_NUM:
        new_features[f'{col}_pct_rank'] = df[col].rank(pct=True)

    # Combine new features
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)

    # f) Unsupervised Anomaly Score
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['isolation_anomaly_score'] = iso.fit_predict(df[HINT_FEATURES_NUM])
    df['is_anomaly'] = df['isolation_anomaly_score'].map({1: 0, -1: 1})

    # Clean feature names for LightGBM
    import re
    df.columns = [re.sub(r'[^\w\s]', '_', col) for col in df.columns]
    # Update HINT_FEATURES_NUM and TARGET after cleaning
    HINT_FEATURES_NUM = [re.sub(r'[^\w\s]', '_', col) for col in HINT_FEATURES_NUM]
    TARGET_CLEAN = re.sub(r'[^\w\s]', '_', TARGET)

    # 4. FEATURE SELECTION
    print("\n4. Selecting Features...")
    # Prepare X and y
    X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_CLEAN])
    y = df[TARGET_CLEAN]

    # Step A: Remove near-zero variance features
    selector = VarianceThreshold(threshold=0.01)
    X_var = X.loc[:, selector.fit(X).get_support()]
    print(f"Features after variance threshold: {X_var.shape[1]}")

    # Step B: Remove highly correlated features
    corr_matrix = X_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    # Keep hint features and engineered features if possible
    engineered_cols = ['debit_credit_ratio', 'high_value_ratio', 'freq_amount_ratio', 'transfer_ratio', 
                      'hint_sum', 'hint_mean', 'hint_std', 'hint_max', 'hint_min', 'hint_range', 'hint_skew',
                      'isolation_anomaly_score', 'is_anomaly']
    to_drop = [c for c in to_drop if c not in HINT_FEATURES_NUM and c not in engineered_cols]
    X_corr = X_var.drop(columns=to_drop)
    print(f"Features after correlation filter: {X_corr.shape[1]}")

    # Step C: LightGBM quick fit
    dtrain = lgb.Dataset(X_corr, label=y)
    params = {'objective': 'binary', 'verbose': -1, 'seed': 42}
    model = lgb.train(params, dtrain, num_boost_round=100)
    importances = pd.Series(model.feature_importance(), index=X_corr.columns).sort_values(ascending=False)
    top_100_cols = importances.head(100).index.tolist()

    # Final feature set
    engineered_features = [c for c in df.columns if any(suffix in c for suffix in ['_ratio', 'hint_', '_x_', '_log', '_pct_rank', 'isolation_', 'is_anomaly'])]
    final_features = list(set(HINT_FEATURES_NUM + engineered_features + top_100_cols))
    
    # Filter final_features to ensure they exist in df
    final_features = [f for f in final_features if f in df.columns]
    
    with open(os.path.join(REPORTS_DIR, 'selected_features.txt'), 'w') as f:
        f.write("\n".join(final_features))
    print(f"Final selected features: {len(final_features)}")

    # 5. CLASS IMBALANCE & TRAIN/TEST SPLIT
    print("\n5. Splitting and Handling Imbalance...")
    X_final = df[final_features]
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"Original fraud ratio: {y.mean():.4f}")
    
    # Apply SMOTE on training set
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"SMOTE fraud ratio: {y_train_smote.mean():.4f}")

    # 6. SAVE PROCESSED DATA
    print("\n6. Saving Processed Data...")
    X_train_smote.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train_smote.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    print("\nSTEP 2 COMPLETE - Preprocessing and feature engineering done.")
    print("Type 'continue to next step' to begin Model Building.")

if __name__ == "__main__":
    preprocess_and_engineer()
