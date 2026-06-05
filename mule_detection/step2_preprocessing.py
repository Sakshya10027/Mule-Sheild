"""
╔══════════════════════════════════════════════════════════════════╗
║   STEP 2 — DATA PREPROCESSING & FEATURE ENGINEERING            ║
║   MuleShield AI | Bank of India Cybersecurity Challenge          ║
╚══════════════════════════════════════════════════════════════════╝

Key dataset facts from EDA:
  - 9,082 rows × 3,925 columns
  - Imbalance: 111:1 (99.11% legit, 0.89% suspicious)
  - F3889 ('G365D' codes) & F3891 (occupation) are categorical
  - Several hint features have extreme skewness
  - F3836 has large negative values

Run:
    python mule_detection/step2_preprocessing.py
"""

import os, sys, glob, warnings, json
import numpy as np
import pandas as pd
from scipy import stats, sparse
from scipy.stats import skew

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold

import lightgbm as lgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
PARENT_DIR   = os.path.dirname(PROJECT_ROOT)

DATA_RAW_DIR  = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(DATA_PROC_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR,   exist_ok=True)

TARGET_COL = "F3924"
RANDOM_STATE = 42

HINT_FEATURES = [
    "F115", "F321", "F527", "F531", "F670",
    "F1692", "F2082", "F2122", "F2582", "F2678",
    "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894"
]

# Categorical hint features (identified in EDA)
CAT_HINT_FEATURES = ["F3889", "F3891"]
NUM_HINT_FEATURES = [f for f in HINT_FEATURES if f not in CAT_HINT_FEATURES]

def section(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")

def find_csv():
    for d in [DATA_RAW_DIR, PROJECT_ROOT, PARENT_DIR]:
        hits = glob.glob(os.path.join(d, "*.csv"))
        if hits:
            return hits[0]
    raise FileNotFoundError("No CSV found.")

# ─────────────────────────────────────────────────────────────────
# LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────

section("LOAD RAW DATA")

csv_path = find_csv()
print(f"  Loading: {csv_path}")
df = pd.read_csv(csv_path, low_memory=False)
print(f"  Shape: {df.shape}")

# Convert all but categorical hint features to numeric
cat_cols_to_keep = [c for c in CAT_HINT_FEATURES if c in df.columns]
other_cols = [c for c in df.columns if c not in cat_cols_to_keep]
for col in other_cols:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop target NaNs
df.dropna(subset=[TARGET_COL], inplace=True)
df[TARGET_COL] = df[TARGET_COL].astype(int)
df.reset_index(drop=True, inplace=True)
print(f"  Shape after target NaN drop: {df.shape}")
print(f"  Class distribution: {df[TARGET_COL].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────
# STEP 2-1: MISSING VALUE TREATMENT
# ─────────────────────────────────────────────────────────────────

section("2-1  MISSING VALUE TREATMENT")

# A. Drop columns with >50% missing
null_pct = df.isnull().sum() / len(df)
cols_to_drop = null_pct[null_pct > 0.50].index.tolist()

# Never drop TARGET or HINT features (even high-null ones get imputed)
# But F3889 & F3891 which are 100% NaN (categorical forced to NaN) will be dropped
protected = [TARGET_COL] + [f for f in NUM_HINT_FEATURES if f in df.columns]
cols_to_drop_final = [c for c in cols_to_drop if c not in protected]

print(f"  Columns with >50% missing: {len(cols_to_drop)}")
print(f"  Protected hint columns  : {[c for c in cols_to_drop if c in protected]}")
print(f"  Columns to drop         : {len(cols_to_drop_final)}")

# Save dropped column list
dropped_log_path = os.path.join(REPORTS_DIR, "dropped_columns.txt")
with open(dropped_log_path, "w") as f:
    f.write(f"Dropped {len(cols_to_drop_final)} columns with >50% missing:\n")
    for c in cols_to_drop_final:
        f.write(f"  {c}: {null_pct[c]*100:.1f}%\n")
print(f"  Saved dropped column log: {dropped_log_path}")

df.drop(columns=cols_to_drop_final, inplace=True)
print(f"  Shape after dropping high-null cols: {df.shape}")

# B. Median imputation for ALL remaining numeric columns (incl. hint features)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

print(f"\n  Applying median imputation to {len(numeric_cols)} numeric columns …")
medians = df[numeric_cols].median()
df[numeric_cols] = df[numeric_cols].fillna(medians)

# C. Handle categorical hint features — label encode if present, else create dummy 0
for cat_feat in CAT_HINT_FEATURES:
    if cat_feat in df.columns and df[cat_feat].dtype == 'object':
        # Check uniqueness
        uniques = df[cat_feat].dropna().unique()
        if len(uniques) <= 1:
            # Constant or useless — drop
            print(f"  {cat_feat}: constant/useless (unique values: {uniques}) → dropped")
            df.drop(columns=[cat_feat], inplace=True)
        else:
            # Label encode
            le = LabelEncoder()
            df[cat_feat] = df[cat_feat].fillna("Unknown")
            df[cat_feat] = le.fit_transform(df[cat_feat].astype(str))
            print(f"  {cat_feat}: label-encoded → {len(uniques)} unique values")
    elif cat_feat not in df.columns:
        print(f"  {cat_feat}: not in dataframe (was dropped due to >50% nulls) → adding zero column")
        df[cat_feat] = 0

# Confirm no nulls remain in numeric cols
remaining_nulls = df[numeric_cols].isnull().sum().sum()
print(f"\n  Remaining nulls in numeric columns: {remaining_nulls}")

# ─────────────────────────────────────────────────────────────────
# STEP 2-2: OUTLIER TREATMENT (Winsorization on hint features)
# ─────────────────────────────────────────────────────────────────

section("2-2  OUTLIER TREATMENT — IQR WINSORIZATION ON HINT FEATURES")

# Active numeric hint features after processing
active_hints = [f for f in NUM_HINT_FEATURES if f in df.columns]

for col in active_hints:
    p01 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    before_range = df[col].max() - df[col].min()
    df[col] = df[col].clip(lower=p01, upper=p99)
    after_range  = df[col].max() - df[col].min()
    n_capped = ((df[col] == p01) | (df[col] == p99)).sum()
    print(f"  {col}: range {before_range:.2f} → {after_range:.2f}  (capped {n_capped} values)")

# ─────────────────────────────────────────────────────────────────
# STEP 2-3: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

section("2-3  FEATURE ENGINEERING")

EPS = 1e-5
engineered = []

# ── a) RATIO FEATURES ─────────────────────────────────────────────
print("  [a] Ratio features …")

def safe_div(a, b):
    return a / (b + EPS)

df["debit_credit_ratio"]  = safe_div(df.get("F527", 0),  df.get("F531",  0))
df["high_value_ratio"]    = safe_div(df.get("F2082", 0), df.get("F2122", 0))
df["freq_amount_ratio"]   = safe_div(df.get("F670", 0),  df.get("F1692", 0))
df["transfer_ratio"]      = safe_div(df.get("F3887", 0), df.get("F3889", 0))

ratio_feats = ["debit_credit_ratio", "high_value_ratio", "freq_amount_ratio", "transfer_ratio"]
engineered += ratio_feats
print(f"    Created: {ratio_feats}")

# ── b) AGGREGATION FEATURES across hint features per row ──────────
print("  [b] Aggregation features …")

hint_matrix = df[active_hints].astype(float)

df["hint_sum"]   = hint_matrix.sum(axis=1)
df["hint_mean"]  = hint_matrix.mean(axis=1)
df["hint_std"]   = hint_matrix.std(axis=1).fillna(0)
df["hint_max"]   = hint_matrix.max(axis=1)
df["hint_min"]   = hint_matrix.min(axis=1)
df["hint_range"] = df["hint_max"] - df["hint_min"]
df["hint_skew"]  = hint_matrix.apply(lambda row: row.skew(), axis=1).fillna(0)

agg_feats = ["hint_sum", "hint_mean", "hint_std", "hint_max", "hint_min", "hint_range", "hint_skew"]
engineered += agg_feats
print(f"    Created: {agg_feats}")

# ── c) INTERACTION FEATURES ───────────────────────────────────────
print("  [c] Interaction features …")

interaction_pairs = [
    ("F115",  "F321",  "F115_x_F321"),
    ("F527",  "F670",  "F527_x_F670"),
    ("F2082", "F2678", "F2082_x_F2678"),
    ("F3836", "F3887", "F3836_x_F3887"),
]
interact_feats = []
for f1, f2, name in interaction_pairs:
    if f1 in df.columns and f2 in df.columns:
        df[name] = df[f1] * df[f2]
        interact_feats.append(name)

engineered += interact_feats
print(f"    Created: {interact_feats}")

# ── d) LOG TRANSFORMS on skewed hint features ──────────────────────
print("  [d] Log transforms on skewed hint features …")

# Features with negative possible values need signed log
# Use np.sign(x) * np.log1p(np.abs(x)) for features that can be negative
# Use np.log1p(x) for strictly positive features

log_feats = []
skewness_threshold = 1.0  # apply log to features with |skewness| > 1

for col in active_hints:
    col_skew = df[col].skew()
    if abs(col_skew) > skewness_threshold:
        min_val = df[col].min()
        new_name = f"{col}_log"
        if min_val < 0:
            # Signed log transform (handles negatives)
            df[new_name] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
        else:
            # Standard log1p
            df[new_name] = np.log1p(df[col])
        log_feats.append(new_name)

engineered += log_feats
print(f"    Created {len(log_feats)} log-transformed features: {log_feats}")

# ── e) PERCENTILE RANK FEATURES ───────────────────────────────────
print("  [e] Percentile rank features …")

pct_feats = []
for col in active_hints:
    new_name = f"{col}_pct_rank"
    df[new_name] = df[col].rank(pct=True)
    pct_feats.append(new_name)

engineered += pct_feats
print(f"    Created {len(pct_feats)} percentile rank features")

# ── f) ISOLATION FOREST ANOMALY SCORE ─────────────────────────────
print("  [f] Isolation Forest anomaly score …")

iso_feats = df[active_hints].astype(float).values
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso_forest.fit(iso_feats)

df["isolation_anomaly_score"] = -iso_forest.score_samples(iso_feats)   # higher = more anomalous
df["is_anomaly"]              = (iso_forest.predict(iso_feats) == -1).astype(int)

iso_anomaly_count = df["is_anomaly"].sum()
print(f"    Anomalies detected by IsolationForest: {iso_anomaly_count} "
      f"({iso_anomaly_count/len(df)*100:.2f}%)")

engineered += ["isolation_anomaly_score", "is_anomaly"]

# Summary of engineered features
print(f"\n  Total engineered features created: {len(engineered)}")

# ─────────────────────────────────────────────────────────────────
# STEP 2-4: FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────

section("2-4  FEATURE SELECTION")

# All feature columns (everything except target)
all_feature_cols = [c for c in df.columns if c != TARGET_COL]
X_all = df[all_feature_cols].copy()
y_all = df[TARGET_COL].copy()

print(f"  Starting feature count: {len(all_feature_cols)}")

# ── Step A: Remove near-zero variance features ────────────────────
print("\n  [A] Near-zero variance removal …")

vt = VarianceThreshold(threshold=0.0)
vt.fit(X_all)
low_var_mask  = vt.get_support()
X_all         = X_all.loc[:, low_var_mask]
print(f"    After variance filter: {X_all.shape[1]} features")

# ── Step B: Remove highly correlated features (>0.95) ────────────
print("\n  [B] High-correlation removal (threshold = 0.95) …")

# Compute correlation on sample if dataset is large
sample_size = min(len(X_all), 5000)
corr_matrix = X_all.sample(sample_size, random_state=RANDOM_STATE).corr().abs()

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]

# Protect hint features and engineered features from being dropped by correlation
protected_features = [f for f in HINT_FEATURES if f in X_all.columns] + engineered
to_drop_corr = [c for c in to_drop_corr if c not in protected_features]

X_all.drop(columns=to_drop_corr, inplace=True)
print(f"    Dropped {len(to_drop_corr)} highly correlated features")
print(f"    After correlation filter: {X_all.shape[1]} features")

# ── Step C: LightGBM feature importance → top 100 ────────────────
print("\n  [C] LightGBM quick fit for feature importance …")

# LightGBM rejects feature names with special JSON characters ({, }, [, ], :, ,, ", ')
# Sanitize ALL column names before fitting
import re

def sanitize_col_name(name):
    """Replace any character not alphanumeric or underscore with underscore."""
    return re.sub(r'[^A-Za-z0-9_]', '_', str(name))

# Build mapping: original → sanitized (handle duplicates by appending index)
orig_cols = list(X_all.columns)
seen       = {}
col_rename = {}
for col in orig_cols:
    clean = sanitize_col_name(col)
    if clean in seen:
        seen[clean] += 1
        clean = f"{clean}_{seen[clean]}"
    else:
        seen[clean] = 0
    col_rename[col] = clean

# Also sanitize engineered & hint lists so they match renamed cols
def remap_list(lst, mapping):
    return [mapping.get(c, c) for c in lst if mapping.get(c, c) in mapping.values()]

X_all.rename(columns=col_rename, inplace=True)

# Update protected feature lists to use sanitized names
engineered_clean       = [col_rename.get(c, c) for c in engineered if col_rename.get(c, c) in X_all.columns]
hint_features_clean    = [col_rename.get(c, c) for c in HINT_FEATURES if col_rename.get(c, c) in X_all.columns]

# Also rename in df for downstream steps
df.rename(columns=col_rename, inplace=True)

# Update active_hints to new names
active_hints_clean = [col_rename.get(c, c) for c in active_hints if col_rename.get(c, c) in df.columns]

print(f"    Sanitized {sum(1 for k,v in col_rename.items() if k != v)} column names with special chars")

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    is_unbalance=True,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=-1
)
lgb_model.fit(X_all, y_all)

importances = pd.Series(lgb_model.feature_importances_, index=X_all.columns)
importances.sort_values(ascending=False, inplace=True)

# Keep top 100 + all hint + all engineered (use sanitized names)
top_100_cols   = list(importances.head(100).index)
mandatory_cols = ([f for f in hint_features_clean if f in X_all.columns]
                 + [f for f in engineered_clean   if f in X_all.columns])

final_feature_cols = list(dict.fromkeys(top_100_cols + mandatory_cols))
final_feature_cols = [c for c in final_feature_cols if c in X_all.columns]


print(f"    Top 100 from LightGBM + mandatory features")
print(f"    Total final features: {len(final_feature_cols)}")

# ── Step D: Save final feature list ──────────────────────────────
feat_file = os.path.join(REPORTS_DIR, "selected_features.txt")
with open(feat_file, "w") as f:
    f.write(f"Total features selected: {len(final_feature_cols)}\n\n")
    f.write("TOP FEATURE IMPORTANCES (from LightGBM quick fit):\n")
    for feat, imp in importances.head(30).items():
        f.write(f"  {feat}: {imp}\n")
    f.write(f"\nFINAL FEATURE LIST:\n")
    for feat in final_feature_cols:
        f.write(f"  {feat}\n")
print(f"  Saved feature list: {feat_file}")

# ─────────────────────────────────────────────────────────────────
# STEP 2-5: CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────────────────────────

section("2-5  CLASS IMBALANCE HANDLING")

X_final = df[final_feature_cols].copy()
y_final = df[TARGET_COL].copy()

counts  = y_final.value_counts()
legit   = counts.get(0, 0)
fraud   = counts.get(1, 0)
ratio   = legit / max(fraud, 1)

print(f"  Legit (0)   : {legit:,}")
print(f"  Fraud (1)   : {fraud:,}")
print(f"  Ratio       : {ratio:.1f}:1")

# ── Train / Test Split (80/20, stratified) ────────────────────────

section("2-6  TRAIN / TEST SPLIT  (80/20 stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final,
    test_size=0.2,
    stratify=y_final,
    random_state=RANDOM_STATE
)

print(f"  X_train: {X_train.shape}   y_train fraud: {y_train.sum()}")
print(f"  X_test : {X_test.shape}    y_test  fraud: {y_test.sum()}")

# ── SMOTE on training set ONLY ───────────────────────────────────
print("\n  Applying SMOTE to training set …")

try:
    from imblearn.over_sampling import SMOTE
    # k_neighbors must be < number of minority class samples
    k_nn = min(5, y_train.sum() - 1)
    smote = SMOTE(k_neighbors=k_nn, random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    smote_counts = pd.Series(y_train_smote).value_counts()
    print(f"    SMOTE applied (k_neighbors={k_nn})")
    print(f"    After SMOTE — Legit: {smote_counts.get(0,0):,}, Fraud: {smote_counts.get(1,0):,}")
    print(f"    New ratio: {smote_counts.get(0,0)/max(smote_counts.get(1,0),1):.1f}:1")

except Exception as e:
    print(f"    SMOTE failed: {e}")
    print("    Falling back to original (imbalanced) training set")
    X_train_smote = X_train.copy()
    y_train_smote = y_train.copy()

# ─────────────────────────────────────────────────────────────────
# STEP 2-7: SAVE PROCESSED DATA
# ─────────────────────────────────────────────────────────────────

section("2-7  SAVING PROCESSED DATA")

X_train_smote.to_csv(os.path.join(DATA_PROC_DIR, "X_train.csv"), index=False)
X_test.to_csv(       os.path.join(DATA_PROC_DIR, "X_test.csv"),  index=False)
pd.Series(y_train_smote, name=TARGET_COL).to_csv(
    os.path.join(DATA_PROC_DIR, "y_train.csv"), index=False)
pd.Series(y_test, name=TARGET_COL).to_csv(
    os.path.join(DATA_PROC_DIR, "y_test.csv"),  index=False)

# Save original (pre-SMOTE) split as well for reference
X_train.to_csv(os.path.join(DATA_PROC_DIR, "X_train_original.csv"), index=False)
pd.Series(y_train, name=TARGET_COL).to_csv(
    os.path.join(DATA_PROC_DIR, "y_train_original.csv"), index=False)

# Save medians for inference time (needed by dashboard)
medians.to_csv(os.path.join(DATA_PROC_DIR, "medians.csv"), header=True)

# Save isolation forest anomaly info
iso_cols = ["isolation_anomaly_score", "is_anomaly"]
iso_cols_available = [c for c in iso_cols if c in df.columns]
df[iso_cols_available].to_csv(os.path.join(DATA_PROC_DIR, "anomaly_scores.csv"), index=False)

print(f"  Saved to {DATA_PROC_DIR}:")
print(f"    X_train.csv          (SMOTE applied) : {X_train_smote.shape}")
print(f"    X_test.csv           (original)      : {X_test.shape}")
print(f"    y_train.csv                          : {len(y_train_smote)}")
print(f"    y_test.csv                           : {len(y_test)}")
print(f"    X_train_original.csv (pre-SMOTE)     : {X_train.shape}")
print(f"    medians.csv")
print(f"    anomaly_scores.csv")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────

section("STEP 2 SUMMARY")

print(f"""
  PREPROCESSING SUMMARY
    Original shape                 : 9082 rows x 3925 cols
    After dropping 50%+ null cols  : {df.shape}
    Imputation                     : Median (all numeric columns)
    Winsorization                  : 1st-99th percentile on {len(active_hints)} hint features

  FEATURE ENGINEERING
    Ratio features         : {len(ratio_feats)}
    Aggregation features   : {len(agg_feats)}
    Interaction features   : {len(interact_feats)}
    Log-transformed        : {len(log_feats)}
    Percentile rank        : {len(pct_feats)}
    Anomaly score          : 2 (IsolationForest)
    Total engineered       : {len(engineered)}

  FEATURE SELECTION
    Final feature count    : {len(final_feature_cols)}
    (Hint + engineered + top 100 LightGBM)

  TRAIN/TEST SPLIT
    X_train (SMOTE)        : {X_train_smote.shape}
    X_test (original)      : {X_test.shape}
    Train fraud count      : {y_train_smote.sum()}
    Test fraud count       : {y_test.sum()}
""")

print("=" * 65)
print("STEP 2 COMPLETE — Preprocessing and feature engineering done.")
print("Type 'continue to next step' to begin Model Building.")
print("=" * 65)
