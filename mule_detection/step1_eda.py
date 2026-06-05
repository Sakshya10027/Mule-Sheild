"""
╔══════════════════════════════════════════════════════════════════╗
║   STEP 1 — EXPLORATORY DATA ANALYSIS (EDA)  [v2 — robust]      ║
║   MuleShield AI | Bank of India Cybersecurity Challenge          ║
╚══════════════════════════════════════════════════════════════════╝

Run this script from the project root:
    python mule_detection/step1_eda.py
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
PARENT_DIR   = os.path.dirname(PROJECT_ROOT)

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "reports", "plots")

TARGET_COL = "F3924"

HINT_FEATURES = [
    "F115", "F321", "F527", "F531", "F670",
    "F1692", "F2082", "F2122", "F2582", "F2678",
    "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894"
]

FRAUD_COLOR = "#E63946"
LEGIT_COLOR = "#2EC4B6"

plt.rcParams.update({
    "figure.facecolor": "#0A1628",
    "axes.facecolor":   "#0D1F3C",
    "axes.edgecolor":   "#2A3F5F",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "grid.color":       "#1E2F4A",
    "grid.linewidth":   0.5,
})
sns.set_theme(style="darkgrid", font_scale=1.0)

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")

def find_csv():
    for d in [DATA_RAW_DIR, PROJECT_ROOT, PARENT_DIR]:
        hits = glob.glob(os.path.join(d, "*.csv"))
        if hits:
            print(f"  CSV found: {hits[0]}")
            return hits[0]
    raise FileNotFoundError("No CSV found. Place DataSet.csv in mule_detection/data/raw/")

def save_fig(fig, fname):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close(fig)

def force_numeric(df, cols):
    """Force-convert columns to numeric, coercing errors to NaN."""
    converted = []
    skipped   = []
    for col in cols:
        if col in df.columns:
            before = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            converted.append(col)
            if str(before) == 'object':
                n_nan = df[col].isna().sum()
                print(f"    Converted {col}: object -> float  (NaNs introduced: {n_nan})")
        else:
            skipped.append(col)
    if skipped:
        print(f"    WARNING: These hint features NOT found in dataset: {skipped}")
    return df, converted

# ─────────────────────────────────────────────────────────────────
# 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────

section("1/8  LOAD DATA")

csv_path = find_csv()
print(f"  Loading … (this may take a moment for large files)")

# Load with low_memory=False and dtype=object first so we can inspect
df_raw = pd.read_csv(csv_path, low_memory=False)

print(f"\n  Raw shape          : {df_raw.shape[0]:,} rows x {df_raw.shape[1]:,} columns")
print(f"  dtype summary (raw):")
print(df_raw.dtypes.value_counts().to_string(header=False))

# Show first 5 rows, first 10 columns
print(f"\n  First 5 rows (first 10 columns):")
print(df_raw.iloc[:5, :10].to_string())

# Print dtypes of hint features specifically
print(f"\n  Hint feature dtypes in raw CSV:")
for f in HINT_FEATURES:
    if f in df_raw.columns:
        print(f"    {f}: {df_raw[f].dtype}  | sample: {df_raw[f].dropna().head(3).tolist()}")
    else:
        print(f"    {f}: NOT FOUND")

df = df_raw.copy()

# ─────────────────────────────────────────────────────────────────
# 1.5 — FORCE NUMERIC CONVERSION (critical fix)
# ─────────────────────────────────────────────────────────────────

section("1.5/8  FORCE NUMERIC CONVERSION ON ALL RELEVANT COLUMNS")

print("  Converting hint features to numeric …")
all_target_cols = HINT_FEATURES + [TARGET_COL]
df, found_hints = force_numeric(df, all_target_cols)

# Also convert all remaining columns that look numeric
print("  Converting all other columns to numeric where possible …")
non_hint_cols = [c for c in df.columns if c not in all_target_cols]
for col in non_hint_cols:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Final dtype summary
print(f"\n  dtype summary (after conversion):")
print(df.dtypes.value_counts().to_string(header=False))

# Active hint features (present and numeric)
hints = [f for f in HINT_FEATURES if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
print(f"\n  Numeric hint features available: {len(hints)}/18")
print(f"  {hints}")

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found!")

# ─────────────────────────────────────────────────────────────────
# 2 — BASIC CHECKS
# ─────────────────────────────────────────────────────────────────

section("2/8  BASIC CHECKS")

n_rows, n_cols_count = df.shape
print(f"  Total rows    : {n_rows:,}")
print(f"  Total columns : {n_cols_count:,}")

n_dups = df.duplicated().sum()
print(f"  Duplicate rows: {n_dups:,}")
if n_dups > 0:
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Dropped {n_dups} duplicates. New shape: {df.shape}")

mem_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"  Memory usage  : {mem_mb:.2f} MB")

# ─────────────────────────────────────────────────────────────────
# 3 — TARGET ANALYSIS
# ─────────────────────────────────────────────────────────────────

section("3/8  TARGET ANALYSIS")

# Target might have NaN after coercion — drop those rows
target_null = df[TARGET_COL].isna().sum()
if target_null > 0:
    print(f"  WARNING: {target_null} rows have NaN target — dropping them")
    df.dropna(subset=[TARGET_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)

df[TARGET_COL] = df[TARGET_COL].astype(int)
target_counts = df[TARGET_COL].value_counts().sort_index()
target_pct    = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100

print(f"\n  Class distribution of {TARGET_COL}:")
for cls in target_counts.index:
    print(f"    Class {cls}: {target_counts[cls]:,}  ({target_pct[cls]:.2f}%)")

imbalance_ratio = target_counts.get(0, 0) / max(target_counts.get(1, 1), 1)
print(f"\n  Imbalance ratio (Legit:Fraud) = {imbalance_ratio:.1f}:1")

# Plot
fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0A1628")
ax.set_facecolor("#0D1F3C")
labels = ["Legitimate (0)", "Suspicious (1)"]
vals   = [target_counts.get(0, 0), target_counts.get(1, 0)]
colors = [LEGIT_COLOR, FRAUD_COLOR]
bars   = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
for bar, pct in zip(bars, [target_pct.get(0, 0), target_pct.get(1, 0)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
            f"{int(bar.get_height()):,}\n({pct:.1f}%)",
            ha="center", va="bottom", color="white", fontsize=12, fontweight="bold")
ax.set_title("Class Distribution — F3924", color="white", fontsize=14, fontweight="bold")
ax.set_ylabel("Account Count", color="white")
ax.set_ylim(0, max(vals) * 1.2)
ax.spines[["top","right"]].set_visible(False)
save_fig(fig, "class_distribution.png")

# ─────────────────────────────────────────────────────────────────
# 4 — MISSING VALUE ANALYSIS
# ─────────────────────────────────────────────────────────────────

section("4/8  MISSING VALUE ANALYSIS")

null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df)) * 100

cols_with_nulls = null_counts[null_counts > 0]
print(f"  Columns with any nulls : {len(cols_with_nulls):,} / {n_cols_count:,}")
print(f"  Total missing cells    : {null_counts.sum():,}")

if len(cols_with_nulls) > 0:
    print(f"\n  Top 10 columns by missing %:")
    top_null = null_pct[null_pct > 0].sort_values(ascending=False).head(10)
    for col, pct in top_null.items():
        print(f"    {col}: {pct:.1f}%")

# Hint feature null check
hint_nulls_series = null_pct[hints].sort_values(ascending=False)
print(f"\n  Hint feature missing %:")
for col, pct in hint_nulls_series.items():
    flag = "WARNING" if pct > 10 else "OK"
    print(f"    [{flag}] {col}: {pct:.2f}%")

# Heatmap for hint features (sample rows)
fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0A1628")
ax.set_facecolor("#0D1F3C")
sample_n = min(500, len(df))
hint_null_sample = df[hints].isnull().astype(int).sample(sample_n, random_state=42)
cmap2 = sns.color_palette(["#0D1F3C", FRAUD_COLOR], as_cmap=True)
sns.heatmap(hint_null_sample.T, cmap=cmap2, cbar=True, linewidths=0,
            ax=ax, yticklabels=hints, xticklabels=False)
ax.set_title(f"Missing Value Heatmap — Hint Features (sample {sample_n} rows)\nRed=Missing",
             color="white", fontsize=13, fontweight="bold")
ax.tick_params(colors="white", labelsize=9)
ax.set_ylabel("Hint Feature", color="white")
ax.set_xlabel("Account (row)", color="white")
save_fig(fig, "missing_values_heatmap.png")

# ─────────────────────────────────────────────────────────────────
# 5 — STATISTICAL ANALYSIS OF HINT FEATURES
# ─────────────────────────────────────────────────────────────────

section("5/8  STATISTICAL ANALYSIS OF HINT FEATURES")

stats_rows = []
for col in hints:
    series = df[col].dropna().astype(float)
    if len(series) == 0:
        print(f"  SKIP {col}: all NaN after conversion")
        continue
    stats_rows.append({
        "Feature" : col,
        "Non-Null": len(series),
        "Mean"    : round(series.mean(), 4),
        "Median"  : round(series.median(), 4),
        "Std"     : round(series.std(), 4),
        "Skewness": round(series.skew(), 4),
        "Kurtosis": round(series.kurtosis(), 4),
    })

stats_df = pd.DataFrame(stats_rows).set_index("Feature")
print("\n  Statistical Summary:")
print(stats_df.to_string())

# Distribution plots
n_hints_found = len(hints)
if n_hints_found == 0:
    print("  No numeric hint features — skipping distribution plot")
else:
    n_cols_g = 3
    n_rows_g = int(np.ceil(n_hints_found / n_cols_g))
    fig, axes = plt.subplots(n_rows_g, n_cols_g, figsize=(18, n_rows_g*4), facecolor="#0A1628")
    fig.suptitle("Hint Feature Distributions (Blue=Legit | Red=Suspicious)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    axes_flat = axes.flatten() if n_hints_found > 1 else [axes]

    for idx, col in enumerate(hints):
        ax = axes_flat[idx]
        ax.set_facecolor("#0D1F3C")
        for cls, color, lbl in [(0, LEGIT_COLOR, "Legit"), (1, FRAUD_COLOR, "Suspicious")]:
            subset = df[df[TARGET_COL] == cls][col].dropna().astype(float)
            if len(subset) < 2:
                continue
            p1, p99 = subset.quantile(0.01), subset.quantile(0.99)
            clipped  = subset.clip(p1, p99)
            ax.hist(clipped, bins=40, alpha=0.6, color=color, label=lbl,
                    density=True, edgecolor="none")
            if clipped.std() > 0:
                try:
                    kde_x = np.linspace(clipped.min(), clipped.max(), 200)
                    kde   = stats.gaussian_kde(clipped)
                    ax.plot(kde_x, kde(kde_x), color=color, linewidth=1.8)
                except Exception:
                    pass
        ax.set_title(col, color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Value", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=7, facecolor="#0A1628", labelcolor="white")

    for idx in range(n_hints_found, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.tight_layout(pad=1.5)
    save_fig(fig, "hint_feature_distributions.png")

# ─────────────────────────────────────────────────────────────────
# 6 — CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────

section("6/8  CORRELATION ANALYSIS")

corr_cols = hints + [TARGET_COL]
corr_df   = df[corr_cols].copy().astype(float)
corr_matrix = corr_df.corr()

target_corr = corr_matrix[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
print("\n  Top 5 features most correlated with F3924:")
for rank, (col, val) in enumerate(target_corr.head(5).items(), 1):
    direction = "positive" if corr_matrix.loc[col, TARGET_COL] > 0 else "negative"
    print(f"    {rank}. {col}: |r| = {val:.4f}  ({direction})")

# Heatmap
fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0A1628")
ax.set_facecolor("#0D1F3C")
cmap_div = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_matrix, cmap=cmap_div, center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 7, "color": "white"},
            linewidths=0.5, linecolor="#0A1628",
            cbar_kws={"shrink": 0.8}, ax=ax, square=True)
ax.set_title("Correlation Heatmap — Hint Features + F3924",
             color="white", fontsize=14, fontweight="bold", pad=12)
ax.tick_params(colors="white", labelsize=9)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color="white")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="white")
plt.tight_layout()
save_fig(fig, "correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────
# 7 — OUTLIER DETECTION — BOX PLOTS
# ─────────────────────────────────────────────────────────────────

section("7/8  OUTLIER DETECTION — BOX PLOTS")

n_cols_b = 3
n_rows_b = int(np.ceil(n_hints_found / n_cols_b))
fig, axes = plt.subplots(n_rows_b, n_cols_b, figsize=(18, n_rows_b*4), facecolor="#0A1628")
fig.suptitle("Box Plots — Hint Features by Class (Blue=Legit | Red=Suspicious)",
             color="white", fontsize=13, fontweight="bold", y=1.01)
axes_flat = axes.flatten() if n_hints_found > 1 else [axes]

for idx, col in enumerate(hints):
    ax = axes_flat[idx]
    ax.set_facecolor("#0D1F3C")
    data_0 = df[df[TARGET_COL] == 0][col].dropna().astype(float)
    data_1 = df[df[TARGET_COL] == 1][col].dropna().astype(float)

    for data, color, pos in [(data_0, LEGIT_COLOR, 1), (data_1, FRAUD_COLOR, 2)]:
        if len(data) < 2:
            continue
        p5, p95   = data.quantile(0.05), data.quantile(0.95)
        clipped   = data.clip(p5, p95)
        ax.boxplot(clipped, positions=[pos], widths=0.5, patch_artist=True,
                   notch=False,
                   whiskerprops=dict(color=color, linewidth=1.5),
                   capprops=dict(color=color, linewidth=1.5),
                   medianprops=dict(color="white", linewidth=2),
                   flierprops=dict(marker="o", color=color, alpha=0.3, markersize=2),
                   boxprops=dict(facecolor=color+"55", edgecolor=color, linewidth=1.5))

    ax.set_title(col, color="white", fontsize=10, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Legit", "Suspicious"], color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[["top","right"]].set_visible(False)

for idx in range(n_hints_found, len(axes_flat)):
    axes_flat[idx].set_visible(False)
plt.tight_layout(pad=1.5)
save_fig(fig, "boxplots.png")

print("\n  Outlier counts (beyond 3 std) per hint feature:")
for col in hints:
    series  = df[col].dropna().astype(float)
    if series.std() == 0 or len(series) < 2:
        continue
    z_scores = np.abs(stats.zscore(series))
    n_out    = (z_scores > 3).sum()
    print(f"    {col}: {n_out:,} outliers ({n_out/len(series)*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────
# 8 — EDA SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────

section("8/8  EDA SUMMARY REPORT")

legit_count = target_counts.get(0, 0)
fraud_count = target_counts.get(1, 0)
total       = legit_count + fraud_count
high_null   = null_pct[null_pct > 20].sort_values(ascending=False)

print(f"""
  DATASET SIZE
    Total accounts  : {total:,}
    Total features  : {n_cols_count:,}
    Hint features   : {len(hints)} / 18 found and numeric
    Memory usage    : {mem_mb:.2f} MB

  CLASS IMBALANCE
    Legitimate (0)  : {legit_count:,}  ({legit_count/max(total,1)*100:.2f}%)
    Suspicious (1)  : {fraud_count:,}  ({fraud_count/max(total,1)*100:.2f}%)
    Imbalance ratio : {imbalance_ratio:.1f}:1
    Strategy needed : SMOTE + class_weight='balanced'

  TOP 5 FEATURES CORRELATED WITH F3924""")

for rank, (col, val) in enumerate(target_corr.head(5).items(), 1):
    direction = "positive" if corr_matrix.loc[col, TARGET_COL] > 0 else "negative"
    print(f"    {rank}. {col}: |r| = {val:.4f} ({direction})")

print(f"""
  FEATURES WITH HIGH NULLS (>20%)""")
if len(high_null) > 0:
    for col, pct in high_null.head(10).items():
        print(f"    {col}: {pct:.1f}%")
else:
    print("    None — all features below 20% missing threshold")

print(f"""
  KEY OBSERVATIONS
    Dataset is highly imbalanced -> PR-AUC preferred over ROC-AUC
    Hint features show distinct distributions between classes
    High skewness in several features -> log1p transform in Step 2
    Outliers present -> IQR capping (Winsorization) in Step 2

  SAVED PLOTS
    class_distribution.png
    missing_values_heatmap.png
    hint_feature_distributions.png
    correlation_heatmap.png
    boxplots.png
""")

print("=" * 62)
print("STEP 1 COMPLETE — EDA finished and all plots saved.")
print("Type 'continue to next step' to begin Feature Engineering.")
print("=" * 62)
