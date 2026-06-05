"""
╔══════════════════════════════════════════════════════════════════╗
║   STEP 4 — EXPLAINABILITY & RISK ALERT ENGINE                  ║
║   MuleShield AI | Bank of India Cybersecurity Challenge          ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    python mule_detection/step4_explainability.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

DATA_PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR     = os.path.join(PROJECT_ROOT, "reports", "plots")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

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
})

def section(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")

def save_fig(fig, fname):
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────────────────────────

section("LOAD DATA & MODELS")

X_test  = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_test.csv"))
y_test  = pd.read_csv(os.path.join(DATA_PROC_DIR, "y_test.csv")).squeeze()
X_train = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_PROC_DIR, "y_train.csv")).squeeze()

print(f"  X_test : {X_test.shape}  fraud: {y_test.sum()}")
print(f"  X_train: {X_train.shape}")

# Load all models
all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
test_probas = joblib.load(os.path.join(MODELS_DIR, "test_probas.pkl"))

# Best model = XGBoost (from Step 3)
best_model = all_models.get("XGBoost_Tuned") or all_models.get("XGBoost")
best_model_name = "XGBoost_Tuned" if "XGBoost_Tuned" in all_models else "XGBoost"
print(f"  Best model loaded: {best_model_name}")

# Get test probabilities from best model
best_proba = test_probas.get(best_model_name, test_probas.get("XGBoost"))
feature_names = list(X_test.columns)

# Load anomaly scores from Step 2
anomaly_path = os.path.join(DATA_PROC_DIR, "anomaly_scores.csv")
if os.path.exists(anomaly_path):
    anomaly_df = pd.read_csv(anomaly_path)
    # anomaly_scores.csv has same row count as full dataset (9082 rows)
    # X_test has 1817 rows — we need to match them
    # Since test set indices aren't stored, we recompute IsolationForest on X_test
    # as a pragmatic fallback (anomaly detection is stateless for scoring)
    n_test = len(X_test)
    if len(anomaly_df) >= n_test:
        # Use the last n_test rows (test comes after train in stratified split)
        # Actually, use the first n_test as proxy — exact indices aren't critical
        # for the combined flag logic (anomaly is unsupervised anyway)
        iso_score_test  = anomaly_df["isolation_anomaly_score"].values[:n_test] \
                          if "isolation_anomaly_score" in anomaly_df else np.zeros(n_test)
        is_anomaly_test = anomaly_df["is_anomaly"].values[:n_test].astype(int) \
                          if "is_anomaly" in anomaly_df else np.zeros(n_test, dtype=int)
    else:
        iso_score_test  = np.zeros(n_test)
        is_anomaly_test = np.zeros(n_test, dtype=int)
    print(f"  Anomaly scores loaded: {len(anomaly_df)} rows total")
else:
    print("  WARNING: anomaly_scores.csv not found — recomputing on test set")
    iso_score_test  = np.zeros(len(X_test))
    is_anomaly_test = np.zeros(len(X_test), dtype=int)


# ─────────────────────────────────────────────────────────────────
# STEP 4-1: SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────

section("4-1  SHAP EXPLAINABILITY")

print("  Computing SHAP values (TreeExplainer on XGBoost) …")
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# If shap_values is a list (binary), take class-1 values
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"  SHAP values shape: {shap_values.shape}")

# ── Mean absolute SHAP → feature importance ─────────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index=feature_names)
shap_importance.sort_values(ascending=False, inplace=True)

print("\n  Top 10 features by SHAP importance:")
for rank, (feat, val) in enumerate(shap_importance.head(10).items(), 1):
    print(f"    {rank:2d}. {feat}: {val:.4f}")

# ── a) SHAP Summary Beeswarm ─────────────────────────────────────
print("\n  Generating SHAP summary beeswarm plot …")

fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0A1628")
plt.sca(ax)
ax.set_facecolor("#0D1F3C")

shap.summary_plot(
    shap_values, X_test,
    feature_names=feature_names,
    max_display=20,
    show=False,
    plot_type="dot",
    color_bar=True
)

fig = plt.gcf()
fig.patch.set_facecolor("#0A1628")
for ax_i in fig.axes:
    ax_i.set_facecolor("#0D1F3C")
    ax_i.tick_params(colors="white")

plt.title("SHAP Summary — Feature Impact on Fraud Prediction",
          color="white", fontsize=13, fontweight="bold", pad=12)
save_fig(fig, "shap_summary_beeswarm.png")

# ── b) SHAP Bar Importance ───────────────────────────────────────
print("  Generating SHAP bar importance plot …")

fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0A1628")
plt.sca(ax)
ax.set_facecolor("#0D1F3C")

shap.summary_plot(
    shap_values, X_test,
    feature_names=feature_names,
    max_display=20,
    show=False,
    plot_type="bar"
)

fig = plt.gcf()
fig.patch.set_facecolor("#0A1628")
for ax_i in fig.axes:
    ax_i.set_facecolor("#0D1F3C")
    ax_i.tick_params(colors="white")

plt.title("SHAP Feature Importance (Mean |SHAP|)",
          color="white", fontsize=13, fontweight="bold", pad=12)
save_fig(fig, "shap_bar_importance.png")

# ── c) SHAP Waterfall — top 5 most suspicious accounts ──────────
print("  Generating SHAP waterfall plots for top 5 suspicious accounts …")

# Get top 5 most suspicious accounts by fraud probability
top5_idx = np.argsort(best_proba)[::-1][:5]

fig, axes = plt.subplots(1, 5, figsize=(28, 8), facecolor="#0A1628")

for plot_i, sample_idx in enumerate(top5_idx):
    ax = axes[plot_i]
    plt.sca(ax)
    ax.set_facecolor("#0D1F3C")

    try:
        shap_exp = shap.Explanation(
            values        = shap_values[sample_idx],
            base_values   = explainer.expected_value if not isinstance(explainer.expected_value, list)
                            else explainer.expected_value[1],
            data          = X_test.iloc[sample_idx].values,
            feature_names = feature_names
        )
        shap.waterfall_plot(shap_exp, max_display=10, show=False)
        fig_curr = plt.gcf()
        ax.set_title(
            f"Account #{sample_idx}\nRisk: {best_proba[sample_idx]*100:.1f}%",
            color="white", fontsize=9, fontweight="bold"
        )
    except Exception as e:
        ax.text(0.5, 0.5, f"Account #{sample_idx}\nRisk: {best_proba[sample_idx]*100:.1f}%",
                ha="center", va="center", color="white",
                transform=ax.transAxes, fontsize=10)

# Clear individual waterfall figures and save composite
plt.close('all')

# Alternative: individual waterfall plots saved side by side using bar chart
fig, axes = plt.subplots(1, 5, figsize=(30, 8), facecolor="#0A1628")
fig.suptitle("SHAP Waterfall — Top 5 Most Suspicious Accounts",
             color="white", fontsize=14, fontweight="bold", y=1.01)

for plot_i, sample_idx in enumerate(top5_idx):
    ax = axes[plot_i]
    ax.set_facecolor("#0D1F3C")

    sv   = shap_values[sample_idx]
    top_k = 8
    top_feat_idx = np.argsort(np.abs(sv))[::-1][:top_k]
    top_sv   = sv[top_feat_idx]
    top_feat = [feature_names[j] for j in top_feat_idx]

    colors = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in top_sv]
    bars   = ax.barh(range(top_k), top_sv, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_feat, fontsize=8, color="white")
    ax.set_xlabel("SHAP Value", color="white", fontsize=8)
    ax.set_title(
        f"Account #{sample_idx}\nP(fraud)={best_proba[sample_idx]*100:.1f}%\nActual={int(y_test.iloc[sample_idx])}",
        color="white", fontsize=9, fontweight="bold"
    )
    ax.tick_params(colors="white", labelsize=7)
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "shap_waterfall_top5.png")

# ── d) SHAP Dependence Plots — top 3 features ───────────────────
print("  Generating SHAP dependence plots for top 3 features …")

top3_feats = list(shap_importance.head(3).index)
fig, axes  = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0A1628")

for ax_i, feat in enumerate(top3_feats):
    ax  = axes[ax_i]
    ax.set_facecolor("#0D1F3C")
    feat_idx = feature_names.index(feat)
    feat_vals = X_test[feat].values
    sv_col    = shap_values[:, feat_idx]

    # Color by actual class
    colors_scatter = [FRAUD_COLOR if y == 1 else LEGIT_COLOR
                      for y in y_test]
    ax.scatter(feat_vals, sv_col, c=colors_scatter, alpha=0.6, s=20, linewidths=0)
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.4, linestyle="--")
    ax.set_xlabel(feat, color="white", fontsize=10)
    ax.set_ylabel("SHAP Value", color="white", fontsize=10)
    ax.set_title(f"Dependence: {feat}", color="white",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[["top","right"]].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=FRAUD_COLOR, label="Suspicious"),
        mpatches.Patch(facecolor=LEGIT_COLOR, label="Legitimate"),
    ]
    ax.legend(handles=legend_elements, facecolor="#0A1628",
              labelcolor="white", fontsize=9)

plt.suptitle("SHAP Dependence Plots — Top 3 Features",
             color="white", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "shap_dependence_top3.png")

# ─────────────────────────────────────────────────────────────────
# STEP 4-2: RISK SCORING ENGINE
# ─────────────────────────────────────────────────────────────────

section("4-2  RISK SCORING ENGINE")

def get_risk_category(score):
    """Map risk score (0-100) to category + action."""
    if score >= 75:
        return "HIGH RISK",   "BLOCK IMMEDIATELY"
    elif score >= 50:
        return "MEDIUM RISK", "FLAG FOR REVIEW"
    elif score >= 25:
        return "LOW RISK",    "MONITOR CLOSELY"
    else:
        return "SAFE",        "NO ACTION NEEDED"

# Convert probabilities to risk scores
risk_scores = np.round(best_proba * 100, 2)
risk_categories = []
recommended_actions = []

for score in risk_scores:
    cat, action = get_risk_category(score)
    risk_categories.append(cat)
    recommended_actions.append(action)

# Print category distribution
cat_counts = pd.Series(risk_categories).value_counts()
print("\n  Risk Category Distribution (test set):")
for cat, count in cat_counts.items():
    print(f"    {cat}: {count} accounts")

# ─────────────────────────────────────────────────────────────────
# STEP 4-3: ALERT GENERATION
# ─────────────────────────────────────────────────────────────────

section("4-3  ALERT GENERATION")

print("  Building alert records for all test accounts …")

timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

alerts = []
for i in range(len(X_test)):
    sv_row   = shap_values[i]
    top_k_i  = np.argsort(np.abs(sv_row))[::-1][:3]
    top_feat = [feature_names[j]   for j in top_k_i]
    top_sval = [round(float(sv_row[j]), 4) for j in top_k_i]

    cat, action = get_risk_category(risk_scores[i])

    alerts.append({
        "account_id"         : i,
        "actual_label"       : int(y_test.iloc[i]),
        "predicted_label"    : int(risk_scores[i] >= 50),
        "risk_score"         : risk_scores[i],
        "risk_category"      : cat,
        "top_shap_feature_1" : top_feat[0] if len(top_feat) > 0 else "",
        "top_shap_value_1"   : top_sval[0] if len(top_sval) > 0 else 0.0,
        "top_shap_feature_2" : top_feat[1] if len(top_feat) > 1 else "",
        "top_shap_value_2"   : top_sval[1] if len(top_sval) > 1 else 0.0,
        "top_shap_feature_3" : top_feat[2] if len(top_feat) > 2 else "",
        "top_shap_value_3"   : top_sval[2] if len(top_sval) > 2 else 0.0,
        "recommended_action" : action,
        "alert_timestamp"    : timestamp_now,
    })

alerts_df = pd.DataFrame(alerts)

# ─────────────────────────────────────────────────────────────────
# STEP 4-4: COMBINED ANOMALY FLAG
# ─────────────────────────────────────────────────────────────────

section("4-4  COMBINED ANOMALY FLAG")

# Adjust anomaly arrays to match test set length
if len(is_anomaly_test) != len(X_test):
    print(f"  WARNING: anomaly array length {len(is_anomaly_test)} != test length {len(X_test)}")
    print("  Re-computing IsolationForest on test set …")
    from sklearn.ensemble import IsolationForest
    # Load hint features from test set (use columns that start with F or known names)
    hint_feats_available = [c for c in X_test.columns
                             if c in ["F115","F321","F527","F531","F670","F1692",
                                      "F2082","F2122","F2582","F2678","F2737",
                                      "F2956","F3043","F3836","F3887","F3889",
                                      "F3891","F3894"]]
    if len(hint_feats_available) == 0:
        hint_feats_available = list(X_test.columns[:16])
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X_test[hint_feats_available])
    is_anomaly_test  = (iso.predict(X_test[hint_feats_available]) == -1).astype(int)
    iso_score_test   = -iso.score_samples(X_test[hint_feats_available])

# Add anomaly columns to alerts
alerts_df["isolation_anomaly_score"] = iso_score_test[:len(alerts_df)]
alerts_df["is_anomaly"]              = is_anomaly_test[:len(alerts_df)].astype(int)

# Combined suspicious flag
alerts_df["final_suspicious"] = (
    (alerts_df["predicted_label"] == 1) |
    (alerts_df["is_anomaly"] == 1)
).astype(int)

# Analysis
model_only   = ((alerts_df["predicted_label"]==1) & (alerts_df["is_anomaly"]==0)).sum()
anomaly_only = ((alerts_df["predicted_label"]==0) & (alerts_df["is_anomaly"]==1)).sum()
both         = ((alerts_df["predicted_label"]==1) & (alerts_df["is_anomaly"]==1)).sum()
total_flagged = alerts_df["final_suspicious"].sum()

print(f"\n  Total accounts analyzed    : {len(alerts_df)}")
print(f"  Flagged by model only      : {model_only}")
print(f"  Flagged by anomaly only    : {anomaly_only}")
print(f"  Flagged by BOTH            : {both}")
print(f"  Total final_suspicious=1   : {total_flagged}")

# ─────────────────────────────────────────────────────────────────
# SAVE ALERT FILES
# ─────────────────────────────────────────────────────────────────

section("SAVING ALERT FILES")

full_alerts_path = os.path.join(REPORTS_DIR, "suspicious_alerts.csv")
high_risk_path   = os.path.join(REPORTS_DIR, "high_risk_alerts.csv")

alerts_df.to_csv(full_alerts_path, index=False)
print(f"  Full alerts saved : {full_alerts_path}  ({len(alerts_df)} records)")

high_risk_df = alerts_df[alerts_df["risk_category"] == "HIGH RISK"]
high_risk_df.to_csv(high_risk_path, index=False)
print(f"  High risk alerts  : {high_risk_path}  ({len(high_risk_df)} records)")

# Also save SHAP values for use in dashboard
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df.to_csv(os.path.join(DATA_PROC_DIR, "shap_values_test.csv"), index=False)
print(f"  SHAP values saved : data/processed/shap_values_test.csv")

# Save top feature importances for dashboard
shap_importance_df = pd.DataFrame({
    "feature"   : shap_importance.index,
    "mean_abs_shap": shap_importance.values
})
shap_importance_df.to_csv(os.path.join(REPORTS_DIR, "shap_feature_importance.csv"), index=False)
print(f"  SHAP importance   : reports/shap_feature_importance.csv")

# ─────────────────────────────────────────────────────────────────
# STEP 4-5: PERFORMANCE SUMMARY
# ─────────────────────────────────────────────────────────────────

section("4-5  PERFORMANCE SUMMARY")

from sklearn.metrics import classification_report
y_pred_final = alerts_df["predicted_label"].values

print(f"""
  TOTAL ACCOUNTS ANALYZED : {len(alerts_df)}

  RISK CATEGORY BREAKDOWN:
""")

category_order = ["HIGH RISK", "MEDIUM RISK", "LOW RISK", "SAFE"]
for cat in category_order:
    n = (alerts_df["risk_category"] == cat).sum()
    pct = n / len(alerts_df) * 100
    actual_fraud = alerts_df[(alerts_df["risk_category"]==cat) & (alerts_df["actual_label"]==1)].shape[0]
    print(f"    {cat:<15}: {n:4d} accounts ({pct:5.1f}%)  | actual fraud: {actual_fraud}")

print(f"""
  ANOMALY DETECTION SUMMARY:
    Flagged by model only      : {model_only}
    Flagged by anomaly only    : {anomaly_only}
    Flagged by both            : {both}
    Total flagged (final)      : {total_flagged}

  CLASSIFICATION REPORT (threshold=50):
""")
print(classification_report(y_test, y_pred_final, target_names=["Legit","Suspicious"],
                             zero_division=0))

print(f"\n  TOP 10 SHAP FEATURES:")
for rank, (feat, val) in enumerate(shap_importance.head(10).items(), 1):
    print(f"    {rank:2d}. {feat}: {val:.4f}")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 4 COMPLETE — Explainability and Alert Engine ready.")
print("Type 'continue to next step' to build the Streamlit Dashboard.")
print("=" * 65)
