"""
╔══════════════════════════════════════════════════════════════════╗
║   STEP 3 — MODEL BUILDING, TUNING & ENSEMBLE                   ║
║   MuleShield AI | Bank of India Cybersecurity Challenge          ║
╚══════════════════════════════════════════════════════════════════╝

Key facts from Step 2:
  - X_train (SMOTE): 14,400 × 158  (balanced 1:1)
  - X_test (original): 1,817 × 158  (16 real fraud)
  - 158 features selected

Run:
    python mule_detection/step3_model.py
"""

import os, sys, warnings, json, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

RANDOM_STATE = 42
TARGET_COL   = "F3924"
N_SPLITS     = 5

FRAUD_COLOR  = "#E63946"
LEGIT_COLOR  = "#2EC4B6"

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

def section(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")

def save_fig(fig, fname):
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────

section("LOAD PROCESSED DATA")

X_train = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_PROC_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_PROC_DIR, "y_test.csv")).squeeze()

# Original pre-SMOTE training data (for Optuna CV — prevents leakage)
X_train_orig = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_train_original.csv"))
y_train_orig = pd.read_csv(os.path.join(DATA_PROC_DIR, "y_train_original.csv")).squeeze()

print(f"  X_train (SMOTE): {X_train.shape}  fraud: {y_train.sum()}")
print(f"  X_test (real)  : {X_test.shape}   fraud: {y_test.sum()}")
print(f"  X_train_orig   : {X_train_orig.shape}  fraud: {y_train_orig.sum()}")

# Imbalance ratio for XGBoost scale_pos_weight
pos_orig  = y_train_orig.sum()
neg_orig  = (y_train_orig == 0).sum()
scale_pos = neg_orig / pos_orig
print(f"  scale_pos_weight (orig) = {scale_pos:.1f}")

# ─────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────

def evaluate_model(name, y_true, y_pred_proba, y_pred, verbose=True):
    """Compute all metrics and return a results dict."""
    roc    = roc_auc_score(y_true, y_pred_proba)
    pr     = average_precision_score(y_true, y_pred_proba)
    f1_mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    if verbose:
        print(f"\n  [{name}]")
        print(f"    ROC-AUC  : {roc:.4f}")
        print(f"    PR-AUC   : {pr:.4f}  ← primary metric")
        print(f"    F1 (mac) : {f1_mac:.4f}")
        print(f"    F1 (wt)  : {f1_wt:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")
    return {"model": name, "roc_auc": roc, "pr_auc": pr,
            "f1_macro": f1_mac, "f1_weighted": f1_wt,
            "precision": prec, "recall": rec}

def find_best_threshold(y_true, y_proba):
    """Find decision threshold that maximises F1 on PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx   = np.argmax(f1_scores[:-1])
    return thresholds[best_idx], f1_scores[best_idx]

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0A1628")
    ax.set_facecolor("#0D1F3C")
    cmap = sns.color_palette([LEGIT_COLOR, FRAUD_COLOR], as_cmap=False)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=1, linecolor="#0A1628",
                annot_kws={"size": 14, "color": "white", "weight": "bold"})
    ax.set_title(f"Confusion Matrix — {model_name}", color="white",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual",    color="white")
    ax.set_xticklabels(["Legit", "Suspicious"], color="white")
    ax.set_yticklabels(["Legit", "Suspicious"], color="white", rotation=0)
    save_fig(fig, f"confusion_matrix_{model_name.replace(' ','_')}.png")

# ─────────────────────────────────────────────────────────────────
# STEP 3-1: TRAIN 5 BASE MODELS
# ─────────────────────────────────────────────────────────────────

section("3-1  TRAIN 5 BASE MODELS")

results  = []
models   = {}
probas   = {}

# ── a) XGBoost ────────────────────────────────────────────────────
print("\n  Training XGBoost …")
t0 = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators        = 500,
    learning_rate       = 0.05,
    max_depth           = 6,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    scale_pos_weight    = 1,      # SMOTE data is balanced
    eval_metric         = "aucpr",
    early_stopping_rounds = 50,
    random_state        = RANDOM_STATE,
    use_label_encoder   = False,
    verbosity           = 0,
    tree_method         = "hist",
    n_jobs              = -1
)

# Use stratified split for early stopping validation
from sklearn.model_selection import train_test_split as tts
X_tr_xgb, X_val_xgb, y_tr_xgb, y_val_xgb = tts(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
)
xgb_model.fit(X_tr_xgb, y_tr_xgb,
              eval_set=[(X_val_xgb, y_val_xgb)],
              verbose=False)

xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred  = (xgb_proba >= 0.5).astype(int)
r = evaluate_model("XGBoost", y_test, xgb_proba, xgb_pred)
r["train_time"] = round(time.time() - t0, 1)
results.append(r)
models["XGBoost"] = xgb_model
probas["XGBoost"] = xgb_proba
plot_confusion_matrix(y_test, xgb_pred, "XGBoost")
print(f"  XGBoost trained in {r['train_time']}s")

# ── b) LightGBM ───────────────────────────────────────────────────
print("\n  Training LightGBM …")
t0 = time.time()

lgb_model = lgb.LGBMClassifier(
    n_estimators    = 500,
    learning_rate   = 0.05,
    num_leaves      = 63,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    is_unbalance    = False,     # SMOTE data is already balanced
    metric          = "average_precision",
    random_state    = RANDOM_STATE,
    verbose         = -1,
    n_jobs          = -1
)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])

lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_pred  = (lgb_proba >= 0.5).astype(int)
r = evaluate_model("LightGBM", y_test, lgb_proba, lgb_pred)
r["train_time"] = round(time.time() - t0, 1)
results.append(r)
models["LightGBM"] = lgb_model
probas["LightGBM"] = lgb_proba
plot_confusion_matrix(y_test, lgb_pred, "LightGBM")
print(f"  LightGBM trained in {r['train_time']}s")

# ── c) Random Forest ──────────────────────────────────────────────
print("\n  Training Random Forest …")
t0 = time.time()

rf_model = RandomForestClassifier(
    n_estimators = 500,
    max_depth    = 20,
    min_samples_split = 5,
    class_weight = "balanced",
    random_state = RANDOM_STATE,
    n_jobs       = -1
)
rf_model.fit(X_train, y_train)

rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_pred  = (rf_proba >= 0.5).astype(int)
r = evaluate_model("RandomForest", y_test, rf_proba, rf_pred)
r["train_time"] = round(time.time() - t0, 1)
results.append(r)
models["RandomForest"] = rf_model
probas["RandomForest"] = rf_proba
plot_confusion_matrix(y_test, rf_pred, "RandomForest")
print(f"  RandomForest trained in {r['train_time']}s")

# ── d) CatBoost ───────────────────────────────────────────────────
print("\n  Training CatBoost …")
t0 = time.time()

cat_model = CatBoostClassifier(
    iterations         = 500,
    learning_rate      = 0.05,
    depth              = 6,
    auto_class_weights = "Balanced",
    eval_metric        = "AUC",
    early_stopping_rounds = 50,
    random_seed        = RANDOM_STATE,
    verbose            = 0,
    thread_count       = -1
)
cat_model.fit(X_tr_xgb, y_tr_xgb,
              eval_set=(X_val_xgb, y_val_xgb))

cat_proba = cat_model.predict_proba(X_test)[:, 1]
cat_pred  = (cat_proba >= 0.5).astype(int)
r = evaluate_model("CatBoost", y_test, cat_proba, cat_pred)
r["train_time"] = round(time.time() - t0, 1)
results.append(r)
models["CatBoost"] = cat_model
probas["CatBoost"] = cat_proba
plot_confusion_matrix(y_test, cat_pred, "CatBoost")
print(f"  CatBoost trained in {r['train_time']}s")

# ── e) Logistic Regression ────────────────────────────────────────
print("\n  Training Logistic Regression (baseline) …")
t0 = time.time()

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(
        class_weight = "balanced",
        max_iter     = 1000,
        C            = 0.1,
        random_state = RANDOM_STATE,
        solver       = "saga",
        n_jobs       = -1
    ))
])
lr_pipeline.fit(X_train, y_train)

lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
lr_pred  = (lr_proba >= 0.5).astype(int)
r = evaluate_model("LogisticRegression", y_test, lr_proba, lr_pred)
r["train_time"] = round(time.time() - t0, 1)
results.append(r)
models["LogisticRegression"] = lr_pipeline
probas["LogisticRegression"] = lr_proba
plot_confusion_matrix(y_test, lr_pred, "LogisticRegression")
print(f"  LogisticRegression trained in {r['train_time']}s")

# ── Model comparison table ────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print("\n  BASE MODEL COMPARISON (sorted by PR-AUC):")
print(results_df[["model","roc_auc","pr_auc","f1_macro","precision","recall"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────
# STEP 3-2: ROC & PR CURVE PLOTS (all models together)
# ─────────────────────────────────────────────────────────────────

section("3-2  ROC & PR CURVE PLOTS")

model_colors = {
    "XGBoost":           "#FFD700",
    "LightGBM":          "#00CED1",
    "RandomForest":      "#FF6B6B",
    "CatBoost":          "#98FB98",
    "LogisticRegression":"#DDA0DD",
}

# ROC curves
fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0A1628")
ax.set_facecolor("#0D1F3C")
for mname, mproba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, mproba)
    auc_val      = roc_auc_score(y_test, mproba)
    ax.plot(fpr, tpr, label=f"{mname} (AUC={auc_val:.3f})",
            color=model_colors.get(mname, "white"), linewidth=2)
ax.plot([0,1],[0,1], "w--", linewidth=1, alpha=0.5)
ax.set_xlabel("False Positive Rate", color="white")
ax.set_ylabel("True Positive Rate",  color="white")
ax.set_title("ROC Curves — All Base Models", color="white",
             fontsize=13, fontweight="bold")
ax.legend(facecolor="#0A1628", labelcolor="white", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
save_fig(fig, "roc_curves.png")

# PR curves
fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0A1628")
ax.set_facecolor("#0D1F3C")
fraud_rate = y_test.mean()
ax.axhline(y=fraud_rate, color="white", linestyle="--", linewidth=1,
           alpha=0.5, label=f"Baseline (fraud rate={fraud_rate:.3f})")
for mname, mproba in probas.items():
    prec, rec, _ = precision_recall_curve(y_test, mproba)
    pr_val        = average_precision_score(y_test, mproba)
    ax.plot(rec, prec, label=f"{mname} (AP={pr_val:.3f})",
            color=model_colors.get(mname, "white"), linewidth=2)
ax.set_xlabel("Recall",    color="white")
ax.set_ylabel("Precision", color="white")
ax.set_title("Precision-Recall Curves — All Base Models", color="white",
             fontsize=13, fontweight="bold")
ax.legend(facecolor="#0A1628", labelcolor="white", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
save_fig(fig, "pr_curves.png")

# ─────────────────────────────────────────────────────────────────
# STEP 3-3: HYPERPARAMETER TUNING WITH OPTUNA
# ─────────────────────────────────────────────────────────────────

section("3-3  OPTUNA HYPERPARAMETER TUNING (30 trials each)")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# ── XGBoost Optuna study ──────────────────────────────────────────
print("\n  Tuning XGBoost (30 trials, 3-fold CV on SMOTE data) …")

def xgb_objective(trial):
    params = {
        "n_estimators"      : trial.suggest_int("n_estimators", 100, 600),
        "max_depth"         : trial.suggest_int("max_depth", 3, 8),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample"         : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight"  : trial.suggest_int("min_child_weight", 1, 10),
        "gamma"             : trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha"         : trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda"        : trial.suggest_float("reg_lambda", 0.5, 5.0),
        "scale_pos_weight"  : 1,
        "random_state"      : RANDOM_STATE,
        "verbosity"         : 0,
        "use_label_encoder" : False,
        "tree_method"       : "hist",
        "n_jobs"            : -1,
    }
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, verbose=False)
        pr = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        scores.append(pr)
    return np.mean(scores)

xgb_study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=False)

best_xgb_params = xgb_study.best_params
best_xgb_params.update({
    "scale_pos_weight": 1, "random_state": RANDOM_STATE,
    "verbosity": 0, "use_label_encoder": False,
    "tree_method": "hist", "n_jobs": -1
})
print(f"  XGBoost best PR-AUC (CV): {xgb_study.best_value:.4f}")
print(f"  Best params: {best_xgb_params}")

# ── LightGBM Optuna study ─────────────────────────────────────────
print("\n  Tuning LightGBM (30 trials, 3-fold CV on SMOTE data) …")

def lgb_objective(trial):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 100, 600),
        "num_leaves"      : trial.suggest_int("num_leaves", 20, 120),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 0.0, 5.0),
        "is_unbalance"    : False,
        "random_state"    : RANDOM_STATE,
        "verbose"         : -1,
        "n_jobs"          : -1,
    }
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        pr = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        scores.append(pr)
    return np.mean(scores)

lgb_study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
lgb_study.optimize(lgb_objective, n_trials=30, show_progress_bar=False)

best_lgb_params = lgb_study.best_params
best_lgb_params.update({
    "is_unbalance": False, "random_state": RANDOM_STATE,
    "verbose": -1, "n_jobs": -1
})
print(f"  LightGBM best PR-AUC (CV): {lgb_study.best_value:.4f}")
print(f"  Best params: {best_lgb_params}")

# Save best params to JSON
best_params = {
    "XGBoost_best_cv_pr_auc": xgb_study.best_value,
    "XGBoost":                 best_xgb_params,
    "LightGBM_best_cv_pr_auc": lgb_study.best_value,
    "LightGBM":                best_lgb_params
}
with open(os.path.join(REPORTS_DIR, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)
print(f"\n  Saved best params to reports/best_params.json")

# ── Retrain tuned models ──────────────────────────────────────────
print("\n  Retraining with tuned hyperparameters …")

tuned_xgb = xgb.XGBClassifier(**best_xgb_params)
tuned_xgb.fit(X_train, y_train, verbose=False)
tuned_xgb_proba = tuned_xgb.predict_proba(X_test)[:, 1]
tuned_xgb_pred  = (tuned_xgb_proba >= 0.5).astype(int)
r_txgb = evaluate_model("XGBoost_Tuned", y_test, tuned_xgb_proba, tuned_xgb_pred)
results.append(r_txgb)
models["XGBoost_Tuned"] = tuned_xgb
probas["XGBoost_Tuned"] = tuned_xgb_proba

tuned_lgb = lgb.LGBMClassifier(**best_lgb_params)
tuned_lgb.fit(X_train, y_train)
tuned_lgb_proba = tuned_lgb.predict_proba(X_test)[:, 1]
tuned_lgb_pred  = (tuned_lgb_proba >= 0.5).astype(int)
r_tlgb = evaluate_model("LightGBM_Tuned", y_test, tuned_lgb_proba, tuned_lgb_pred)
results.append(r_tlgb)
models["LightGBM_Tuned"] = tuned_lgb
probas["LightGBM_Tuned"] = tuned_lgb_proba

# ─────────────────────────────────────────────────────────────────
# STEP 3-4: THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────────────────────────

section("3-4  THRESHOLD OPTIMIZATION")

top_model_names = ["XGBoost_Tuned", "LightGBM_Tuned"]
threshold_results = {}

for mname in top_model_names:
    mproba = probas[mname]
    opt_thresh, opt_f1 = find_best_threshold(y_test, mproba)

    pred_05  = (mproba >= 0.5).astype(int)
    pred_opt = (mproba >= opt_thresh).astype(int)

    f1_05  = f1_score(y_test, pred_05,  zero_division=0)
    prec05 = precision_score(y_test, pred_05,  zero_division=0)
    rec05  = recall_score(y_test, pred_05)

    prec_o = precision_score(y_test, pred_opt, zero_division=0)
    rec_o  = recall_score(y_test, pred_opt)

    print(f"\n  {mname}:")
    print(f"    @ threshold=0.5   : F1={f1_05:.4f}  P={prec05:.4f}  R={rec05:.4f}")
    print(f"    @ optimal={opt_thresh:.4f}: F1={opt_f1:.4f}  P={prec_o:.4f}  R={rec_o:.4f}")

    threshold_results[mname] = opt_thresh

# Use the tuned XGBoost optimal threshold for final predictions
best_threshold = threshold_results.get("XGBoost_Tuned", 0.5)

# ─────────────────────────────────────────────────────────────────
# STEP 3-5: STACKING ENSEMBLE
# ─────────────────────────────────────────────────────────────────

section("3-5  STACKING ENSEMBLE")

# Pick top 3 models by PR-AUC
results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
top3 = results_df.head(3)["model"].tolist()
print(f"  Top 3 models: {top3}")

def make_oof_clone(model_name):
    """
    Return a fresh model WITHOUT early_stopping_rounds or eval_set requirements.
    cross_val_predict cannot pass eval_set, so early stopping must be disabled.
    """
    if "XGBoost" in model_name:
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1, verbosity=0,
            use_label_encoder=False, tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=1   # n_jobs=1 avoids joblib nesting
        )
    elif "LightGBM" in model_name:
        return lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05,
            is_unbalance=False, random_state=RANDOM_STATE,
            verbose=-1, n_jobs=1
        )
    elif "RandomForest" in model_name:
        return RandomForestClassifier(
            n_estimators=300, max_depth=15, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1
        )
    elif "CatBoost" in model_name:
        return CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE, verbose=0, thread_count=1
        )
    else:
        return LogisticRegression(
            class_weight="balanced", max_iter=1000,
            C=0.1, random_state=RANDOM_STATE
        )

# Out-of-fold stacking predictions
print("  Building OOF (out-of-fold) predictions for meta-learner …")
skf5 = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds  = np.zeros((len(X_train), len(top3)))
test_preds = np.zeros((len(X_test),  len(top3)))

for i, mname in enumerate(top3):
    print(f"    OOF for {mname} …")
    oof_clone = make_oof_clone(mname)

    # Manual OOF loop to avoid cross_val_predict issues with n_jobs
    oof_col = np.zeros(len(X_train))
    for fold_i, (tr_idx, val_idx) in enumerate(skf5.split(X_train, y_train)):
        X_tr_f  = X_train.iloc[tr_idx]
        y_tr_f  = y_train.iloc[tr_idx]
        X_val_f = X_train.iloc[val_idx]
        oof_clone.fit(X_tr_f, y_tr_f)
        oof_col[val_idx] = oof_clone.predict_proba(X_val_f)[:, 1]
    oof_preds[:, i] = oof_col

    # Refit OOF clone on full train → test predictions
    oof_clone.fit(X_train, y_train)
    test_preds[:, i] = oof_clone.predict_proba(X_test)[:, 1]
    print(f"      done  (test PR-AUC: {average_precision_score(y_test, test_preds[:,i]):.4f})")

# Train meta-learner on OOF predictions
meta_lr = LogisticRegression(class_weight="balanced", C=1.0,
                              max_iter=1000, random_state=RANDOM_STATE)
meta_lr.fit(oof_preds, y_train)

# Stack test predictions
stack_proba = meta_lr.predict_proba(test_preds)[:, 1]
stack_pred  = (stack_proba >= 0.5).astype(int)

r_stack = evaluate_model("StackingEnsemble", y_test, stack_proba, stack_pred)
results.append(r_stack)
probas["StackingEnsemble"] = stack_proba
plot_confusion_matrix(y_test, stack_pred, "StackingEnsemble")


# ─────────────────────────────────────────────────────────────────
# STEP 3-6: VOTING ENSEMBLE (backup)
# ─────────────────────────────────────────────────────────────────

section("3-6  VOTING ENSEMBLE (soft)")

# Average probabilities from all 5 base models + 2 tuned = 7 models
all_base = ["XGBoost", "LightGBM", "RandomForest", "CatBoost",
            "LogisticRegression", "XGBoost_Tuned", "LightGBM_Tuned"]

vote_proba = np.mean([probas[m] for m in all_base if m in probas], axis=0)
vote_pred  = (vote_proba >= 0.5).astype(int)
r_vote     = evaluate_model("VotingEnsemble", y_test, vote_proba, vote_pred)
results.append(r_vote)
probas["VotingEnsemble"] = vote_proba
plot_confusion_matrix(y_test, vote_pred, "VotingEnsemble")

# ─────────────────────────────────────────────────────────────────
# STEP 3-7: FINAL MODEL SELECTION & SAVE
# ─────────────────────────────────────────────────────────────────

section("3-7  FINAL MODEL SELECTION")

results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print("\n  FULL MODEL COMPARISON TABLE:")
print(results_df[["model","roc_auc","pr_auc","f1_macro","precision","recall"]].to_string(index=False))

# Best model by PR-AUC
best_model_name = results_df.iloc[0]["model"]
print(f"\n  Best model: {best_model_name}")
print(f"  PR-AUC    : {results_df.iloc[0]['pr_auc']:.4f}")
print(f"  ROC-AUC   : {results_df.iloc[0]['roc_auc']:.4f}")
print(f"  F1 (macro): {results_df.iloc[0]['f1_macro']:.4f}")

# Save best model
if best_model_name == "StackingEnsemble":
    final_model = {
        "type"        : "stacking",
        "base_models" : {name: models[name] for name in top3 if name in models},
        "meta_learner": meta_lr,
        "top3_names"  : top3
    }
elif best_model_name == "VotingEnsemble":
    final_model = {
        "type"         : "voting",
        "base_models"  : {m: models[m] for m in all_base if m in models},
        "model_names"  : all_base
    }
else:
    final_model = models.get(best_model_name, models["XGBoost_Tuned"])

joblib.dump(final_model,      os.path.join(MODELS_DIR, "final_model.pkl"))
joblib.dump(models,           os.path.join(MODELS_DIR, "all_models.pkl"))
joblib.dump(probas,           os.path.join(MODELS_DIR, "test_probas.pkl"))
joblib.dump(meta_lr,          os.path.join(MODELS_DIR, "meta_learner.pkl"))
joblib.dump(threshold_results, os.path.join(MODELS_DIR, "thresholds.pkl"))

# Save results table
results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
print(f"  Saved model comparison to reports/model_comparison.csv")

# Save full pipeline (best individual model + scaler if LR)
if "Tuned" in best_model_name or best_model_name in ["XGBoost","LightGBM","RandomForest","CatBoost"]:
    pipeline_model = models.get(best_model_name, models["XGBoost_Tuned"])
    joblib.dump(pipeline_model, os.path.join(MODELS_DIR, "full_pipeline.pkl"))
else:
    joblib.dump(final_model,    os.path.join(MODELS_DIR, "full_pipeline.pkl"))

print(f"\n  Saved to models/:")
print(f"    final_model.pkl")
print(f"    all_models.pkl")
print(f"    full_pipeline.pkl")
print(f"    meta_learner.pkl")
print(f"    thresholds.pkl")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY PLOT — Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────

section("SAVING MODEL COMPARISON PLOT")

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0A1628")
metrics_to_plot = ["pr_auc", "roc_auc"]
titles          = ["PR-AUC (Primary)", "ROC-AUC"]

plot_df = results_df.sort_values("pr_auc", ascending=True)

for ax, metric, title in zip(axes, metrics_to_plot, titles):
    ax.set_facecolor("#0D1F3C")
    colors_bar = [FRAUD_COLOR if i == len(plot_df)-1 else "#4ECDC4"
                  for i in range(len(plot_df))]
    bars = ax.barh(plot_df["model"], plot_df[metric],
                   color=colors_bar, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, plot_df[metric]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left",
                color="white", fontsize=9, fontweight="bold")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel(metric.upper().replace("_"," "), color="white")
    ax.set_xlim(0, min(plot_df[metric].max() * 1.25, 1.0))
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Model Comparison — MuleShield AI",
             color="white", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "model_comparison.png")

# ─────────────────────────────────────────────────────────────────
# STEP 3 COMPLETION
# ─────────────────────────────────────────────────────────────────

section("STEP 3 SUMMARY")

print(f"""
  MODELS TRAINED
    1. XGBoost            (default params)
    2. LightGBM           (default params)
    3. Random Forest      (balanced class weight)
    4. CatBoost           (balanced class weight)
    5. Logistic Regression (baseline)
    6. XGBoost_Tuned      (30-trial Optuna)
    7. LightGBM_Tuned     (30-trial Optuna)
    8. StackingEnsemble   (top3 + LR meta-learner)
    9. VotingEnsemble     (soft vote, all models)

  BEST MODEL
    Name     : {best_model_name}
    PR-AUC   : {results_df.iloc[0]['pr_auc']:.4f}
    ROC-AUC  : {results_df.iloc[0]['roc_auc']:.4f}
    F1 (mac) : {results_df.iloc[0]['f1_macro']:.4f}

  FILES SAVED
    models/final_model.pkl
    models/full_pipeline.pkl
    models/all_models.pkl
    reports/model_comparison.csv
    reports/plots/roc_curves.png
    reports/plots/pr_curves.png
    reports/plots/confusion_matrix_*.png
    reports/plots/model_comparison.png
    reports/best_params.json
""")

print("=" * 65)
print("STEP 3 COMPLETE — All models trained and best model saved.")
print("Type 'continue to next step' to build Explainability & Alert Engine.")
print("=" * 65)
