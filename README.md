# 🛡️ MuleShield AI — Suspicious Mule Account Detection System
### Bank of India Hackathon 2026 | AI/ML Cybersecurity Challenge

---

## 📌 Problem Statement

Banks face growing cyber-enabled financial fraud through **mule accounts**  accounts used to receive,
transfer, and conceal fraudulent funds. Traditional rule-based systems fail to detect evolving fraud
patterns in real time. This project builds a production-grade AI/ML system to classify suspicious mule
accounts with high accuracy, full explainability, and intelligent alert generation.

**Target Variable:** `F3924` (1 = Suspicious Mule Account, 0 = Legitimate)

---

## 🚀 Our Solution

- ✅ Analyzed **9,082 financial accounts** with **3,925 features** each
- ✅ Engineered **45+ intelligent features** from 18 bank-provided hint features
- ✅ Trained **9 models** including stacking ensemble with Optuna tuning
- ✅ Achieved **Perfect PR-AUC = 1.0000** across all tree-based models
- ✅ Detected **all 16 real fraud accounts** (100% recall, 100% precision)
- ✅ Generated **SHAP explanations** for every prediction
- ✅ Built a **production-ready Streamlit dashboard** with 6 interactive pages

---

## 💡 Key Innovations

| Innovation | Description |
|---|---|
| 🎯 Multi-model stacking | XGBoost + LightGBM + RandomForest + CatBoost stacked with LR meta-learner |
| ⚙️ 50+ Engineered Features | Ratio features, log transforms, percentile ranks, interaction terms |
| 🔍 SHAP Explainability | Every prediction explained per-account for regulatory compliance |
| 🌀 Dual-layer Detection | Supervised prediction **OR** IsolationForest anomaly → combined flag |
| 📊 Real-time Risk Scoring | 0–100 risk score with 4-tier categorization (HIGH/MEDIUM/LOW/SAFE) |
| 🎛️ Optuna Tuning | 30-trial Bayesian hyperparameter optimization for top 2 models |
| ⚖️ Extreme Imbalance Handling | SMOTE + class_weight='balanced' for 111:1 imbalance ratio |

---

## 🏆 Model Performance

| Model | ROC-AUC | PR-AUC ⭐ | F1 (Macro) | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| LightGBM | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| RandomForest | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| CatBoost | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| LogisticRegression | 1.0 | 1.0 | 0.9567260747032906 | 0.8421052631578947 | 1.0 |
| XGBoost_Tuned | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| LightGBM_Tuned | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| StackingEnsemble | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| VotingEnsemble | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

> **Primary metric: PR-AUC** (Precision-Recall AUC) — most important for fraud detection with extreme imbalance

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA (9082 × 3925)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: EDA & Data Understanding               │
│   Distribution analysis | Correlation | Outlier detection   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           STEP 2: Preprocessing & Feature Engineering       │
│  Drop 1143 cols > 50% null | Median imputation | Winsorize  │
│  45 engineered features | LightGBM feature selection        │
│  SMOTE: 14400 balanced samples | 158 final features         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: Model Building & Ensemble              │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐ ┌────┐    │
│  │ XGBoost  │ │ LightGBM │ │  RFC   │ │CatBoost │ │ LR │    │
│  └────┬─────┘ └────┬─────┘ └───┬────┘ └────┬────┘ └─┬──┘    │
│       └────────────┴───────────┴────────────┴─────────┘     │
│                              │                              │
│                    ┌─────────▼─────────┐                    │
│                    │ Stacking + Voting  │                   │
│                    └─────────┬─────────┘                    │
└──────────────────────────────┼──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           STEP 4: Explainability & Alert Engine             │
│  SHAP TreeExplainer | Risk Score 0-100 | Alert Generation   │
│  IsolationForest anomaly | Combined suspicious flag         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 5: Streamlit Dashboard                    │
│  Overview | Risk Lookup | Alert Center | Feature Insights   │
│  Model Intelligence | About — All with Plotly + Navy theme  │
└─────────────────────────────────────────────────────────────┘
```

---

## ▶️ How To Run

### 1. Install dependencies
```bash
pip install -r mule_detection/requirements.txt
```

### 2. Run EDA (Step 1)
```bash
python mule_detection/step1_eda.py
```

### 3. Preprocessing & Feature Engineering (Step 2)
```bash
python mule_detection/step2_preprocessing.py
```

### 4. Model Building & Ensemble (Step 3)
```bash
python mule_detection/step3_model.py
```

### 5. Explainability & Alerts (Step 4)
```bash
python mule_detection/step4_explainability.py
```

### 6. Launch Dashboard (Step 5)
```bash
streamlit run mule_detection/dashboard/app.py
```
Dashboard opens at: **http://localhost:8501**

### 7. Generate Submission (Step 6)
```bash
python mule_detection/step6_package.py
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| ML Models | XGBoost, LightGBM, CatBoost, Scikit-Learn |
| Optimization | Optuna (Bayesian, 30 trials) |
| Explainability | SHAP (TreeExplainer) |
| Imbalance | imbalanced-learn (SMOTE) |
| Dashboard | Streamlit + Plotly |
| Data | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |
| Persistence | Joblib |

---

## 📁 Project Structure

```
mule_detection/
├── data/
│   ├── raw/                    ← DataSet.csv (original)
│   └── processed/              ← X_train, X_test, y_train, y_test, SHAP values
├── models/
│   ├── final_model.pkl         ← Best model (XGBoost_Tuned)
│   ├── all_models.pkl          ← All 9 trained models
│   ├── full_pipeline.pkl       ← Production pipeline
│   └── meta_learner.pkl        ← Stacking meta-learner
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── explainability.py
│   └── alert_engine.py
├── dashboard/
│   └── app.py                  ← Streamlit 6-page dashboard
├── reports/
│   ├── plots/                  ← All saved charts (15+ plots)
│   ├── selected_features.txt   ← 158 final features
│   ├── best_params.json        ← Optuna best hyperparameters
│   ├── suspicious_alerts.csv   ← Full alert table (1817 accounts)
│   ├── high_risk_alerts.csv    ← HIGH RISK accounts only
│   └── model_comparison.csv    ← All model metrics
├── step1_eda.py
├── step2_preprocessing.py
├── step3_model.py
├── step4_explainability.py
├── step5_dashboard.py          ← (see dashboard/app.py)
├── step6_package.py
├── requirements.txt
└── README.md

submission.csv                  ← Final predictions (project root)
```

---

## 👥 Team

Built for the **Bank of India Cybersecurity Hackathon 2026**

> *"Protecting the financial system through intelligent AI-powered fraud detection"*

---

*MuleShield AI — Because every suspicious account deserves scrutiny* 🛡️
