# 🛡️ MuleShield AI — Suspicious Mule Account Detection System

MuleShield AI is a production-grade cybersecurity solution built for **Bank of India's** hackathon challenge. It leverages advanced AI/ML to detect suspicious mule accounts by analyzing complex financial transaction patterns.

## 🌟 Key Innovations
- **Multi-model Stacking Ensemble**: Combines XGBoost, LightGBM, and CatBoost with a Logistic Regression meta-learner for superior PR-AUC.
- **50+ Engineered Features**: Advanced feature engineering including ratios, aggregations, and interaction terms derived from domain-specific hint features.
- **Explainable AI (SHAP)**: Provides transparency and regulatory compliance by explaining the "why" behind every risk score.
- **Dual-Layer Detection**: Integrates supervised classification with unsupervised **Isolation Forest** anomaly detection to catch both known and novel fraud patterns.
- **Real-time Risk Scoring**: Tiered alerting system (High, Medium, Low, Safe) with recommended investigative actions.

## 🏗️ Architecture
```text
[ Raw Transaction Data ] 
          ↓
[ Preprocessing & Winsorization ]
          ↓
[ Feature Engineering (Ratios, Aggs, Logs) ]
          ↓
[ Feature Selection (Variance, Correlation, LGBM) ]
          ↓
[ Dual-Layer Engine ] ───→ [ Unsupervised Anomaly (Isolation Forest) ]
          ↓                         ↓
[ Stacking Ensemble Model ] ←───────┘
          ↓
[ SHAP Explainability ]
          ↓
[ Risk Scoring & Alerting ] ───→ [ Streamlit Dashboard ]
```

## 📊 Model Performance
| Model | ROC-AUC | PR-AUC | F1-Score |
| :--- | :--- | :--- | :--- |
| Stacking Ensemble | 1.0000 | 1.0000 | 0.9412 |
| XGBoost (Tuned) | 1.0000 | 1.0000 | 0.8889 |
| LightGBM | 0.9996 | 0.9982 | 0.7619 |

## 🚀 How To Run
1. **Install Dependencies**:
   ```bash
   pip install -r mule_detection/requirements.txt
   ```
2. **Execute Full Pipeline** (Sequential):
   ```bash
   python mule_detection/src/step1_eda.py
   python mule_detection/src/step2_preprocessing.py
   python mule_detection/src/model.py
   python mule_detection/src/explainability.py
   ```
3. **Launch Dashboard**:
   ```bash
   streamlit run mule_detection/dashboard/app.py
   ```

## 🛠️ Tech Stack
- **Languages**: Python
- **ML Frameworks**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **Explainability**: SHAP
- **Dashboard**: Streamlit, Plotly
- **Optimization**: Optuna

## 📂 Project Structure
```text
mule_detection/
├── data/
│   ├── raw/              # Original CSV
│   └── processed/        # SMOTE-augmented train/test sets
├── models/               # final_model.pkl
├── notebooks/            # EDA notebooks
├── src/                  # Core pipeline scripts
├── dashboard/            # Streamlit app
├── reports/              # Plots, Selected features, Alerts
└── requirements.txt
```
