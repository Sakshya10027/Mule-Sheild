import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime

# Configuration
PROCESSED_DIR = "mule_detection/data/processed/"
MODELS_DIR = "mule_detection/models/"
REPORTS_DIR = "mule_detection/reports/"
PLOTS_DIR = "mule_detection/reports/plots/"

def run_explainability():
    print("Starting Step 4: Explainability & Risk Alert Engine")
    
    # Load Data
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()
    
    # Load Model
    model = joblib.load(os.path.join(MODELS_DIR, "final_model.pkl"))
    
    # 1. SHAP EXPLAINABILITY
    print("\n1. Computing SHAP values...")
    # Use TreeExplainer for tree-based models (XGBoost, LGBM, CatBoost, RF)
    # For Stacking/Voting, we might need KernelExplainer or use a base model for proxy
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except:
        print("Falling back to KernelExplainer proxy or using a base estimator for SHAP...")
        # If ensemble, use the first base estimator or just a sample for speed
        explainer = shap.Explainer(model.predict, X_test.iloc[:100])
        shap_values = explainer(X_test)

    # a) Beeswarm Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary_beeswarm.png"))
    plt.close()

    # b) Bar Importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_bar_importance.png"))
    plt.close()

    # c) Top 10 Features
    if isinstance(shap_values, list): # For some multiclass/output formats
        vals = np.abs(shap_values[1]).mean(0)
    elif hasattr(shap_values, 'values'): # New API
        vals = np.abs(shap_values.values).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)
        
    feature_importance = pd.DataFrame(list(zip(X_test.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    print("\nTop 10 Important Features by SHAP:")
    print(feature_importance.head(10))

    # 2. RISK SCORING & ALERT ENGINE
    print("\n2. Generating Risk Scores and Alerts...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    alerts_df = pd.DataFrame({
        'account_id': X_test.index,
        'actual_label': y_test,
        'predicted_label': y_pred,
        'risk_score': np.round(y_prob * 100, 2)
    })
    
    # Risk Categories
    def assign_category(score):
        if score >= 75: return 'HIGH RISK', 'BLOCK IMMEDIATELY'
        if score >= 50: return 'MEDIUM RISK', 'FLAG FOR REVIEW'
        if score >= 25: return 'LOW RISK', 'MONITOR CLOSELY'
        return 'SAFE', 'NO ACTION NEEDED'
    
    cat_action = alerts_df['risk_score'].apply(assign_category)
    alerts_df['risk_category'] = [x[0] for x in cat_action]
    alerts_df['recommended_action'] = [x[1] for x in cat_action]
    alerts_df['alert_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add SHAP features per account
    top_features = []
    top_values = []
    
    # Get values based on SHAP output type
    s_vals = shap_values.values if hasattr(shap_values, 'values') else shap_values
    if isinstance(s_vals, list): s_vals = s_vals[1]

    for i in range(len(X_test)):
        row_shap = s_vals[i]
        top_indices = np.argsort(np.abs(row_shap))[-3:][::-1]
        top_features.append(", ".join(X_test.columns[top_indices]))
        top_values.append(", ".join([str(round(row_shap[idx], 4)) for idx in top_indices]))
    
    alerts_df['top_3_shap_features'] = top_features
    alerts_df['top_3_shap_values'] = top_values

    # 3. COMBINED ANOMALY FLAG
    # Note: is_anomaly was calculated in Step 2 but not saved in X_test for final features if not selected.
    # We should have kept it. For now, we'll check if it exists in final_features.
    if 'is_anomaly' in X_test.columns:
        alerts_df['is_anomaly'] = X_test['is_anomaly'].values
        alerts_df['final_suspicious'] = ((alerts_df['predicted_label'] == 1) | (alerts_df['is_anomaly'] == 1)).astype(int)
    else:
        alerts_df['final_suspicious'] = alerts_df['predicted_label']

    # 4. SAVE ALERTS
    alerts_df.to_csv(os.path.join(REPORTS_DIR, "suspicious_alerts.csv"), index=False)
    high_risk = alerts_df[alerts_df['risk_category'] == 'HIGH RISK']
    high_risk.to_csv(os.path.join(REPORTS_DIR, "high_risk_alerts.csv"), index=False)
    
    print(f"\n--- Alert Summary ---")
    print(f"Total Accounts Analyzed: {len(alerts_df)}")
    print(alerts_df['risk_category'].value_counts())
    print(f"Final Suspicious Flags (Model + Anomaly): {alerts_df['final_suspicious'].sum()}")
    
    print("\nSTEP 4 COMPLETE - Explainability and Alert Engine ready.")
    print("Type 'continue to next step' to build the Streamlit Dashboard.")

if __name__ == "__main__":
    run_explainability()
