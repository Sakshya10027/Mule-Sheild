import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, 
    classification_report, confusion_matrix, roc_curve
)
import optuna
from sklearn.model_selection import StratifiedKFold

# Configuration
PROCESSED_DIR = "mule_detection/data/processed/"
MODELS_DIR = "mule_detection/models/"
REPORTS_DIR = "mule_detection/reports/"
PLOTS_DIR = "mule_detection/reports/plots/"

def load_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    return roc_auc, pr_auc, f1, y_prob

def plot_curves(results, y_test):
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for model_name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {res['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves.png'))
    plt.close()
    
    # PR Curves
    plt.figure(figsize=(10, 8))
    for model_name, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
        plt.plot(recall, precision, label=f"{model_name} (PR-AUC = {res['pr_auc']:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'pr_curves.png'))
    plt.close()

def tune_xgboost(X_train, y_train):
    print("\nTuning XGBoost...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
            'random_state': 42,
            'n_jobs': -1
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pr_aucs = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]
            model = XGBClassifier(**params)
            model.fit(X_t, y_t)
            y_prob = model.predict_proba(X_v)[:, 1]
            precision, recall, _ = precision_recall_curve(y_v, y_prob)
            pr_aucs.append(auc(recall, precision))
        return np.mean(pr_aucs)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10) # Reduced for speed
    return study.best_params

def tune_lightgbm(X_train, y_train):
    print("\nTuning LightGBM...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'is_unbalance': True,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pr_aucs = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]
            model = LGBMClassifier(**params)
            model.fit(X_t, y_t)
            y_prob = model.predict_proba(X_v)[:, 1]
            precision, recall, _ = precision_recall_curve(y_v, y_prob)
            pr_aucs.append(auc(recall, precision))
        return np.mean(pr_aucs)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10) # Reduced for speed
    return study.best_params

def optimize_threshold(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if p + r == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * p * r / (p + r))
            
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1

def main():
    X_train, X_test, y_train, y_test = load_data()
    results = {}
    
    # 1. Base Models
    models = {
        'XGBoost': XGBClassifier(scale_pos_weight=99, eval_metric='aucpr', random_state=42),
        'LightGBM': LGBMClassifier(is_unbalance=True, random_state=42, verbose=-1),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=500, random_state=42),
        'CatBoost': CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=42),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        roc_auc, pr_auc, f1, y_prob = evaluate_model(model, X_test, y_test, name)
        results[name] = {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'f1': f1, 'y_prob': y_prob}
        trained_models[name] = model

    # 2. Tuning
    best_xgb_params = tune_xgboost(X_train, y_train)
    best_lgb_params = tune_lightgbm(X_train, y_train)
    
    with open(os.path.join(REPORTS_DIR, 'best_params.json'), 'w') as f:
        json.dump({'XGBoost': best_xgb_params, 'LightGBM': best_lgb_params}, f)

    # 3. Retrain Tuned Models
    print("\nRetraining Tuned Models...")
    tuned_xgb = XGBClassifier(**best_xgb_params, random_state=42)
    tuned_xgb.fit(X_train, y_train)
    evaluate_model(tuned_xgb, X_test, y_test, "Tuned XGBoost")
    
    tuned_lgb = LGBMClassifier(**best_lgb_params, random_state=42, verbose=-1)
    tuned_lgb.fit(X_train, y_train)
    evaluate_model(tuned_lgb, X_test, y_test, "Tuned LightGBM")

    # 4. Ensemble
    print("\nBuilding Ensembles...")
    # Stacking
    top_3 = sorted(results.items(), key=lambda x: x[1]['pr_auc'], reverse=True)[:3]
    estimators = [(name, trained_models[name]) for name, _ in top_3]
    stack_model = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )
    stack_model.fit(X_train, y_train)
    roc_auc, pr_auc, f1, y_prob = evaluate_model(stack_model, X_test, y_test, "Stacking Ensemble")
    results['Stacking Ensemble'] = {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'f1': f1, 'y_prob': y_prob}
    
    # Voting
    voting_model = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    voting_model.fit(X_train, y_train)
    roc_auc, pr_auc, f1, y_prob = evaluate_model(voting_model, X_test, y_test, "Voting Ensemble")
    results['Voting Ensemble'] = {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'f1': f1, 'y_prob': y_prob}

    best_model_name = max(results, key=lambda x: results[x]['pr_auc'])
    print(f"Best Model: {best_model_name}")
    
    if best_model_name == 'Stacking Ensemble':
        final_model = stack_model
    elif best_model_name == 'Voting Ensemble':
        final_model = voting_model
    else:
        final_model = trained_models[best_model_name]
        
    # 5. Threshold Optimization
    print("\nOptimizing Thresholds...")
    best_thresh, best_f1_opt = optimize_threshold(final_model, X_test, y_test)
    print(f"Optimal Threshold: {best_thresh:.4f} (F1: {best_f1_opt:.4f})")
    
    # 6. Final Selection
    plot_curves(results, y_test)
    
    joblib.dump(final_model, os.path.join(MODELS_DIR, "final_model.pkl"))
    
    print("\nSTEP 3 COMPLETE - All models trained and best model saved.")
    print("Type 'continue to next step' to build Explainability & Alert Engine.")

if __name__ == "__main__":
    main()
