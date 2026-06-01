import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import json
from PIL import Image

# Configuration
REPORTS_DIR = "mule_detection/reports/"
PLOTS_DIR = "mule_detection/reports/plots/"
MODELS_DIR = "mule_detection/models/"
DATA_DIR = "mule_detection/data/processed/"

# Page Config
st.set_page_config(layout="wide", page_title="MuleShield AI | Bank of India", page_icon="🛡️")

# Custom CSS for Bank of India Theme
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    [data-testid="stMetricValue"] {
        color: #0A1628 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #4a4a4a !important;
    }
    [data-testid="stMetricDelta"] {
        color: #2ca02c !important;
    }
    .sidebar .sidebar-content {
        background-color: #0A1628;
    }
    h1, h2, h3 {
        color: #0A1628;
    }
    .stButton>button {
        background-color: #FFD700;
        color: #0A1628;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_dashboard_data():
    alerts = pd.read_csv(os.path.join(REPORTS_DIR, "suspicious_alerts.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    return alerts, X_test

def sidebar_navigation():
    st.sidebar.title("🛡️ MuleShield AI")
    st.sidebar.subheader("Bank of India Cyber Security")
    page = st.sidebar.radio("Navigation", [
        "📊 Overview Dashboard", 
        "🔍 Account Risk Lookup", 
        "🚨 Alert Center", 
        "📈 Feature Insights",
        "🤖 Model Intelligence",
        "📖 About This System"
    ], key="nav_radio")
    return page

def page_overview(alerts):
    st.title("📊 Overview Dashboard")
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    total_acc = len(alerts)
    suspicious_acc = (alerts['final_suspicious'] == 1).sum()
    fraud_rate = (suspicious_acc / total_acc) * 100
    
    m1.metric("Total Accounts Analyzed", f"{total_acc:,}")
    m2.metric("Suspicious Detected", f"{suspicious_acc:,}")
    m3.metric("Detection Rate", f"{fraud_rate:.2f}%")
    m4.metric("System Health", "Active", delta="99.9%")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Category Distribution")
        fig_pie = px.pie(alerts, names='risk_category', 
                         color='risk_category',
                         color_discrete_map={
                             'HIGH RISK': '#d62728', 
                             'MEDIUM RISK': '#ff7f0e', 
                             'LOW RISK': '#fdbf6f', 
                             'SAFE': '#2ca02c'
                         },
                         hole=0.4)
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            legend=dict(font=dict(color="white"))
        )
        st.plotly_chart(fig_pie, width='stretch')

    with col2:
        st.subheader("Risk Score Distribution")
        fig_hist = px.histogram(alerts, x='risk_score', nbins=50, 
                               color_discrete_sequence=['#FFD700']) # Use Gold for better visibility
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='gray')
        )
        st.plotly_chart(fig_hist, width='stretch')

    st.subheader("Model Performance Visuals")
    p1, p2, p3 = st.columns(3)
    if os.path.exists(os.path.join(PLOTS_DIR, "confusion_matrix_xgboost.png")):
        p1.image(os.path.join(PLOTS_DIR, "confusion_matrix_xgboost.png"), caption="Confusion Matrix", use_container_width=True)
    if os.path.exists(os.path.join(PLOTS_DIR, "roc_curves.png")):
        p2.image(os.path.join(PLOTS_DIR, "roc_curves.png"), caption="ROC Curves", use_container_width=True)
    if os.path.exists(os.path.join(PLOTS_DIR, "pr_curves.png")):
        p3.image(os.path.join(PLOTS_DIR, "pr_curves.png"), caption="Precision-Recall Curves", use_container_width=True)

def page_lookup(alerts, X_test):
    st.title("🔍 Account Risk Lookup")
    
    acc_id = st.number_input("Enter Account ID (Index)", min_value=0, max_value=len(alerts)-1, value=0)
    
    if st.button("Analyze Account"):
        row = alerts.iloc[acc_id]
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            # Risk Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = row['risk_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Risk Score: {row['risk_category']}"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#0A1628"},
                    'steps': [
                        {'range': [0, 25], 'color': "green"},
                        {'range': [25, 50], 'color': "orange"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="white"
            )
            st.plotly_chart(fig, width='stretch')
            
            st.info(f"**Recommended Action:** {row['recommended_action']}")

        with c2:
            st.subheader("Top Contributing Features (SHAP)")
            features = row['top_3_shap_features'].split(", ")
            values = [float(v) for v in row['top_3_shap_values'].split(", ")]
            
            # Determine bar color based on risk category
            bar_color = '#FFD700' # Default Gold
            if row['risk_category'] == 'HIGH RISK':
                bar_color = '#d62728' # Red
            elif row['risk_category'] == 'MEDIUM RISK':
                bar_color = '#ff7f0e' # Orange
            
            fig_bar = px.bar(x=values, y=features, orientation='h', 
                            labels={'x': 'SHAP Value', 'y': 'Feature'})
            
            fig_bar.update_traces(marker_color=bar_color)
            
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="white",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_bar, width='stretch')

def page_alerts(alerts):
    st.title("🚨 Alert Center")
    
    f1, f2 = st.columns(2)
    risk_filter = f1.multiselect("Filter by Risk Category", options=['HIGH RISK', 'MEDIUM RISK', 'LOW RISK', 'SAFE'], default=['HIGH RISK', 'MEDIUM RISK'])
    score_range = f2.slider("Risk Score Range", 0, 100, (50, 100))
    
    filtered_alerts = alerts[
        (alerts['risk_category'].isin(risk_filter)) & 
        (alerts['risk_score'] >= score_range[0]) & 
        (alerts['risk_score'] <= score_range[1])
    ]
    
    st.write(f"Showing {len(filtered_alerts)} alerts based on filters.")
    
    def color_risk(val):
        color = 'white'
        if val == 'HIGH RISK': color = '#ffcccc'
        elif val == 'MEDIUM RISK': color = '#fff3cd'
        elif val == 'LOW RISK': color = '#ffe5d0'
        return f'background-color: {color}'

    st.dataframe(filtered_alerts.style.map(color_risk, subset=['risk_category']), width='stretch')
    
    st.download_button("Export Alerts to CSV", filtered_alerts.to_csv(index=False), "filtered_alerts.csv", "text/csv")

def page_insights():
    st.title("📈 Feature Insights")
    
    st.subheader("Global Feature Importance (SHAP)")
    if os.path.exists(os.path.join(PLOTS_DIR, "shap_summary_beeswarm.png")):
        st.image(os.path.join(PLOTS_DIR, "shap_summary_beeswarm.png"))
    
    st.subheader("Correlation Analysis")
    if os.path.exists(os.path.join(PLOTS_DIR, "correlation_heatmap.png")):
        st.image(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))

def page_model():
    st.title("🤖 Model Intelligence")
    
    st.info("The system utilizes a **Stacking Ensemble Architecture** with XGBoost, LightGBM, and CatBoost as base learners, meta-optimized by Logistic Regression.")
    
    if os.path.exists(os.path.join(REPORTS_DIR, "best_params.json")):
        with open(os.path.join(REPORTS_DIR, "best_params.json"), 'r') as f:
            params = json.load(f)
        st.subheader("Optimized Hyperparameters")
        st.json(params)

def page_about():
    st.title("📖 About MuleShield AI")
    st.markdown("""
    ### Problem Statement
    Banks face growing cyber-enabled financial fraud through mule accounts used to receive, transfer, and conceal fraudulent funds. Traditional rule-based systems fail to detect evolving fraud patterns in real time.
    
    ### Our Solution
    MuleShield AI is a production-grade classification system designed for the Bank of India. It combines supervised ensemble learning with unsupervised anomaly detection to identify suspicious accounts with high precision.
    
    ### Key Innovations
    - **Multi-model Stacking Ensemble**: High accuracy across diverse fraud patterns.
    - **50+ Engineered Features**: Domain-specific insights from raw transaction data.
    - **Explainable AI (SHAP)**: Provides 'Why' behind every alert for regulatory compliance.
    - **Dual-Layer Detection**: Catches both known and novel fraud behaviors.
    """)

def main():
    alerts, X_test = load_dashboard_data()
    page = sidebar_navigation()
    
    if page == "📊 Overview Dashboard":
        page_overview(alerts)
    elif page == "🔍 Account Risk Lookup":
        page_lookup(alerts, X_test)
    elif page == "🚨 Alert Center":
        page_alerts(alerts)
    elif page == "📈 Feature Insights":
        page_insights()
    elif page == "🤖 Model Intelligence":
        page_model()
    elif page == "📖 About This System":
        page_about()

if __name__ == "__main__":
    main()
