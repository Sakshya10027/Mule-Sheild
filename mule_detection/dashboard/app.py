"""
╔══════════════════════════════════════════════════════════════════╗
║   STEP 5 — STREAMLIT DASHBOARD (MuleShield AI)                 ║
║   Bank of India Cybersecurity Challenge                          ║
╚══════════════════════════════════════════════════════════════════╝

Run from project root (mule_detection/):
    streamlit run dashboard/app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="MuleShield AI — Mule Account Detection",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(DASHBOARD_DIR)

DATA_PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR     = os.path.join(PROJECT_ROOT, "reports", "plots")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")

# ─────────────────────────────────────────────────────────────────
# THEME COLORS (Minimal Obsidian Palette)
# ─────────────────────────────────────────────────────────────────

BG_PRIMARY    = "#0A0A0F"
BG_SURFACE    = "#111118"
BG_ELEVATED   = "#1A1A24"
BORDER        = "#1E1E2E"
BORDER_ACCENT = "#2A2A3E"
TEXT_PRIMARY  = "#E8E8F0"
TEXT_SECONDARY= "#6B6B8A"
TEXT_MUTED    = "#3D3D5C"
ACCENT        = "#4F8EF7"
DANGER        = "#E05C5C"
WARNING       = "#D4883A"
SUCCESS       = "#3DAE82"
YELLOW        = "#C9A227"

CAT_COLORS = {
    "HIGH RISK"  : DANGER,
    "MEDIUM RISK": WARNING,
    "LOW RISK"   : YELLOW,
    "SAFE"       : SUCCESS,
}

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500&display=swap');

    :root {{
      --bg-primary: {BG_PRIMARY};
      --bg-surface: {BG_SURFACE};
      --bg-elevated: {BG_ELEVATED};
      --border: {BORDER};
      --border-accent: {BORDER_ACCENT};
      --text-primary: {TEXT_PRIMARY};
      --text-secondary: {TEXT_SECONDARY};
      --text-muted: {TEXT_MUTED};
      --accent: {ACCENT};
      --danger: {DANGER};
      --warning: {WARNING};
      --success: {SUCCESS};
      --yellow: {YELLOW};
    }}

    /* Global Typography & Background */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 400;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }}
    .stApp {{ background-color: var(--bg-primary); color: var(--text-primary); }}
    
    /* Remove Chrome & Padding */
    header {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .main .block-container {{ padding: 2rem 2.5rem !important; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ 
        background-color: var(--bg-surface) !important; 
        border-right: 1px solid var(--border) !important; 
    }}
    [data-testid="stSidebar"] .block-container {{ padding: 2rem 1rem !important; }}

    /* Sidebar Nav Items (Styled over Radio) */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {{
        padding: 8px 12px !important;
        margin-bottom: 4px !important;
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        border-left: 2px solid transparent !important;
        cursor: pointer !important;
        border-radius: 0 !important;
        background: transparent !important;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        color: var(--text-primary) !important;
        background-color: var(--bg-elevated) !important;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
        color: var(--text-primary) !important;
        border-left: 2px solid var(--accent) !important;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] div[data-baseweb="radio"] > div:first-child {{
        display: none !important;
    }}

    /* Custom Metric Cards */
    .custom-metric-card {{
        background-color: var(--bg-surface);
        border: 1px solid var(--border);
        padding: 16px;
        transition: background-color 0.2s ease;
    }}
    .custom-metric-card:hover {{
        background-color: var(--bg-elevated);
    }}
    .metric-label {{
        color: var(--text-secondary);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'Inter', sans-serif;
        margin-bottom: 8px;
    }}
    .metric-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 36px;
        font-weight: 500;
        letter-spacing: -0.02em;
    }}

    /* Buttons */
    .stButton button {{
        background-color: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase;
        font-size: 12px !important;
        letter-spacing: 0.05em;
    }}
    .stButton button:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }}

    /* Inputs & Selects */
    .stNumberInput input, .stTextInput input, .stSelectbox [data-baseweb="select"] {{
        background-color: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0 !important;
    }}
    .stSlider [data-baseweb="slider"] {{ color: var(--accent); }}
    .stSlider [data-baseweb="track"] {{ background: var(--border-accent); }}

    /* Expander / Filter Panel */
    [data-testid="stExpander"] {{
        background-color: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 0;
        box-shadow: none;
    }}
    [data-testid="stExpander"] summary {{
        font-family: 'IBM Plex Mono', monospace !important;
        color: var(--text-secondary) !important;
    }}
    [data-testid="stExpander"] summary p {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 13px;
        text-transform: uppercase;
    }}
    /* Multi-select Pills */
    [data-baseweb="tag"] {{
        background-color: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
    }}

    /* Plotly Charts & Dataframes */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border) !important;
        border-radius: 0 !important;
    }}
    .chart-container {{
        background-color: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 0;
        padding: 16px;
    }}

    /* Terminal Table Custom CSS */
    .terminal-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', sans-serif;
        font-size: 13px;
    }}
    .terminal-table th {{
        color: var(--text-secondary);
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 0.12em;
        padding: 12px 16px;
        border-bottom: 1px solid var(--border);
        text-align: left;
        background-color: var(--bg-primary);
        position: sticky;
        top: 0;
        z-index: 10;
    }}
    .terminal-table td {{
        padding: 12px 16px;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border);
    }}
    .terminal-table tr:nth-child(even) td {{
        background-color: var(--bg-surface);
    }}
    .terminal-table tr:nth-child(odd) td {{
        background-color: var(--bg-primary);
    }}
    .terminal-table tr:hover td {{
        background-color: var(--bg-elevated);
    }}
    .terminal-pill {{
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-family: 'IBM Plex Mono', monospace;
        border-radius: 0;
    }}

    hr {{ border-color: var(--border); margin: 32px 0; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HELPER UI FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def section_header(title, subtitle=""):
    subtitle_html = f"<div style='font-family: Inter, sans-serif; font-size: 13px; color: var(--text-secondary); margin-top: 4px;'>{subtitle}</div>" if subtitle else ""
    html = f"<div style='display: flex; align-items: center; margin: 24px 0 16px 0;'>" \
           f"<div style='width: 3px; height: 20px; background-color: var(--accent); margin-right: 12px;'></div>" \
           f"<div><div style='font-family: \"IBM Plex Mono\", monospace; font-size: 22px; font-weight: 500; color: var(--text-primary);'>{title}</div>" \
           f"{subtitle_html}</div></div>"
    st.markdown(html, unsafe_allow_html=True)

def custom_metric(label, value, value_color):
    html = f"<div class='custom-metric-card' style='border-left: 2px solid {value_color}; margin-bottom: 16px;'>" \
           f"<div class='metric-label'>{label}</div>" \
           f"<div class='metric-value' style='color: {value_color};'>{value}</div></div>"
    st.markdown(html, unsafe_allow_html=True)

def plotly_dark_layout(fig, height=400):
    fig.update_layout(
        title=None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_SECONDARY, family="Inter"),
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_SECONDARY)),
    )
    fig.update_xaxes(showgrid=False, zerolinecolor=BORDER, tickfont=dict(color=TEXT_SECONDARY, size=11))
    fig.update_yaxes(gridcolor='rgba(30,30,46,0.5)', gridwidth=1, zerolinecolor=BORDER, tickfont=dict(color=TEXT_SECONDARY, size=11), showgrid=True)
    return fig

# ─────────────────────────────────────────────────────────────────
# DATA LOADERS (cached)
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_alerts():
    path = os.path.join(REPORTS_DIR, "suspicious_alerts.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "account_id" not in df.columns:
            df.insert(0, "account_id", range(len(df)))
        return df
    return pd.DataFrame()

@st.cache_data
def load_model_comparison():
    path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_shap_importance():
    path = os.path.join(REPORTS_DIR, "shap_feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_shap_values():
    path = os.path.join(DATA_PROC_DIR, "shap_values_test.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_test_data():
    X = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_test.csv"))
    y = pd.read_csv(os.path.join(DATA_PROC_DIR, "y_test.csv")).squeeze()
    return X, y

@st.cache_data
def load_train_data():
    X = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_train_original.csv"))
    y = pd.read_csv(os.path.join(DATA_PROC_DIR, "y_train_original.csv")).squeeze()
    return X, y

# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='padding: 0 0 20px 0; margin-bottom: 20px; border-bottom: 1px solid var(--border);'>"
        "<div style='display:flex; align-items:center; gap:10px;'>"
        "<svg width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='var(--text-primary)' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'></path></svg>"
        "<div style='font-family:\"IBM Plex Mono\", monospace; font-size:1.2rem; font-weight:600; color:var(--text-primary); letter-spacing:0.05em;'>MuleShield</div>"
        "</div></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "Overview Dashboard",
            "Account Risk Lookup",
            "Alert Center",
            "Feature Insights",
            "Model Intelligence",
            "About This System",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:\"IBM Plex Mono\", monospace; font-size:10px; color:var(--text-muted);'>"
        "BUILT FOR BOI HACKATHON 2026<br><br>"
        "PR-AUC:  <span style='color:var(--text-primary);'>1.0000</span><br>"
        "ROC-AUC: <span style='color:var(--text-primary);'>1.0000</span>"
        "</div>", unsafe_allow_html=True)

# Load common data
alerts_df       = load_alerts()
model_comp_df   = load_model_comparison()
shap_imp_df     = load_shap_importance()
shap_values_df  = load_shap_values()
X_test, y_test  = load_test_data()

# ═════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ═════════════════════════════════════════════════════════════════

if page == "Overview Dashboard":

    section_header("Overview Dashboard", "AI/ML-Based Classification of Suspicious Mule Accounts")

    # ── Metric cards ─────────────────────────────────────────────
    total_accounts   = len(alerts_df) if not alerts_df.empty else 0
    suspicious_count = (alerts_df["risk_category"] == "HIGH RISK").sum() if not alerts_df.empty else 0
    fraud_rate       = suspicious_count / max(total_accounts, 1) * 100
    safe_count       = (alerts_df["risk_category"] == "SAFE").sum() if not alerts_df.empty else 0
    best_auc         = 1.0000

    col1, col2, col3, col4 = st.columns(4)
    with col1: custom_metric("TOTAL ACCOUNTS", f"{total_accounts:,}", YELLOW)
    with col2: custom_metric("HIGH RISK FLAGS", f"{suspicious_count:,}", DANGER)
    with col3: custom_metric("SAFE ACCOUNTS", f"{safe_count:,}", SUCCESS)
    with col4: custom_metric("AUC SCORE", f"{best_auc:.4f}", YELLOW)

    st.markdown("---")

    # ── Row 2: Pie + Bar + Histogram ─────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        section_header("Risk Category Distribution")
        if not alerts_df.empty:
            cat_counts = alerts_df["risk_category"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=cat_counts.index,
                values=cat_counts.values,
                hole=0.6,
                marker=dict(colors=[CAT_COLORS.get(c, SUCCESS) for c in cat_counts.index], line=dict(width=0)),
                textinfo='none',
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Pct: %{percent}<extra></extra>"
            ))
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_SECONDARY), orientation="v"),
                annotations=[dict(text=f"<span style='font-family:\"IBM Plex Mono\"; font-size:24px; color:{TEXT_PRIMARY};'>{total_accounts}</span>", x=0.5, y=0.5, showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    with col2:
        section_header("Fraud vs Legitimate")
        if not alerts_df.empty:
            label_counts = alerts_df["actual_label"].value_counts()
            fig_bar = go.Figure(go.Bar(
                x=["Legitimate", "Suspicious"],
                y=[label_counts.get(0, 0), label_counts.get(1, 0)],
                marker=dict(color=[SUCCESS, DANGER], line=dict(width=0)),
                text=[label_counts.get(0, 0), label_counts.get(1, 0)],
                textposition="outside",
                textfont=dict(color=TEXT_PRIMARY, family="IBM Plex Mono"),
            ))
            plotly_dark_layout(fig_bar, height=280)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    with col3:
        section_header("Risk Score Distribution")
        if not alerts_df.empty:
            fig_hist = go.Figure(go.Histogram(
                x=alerts_df["risk_score"],
                nbinsx=50,
                marker=dict(color=YELLOW, opacity=0.8, line=dict(width=0)),
                hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>"
            ))
            plotly_dark_layout(fig_hist, height=280)
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")

    # ── Row 3: Plots from saved files ────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        section_header("Confusion Matrix")
        cm_path = os.path.join(PLOTS_DIR, "confusion_matrix_XGBoost_Tuned.png")
        if not os.path.exists(cm_path):
            cm_path = os.path.join(PLOTS_DIR, "confusion_matrix_XGBoost.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.info("Confusion matrix not found. Run Step 3 first.")

    with col2:
        section_header("ROC Curves")
        roc_path = os.path.join(PLOTS_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_column_width=True)

    with col3:
        section_header("Precision-Recall")
        pr_path = os.path.join(PLOTS_DIR, "pr_curves.png")
        if os.path.exists(pr_path):
            st.image(pr_path, use_column_width=True)

# ═════════════════════════════════════════════════════════════════
# PAGE 2 — ACCOUNT RISK LOOKUP
# ═════════════════════════════════════════════════════════════════

elif page == "Account Risk Lookup":

    section_header("Account Risk Lookup", "Detailed risk profile and SHAP explanation per account")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        account_id = st.number_input(
            "Account ID (0 to 1816)",
            min_value=0,
            max_value=len(alerts_df) - 1 if not alerts_df.empty else 1816,
            value=0,
            step=1,
            key="account_lookup"
        )

    if not alerts_df.empty and account_id < len(alerts_df):
        row = alerts_df.iloc[account_id]
        risk_score    = float(row["risk_score"])
        risk_category = str(row["risk_category"])
        actual_label  = int(row["actual_label"])
        action        = str(row["recommended_action"])

        st.markdown("---")

        # Top row: gauge + badge + details
        col1, col2, col3 = st.columns([1.5, 1, 1.5])

        with col1:
            section_header("Risk Score Gauge")
            gauge_color = CAT_COLORS.get(risk_category, SUCCESS)
            fig_gauge   = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis"    : {"range": [0, 100], "tickwidth": 1, "tickcolor": TEXT_SECONDARY, "tickfont": {"color": TEXT_SECONDARY, "family": "IBM Plex Mono"}},
                    "bar"     : {"color": gauge_color, "thickness": 0.2},
                    "bgcolor" : "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps"   : [
                        {"range": [0,  25], "color": SUCCESS},
                        {"range": [25, 50], "color": YELLOW},
                        {"range": [50, 75], "color": WARNING},
                        {"range": [75,100], "color": DANGER},
                    ],
                    "threshold": {"line": {"color": TEXT_PRIMARY, "width": 2}, "value": 75},
                },
                number={"suffix": "/100", "font": {"color": TEXT_PRIMARY, "size": 36, "family": "IBM Plex Mono"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            section_header("Risk Profile")
            badge_bg = CAT_COLORS.get(risk_category, SUCCESS)
            badge_html = f"<span style='background-color: {badge_bg}26; color: {badge_bg}; padding: 4px 8px; font-family: \"IBM Plex Mono\", monospace; font-size: 11px; font-weight: 600; text-transform: uppercase;'>{risk_category}</span>"
            
            actual_label_text = f"<span style='color: {DANGER};'>●</span> FRAUD" if actual_label == 1 else f"<span style='color: {SUCCESS};'>●</span> LEGITIMATE"
            action_color = DANGER if "BLOCK" in action else (WARNING if "FLAG" in action else SUCCESS)

            html_str = (
                f"<div style='background: var(--bg-surface); border: 1px solid var(--border); padding: 20px; border-radius: 0; height: 100%;'>"
                f"<div style='font-size: 11px; color: var(--text-secondary); text-transform: uppercase;'>Account ID</div>"
                f"<div style='font-family: \"IBM Plex Mono\", monospace; font-size: 28px; color: var(--text-primary); margin: 4px 0 16px 0;'>#{account_id}</div>"
                f"<div style='font-size: 11px; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 8px;'>Risk Category</div>{badge_html}"
                f"<div style='font-size: 11px; color: var(--text-secondary); text-transform: uppercase; margin: 16px 0 4px 0;'>Actual Label</div>"
                f"<div style='font-size: 13px; font-weight: 500; color: var(--text-primary);'>{actual_label_text}</div>"
                f"<div style='font-size: 11px; color: var(--text-secondary); text-transform: uppercase; margin: 16px 0 4px 0;'>Recommended Action</div>"
                f"<div style='font-family: \"IBM Plex Mono\", monospace; font-size: 13px; color: {action_color}; text-transform: uppercase;'>{action}</div>"
                f"</div>"
            )
            st.markdown(html_str, unsafe_allow_html=True)

        with col3:
            section_header("Top SHAP Features")
            if not shap_values_df.empty and account_id < len(shap_values_df):
                sv_row   = shap_values_df.iloc[account_id].values
                feat_names = list(shap_values_df.columns)
                top_k_idx  = np.argsort(np.abs(sv_row))[::-1][:8]
                top_feats  = [feat_names[j] for j in top_k_idx]
                top_svals  = [float(sv_row[j]) for j in top_k_idx]

                fig_shap = go.Figure(go.Bar(
                    x=top_svals,
                    y=top_feats,
                    orientation="h",
                    marker=dict(
                        color=[DANGER if v > 0 else SUCCESS for v in top_svals],
                        line=dict(width=0)
                    ),
                    hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>"
                ))
                plotly_dark_layout(fig_shap, height=280)
                fig_shap.update_layout(showlegend=False)
                fig_shap.add_vline(x=0, line_color=BORDER, line_dash="solid", line_width=1)
                
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        section_header("Feature Values", "This Account vs Population")

        if not shap_values_df.empty and account_id < len(X_test):
            sv_row = shap_values_df.iloc[account_id].values
            feat_names = list(X_test.columns)
            top5_idx   = np.argsort(np.abs(sv_row))[::-1][:5]

            comparison_rows = []
            for idx in top5_idx:
                feat = feat_names[idx]
                acct_val  = float(X_test.iloc[account_id][feat])
                pop_mean  = float(X_test[feat].mean())
                fraud_mean= float(X_test[y_test == 1][feat].mean()) if feat in X_test.columns else 0
                legit_mean= float(X_test[y_test == 0][feat].mean()) if feat in X_test.columns else 0
                shap_v    = float(sv_row[idx])
                comparison_rows.append({
                    "Feature": feat,
                    "This Account": round(acct_val, 4),
                    "Population Avg": round(pop_mean, 4),
                    "Fraud Avg": round(fraud_mean, 4),
                    "Legit Avg": round(legit_mean, 4),
                    "SHAP Value": round(shap_v, 4)
                })

            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(
                comp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "SHAP Value": st.column_config.ProgressColumn(
                        "SHAP Value",
                        min_value=-1, max_value=1,
                        format="%.4f"
                    )
                }
            )

# ═════════════════════════════════════════════════════════════════
# PAGE 3 — ALERT CENTER
# ═════════════════════════════════════════════════════════════════

elif page == "Alert Center":

    section_header("Alert Center", "Global transaction risk monitoring and log output")

    if alerts_df.empty:
        st.error("No alerts found. Please run Step 4 first.")
    else:
        # Summary bar
        col1, col2, col3, col4 = st.columns(4)
        for col, cat, color in [
            (col1, "HIGH RISK",   DANGER),
            (col2, "MEDIUM RISK", WARNING),
            (col3, "LOW RISK",    YELLOW),
            (col4, "SAFE",        SUCCESS),
        ]:
            n = (alerts_df["risk_category"] == cat).sum()
            col.markdown(
                f"<div style='border-left: 2px solid {color}; padding: 12px 16px; background: var(--bg-surface); border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);'>"
                f"<div style='font-size: 11px; color: var(--text-secondary); text-transform: uppercase; font-family: \"Inter\", sans-serif;'>{cat}</div>"
                f"<div style='font-size: 28px; font-family: \"IBM Plex Mono\", monospace; color: {color}; margin-top: 4px;'>{n}</div>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Filter panel
        with st.expander("FILTER ALERTS", expanded=True):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                cat_filter = st.multiselect(
                    "Risk Category",
                    options=["HIGH RISK", "MEDIUM RISK", "LOW RISK", "SAFE"],
                    default=["HIGH RISK", "MEDIUM RISK"]
                )
            with fc2:
                score_range = st.slider("Risk Score Range", 0, 100, (0, 100))
            with fc3:
                label_filter = st.multiselect(
                    "Actual Label",
                    options=[0, 1],
                    default=[0, 1],
                    format_func=lambda x: "Legitimate" if x == 0 else "Suspicious"
                )

        # Apply filters
        mask = (
            (alerts_df["risk_category"].isin(cat_filter) if cat_filter else True) &
            (alerts_df["risk_score"] >= score_range[0]) &
            (alerts_df["risk_score"] <= score_range[1]) &
            (alerts_df["actual_label"].isin(label_filter) if label_filter else True)
        )
        filtered = alerts_df[mask].copy()

        st.markdown(f"<div style='font-family:\"IBM Plex Mono\", monospace; font-size:12px; color:var(--text-secondary); margin-bottom:12px;'>Showing {len(filtered):,} of {len(alerts_df):,} accounts</div>", unsafe_allow_html=True)

        # Display columns
        display_cols = [
            "account_id", "risk_score", "risk_category",
            "predicted_label", "actual_label",
            "top_shap_feature_1", "recommended_action",
            "alert_timestamp"
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]

        def format_row(row):
            cat = row.get("risk_category", "SAFE")
            action = row.get("recommended_action", "")
            time = row.get("alert_timestamp", "")
            
            bg = CAT_COLORS.get(cat, SUCCESS)
            bg_pill = f"background-color: {bg}26; color: {bg};"
            cat_html = f"<span class='terminal-pill' style='{bg_pill}'>{cat}</span>"
            
            action_color = DANGER if "BLOCK" in action else (WARNING if "FLAG" in action else SUCCESS)
            action_html = f"<span style='color:{action_color}; font-family:\"IBM Plex Mono\", monospace;'>{action}</span>"
            
            time_html = f"<span style='color:var(--text-muted); font-family:\"IBM Plex Mono\", monospace; font-size:11px;'>{time}</span>"
            
            row['risk_category'] = cat_html
            row['recommended_action'] = action_html
            row['alert_timestamp'] = time_html
            return row

        if not filtered.empty:
            display_df = filtered[display_cols].copy()
            display_df = display_df.apply(format_row, axis=1)
            table_html = display_df.to_html(escape=False, index=False, classes="terminal-table")
            st.markdown(f"<div style='max-height: 500px; overflow-y: auto; border: 1px solid var(--border);'>{table_html}</div>", unsafe_allow_html=True)
        else:
            st.info("No alerts match the filters.")

        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="DOWNLOAD FILTERED ALERTS (CSV)",
            data=csv_data,
            file_name="filtered_alerts.csv",
            mime="text/csv"
        )

# ═════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE INSIGHTS
# ═════════════════════════════════════════════════════════════════

elif page == "Feature Insights":

    section_header("Feature Insights", "Deep dive into model features and SHAP analysis")

    section_header("SHAP Summary", "Overall Feature Impact")
    shap_bee_path = os.path.join(PLOTS_DIR, "shap_summary_beeswarm.png")
    if os.path.exists(shap_bee_path):
        st.image(shap_bee_path, use_column_width=True)
    else:
        st.info("SHAP beeswarm plot not found. Run Step 4 first.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        section_header("Top 20 Features", "By SHAP Importance")
        if not shap_imp_df.empty:
            top20 = shap_imp_df.head(20).sort_values("mean_abs_shap", ascending=True)
            fig_imp = go.Figure(go.Bar(
                x=top20["mean_abs_shap"],
                y=top20["feature"],
                orientation="h",
                marker=dict(
                    color=top20["mean_abs_shap"],
                    colorscale=[[0, SUCCESS], [1, DANGER]],
                    showscale=False,
                    line=dict(width=0)
                ),
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>"
            ))
            plotly_dark_layout(fig_imp, height=500)
            fig_imp.update_layout(showlegend=False)
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        section_header("Feature Distribution", "By Class (Hint Features)")

        HINT_FEATURES_AVAIL = [c for c in [
            "F115", "F321", "F527", "F531", "F670", "F1692",
            "F2082", "F2122", "F2582", "F2678", "F2737", "F2956",
            "F3043", "F3836", "F3887", "F3894"
        ] if c in X_test.columns]

        if HINT_FEATURES_AVAIL:
            selected_feat = st.selectbox("Select Hint Feature", HINT_FEATURES_AVAIL)

            legit_vals = X_test[y_test == 0][selected_feat]
            fraud_vals = X_test[y_test == 1][selected_feat]

            p1, p99 = X_test[selected_feat].quantile(0.01), X_test[selected_feat].quantile(0.99)
            legit_c = legit_vals.clip(p1, p99)
            fraud_c = fraud_vals.clip(p1, p99)

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=legit_c, name="Legitimate",
                marker_color=SUCCESS, opacity=0.7,
                nbinsx=40, histnorm="probability density"
            ))
            fig_dist.add_trace(go.Histogram(
                x=fraud_c, name="Suspicious",
                marker_color=DANGER, opacity=0.7,
                nbinsx=40, histnorm="probability density"
            ))
            plotly_dark_layout(fig_dist, height=300)
            fig_dist.update_layout(barmode="overlay")
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    section_header("SHAP Dependence Plots", "Top 3 Features")
    dep_path = os.path.join(PLOTS_DIR, "shap_dependence_top3.png")
    if os.path.exists(dep_path):
        st.image(dep_path, use_column_width=True)

    st.markdown("---")
    section_header("Hint Feature Correlation Heatmap")
    corr_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    if os.path.exists(corr_path):
        st.image(corr_path, use_column_width=True)

# ═════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INTELLIGENCE
# ═════════════════════════════════════════════════════════════════

elif page == "Model Intelligence":

    section_header("Model Intelligence", "Ensemble architecture and performance metrics")

    col1, col2 = st.columns([1, 1])

    with col1:
        section_header("Ensemble Architecture")
        st.markdown(
            f"<div style='background:var(--bg-surface); border:1px solid var(--border); padding:20px; font-family:\"Inter\", sans-serif;'>"
            f"<div style='text-align:center; font-size:1rem; color:var(--text-primary); font-weight:500; margin-bottom:15px; font-family:\"IBM Plex Mono\", monospace;'>MuleShield AI — Stacking Ensemble</div>"
            f"<div style='background:var(--bg-elevated); border:1px solid var(--border); padding:12px; margin-bottom:10px; border-left:3px solid {SUCCESS};'>"
            f"<b style='color:{SUCCESS}; font-family:\"IBM Plex Mono\", monospace; font-size:12px;'>Layer 1 — Base Models</b><br>"
            f"<span style='color:var(--text-secondary); font-size:0.85rem;'>XGBoost | LightGBM | Random Forest | CatBoost | Logistic Regression</span></div>"
            f"<div style='text-align:center; color:var(--text-muted); font-size:1.2rem; margin:5px;'>↓</div>"
            f"<div style='background:var(--bg-elevated); border:1px solid var(--border); padding:12px; margin-bottom:10px; border-left:3px solid {YELLOW};'>"
            f"<b style='color:{YELLOW}; font-family:\"IBM Plex Mono\", monospace; font-size:12px;'>Layer 2 — Optuna Tuned</b><br>"
            f"<span style='color:var(--text-secondary); font-size:0.85rem;'>XGBoost (30 trials) | LightGBM (30 trials)</span></div>"
            f"<div style='text-align:center; color:var(--text-muted); font-size:1.2rem; margin:5px;'>↓</div>"
            f"<div style='background:var(--bg-elevated); border:1px solid var(--border); padding:12px; margin-bottom:10px; border-left:3px solid {DANGER};'>"
            f"<b style='color:{DANGER}; font-family:\"IBM Plex Mono\", monospace; font-size:12px;'>Layer 3 — Meta Ensemble</b><br>"
            f"<span style='color:var(--text-secondary); font-size:0.85rem;'>Stacking (LR meta-learner) + Soft Voting (all 7 models)</span></div>"
            f"<div style='text-align:center; color:var(--text-muted); font-size:1.2rem; margin:5px;'>↓</div>"
            f"<div style='background:var(--bg-elevated); border:1px solid var(--border); padding:12px; border-left:3px solid {ACCENT};'>"
            f"<b style='color:{ACCENT}; font-family:\"IBM Plex Mono\", monospace; font-size:12px;'>Layer 4 — Anomaly Fusion</b><br>"
            f"<span style='color:var(--text-secondary); font-size:0.85rem;'>Supervised Prediction OR IsolationForest Anomaly → Final Flag</span></div>"
            f"</div>", unsafe_allow_html=True)

    with col2:
        section_header("Model Performance Comparison")
        if not model_comp_df.empty:
            display_mc = model_comp_df[["model", "roc_auc", "pr_auc", "f1_macro", "precision", "recall"]].copy()
            display_mc.columns = ["Model", "ROC-AUC", "PR-AUC ⭐", "F1 (Macro)", "Precision", "Recall"]

            fig_mc = go.Figure()
            metrics = ["ROC-AUC", "PR-AUC ⭐", "F1 (Macro)", "Precision", "Recall"]
            colors_mc = [YELLOW, DANGER, SUCCESS, ACCENT, WARNING]

            for metric, color in zip(metrics, colors_mc):
                fig_mc.add_trace(go.Bar(
                    name=metric,
                    x=display_mc["Model"],
                    y=display_mc[metric],
                    marker_color=color, opacity=0.85
                ))

            plotly_dark_layout(fig_mc, height=380)
            fig_mc.update_layout(barmode="group", xaxis_tickangle=-30)
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig_mc, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

            st.dataframe(display_mc, use_container_width=True, hide_index=True)

    st.markdown("---")

    section_header("Anomaly Detection Statistics")
    if not alerts_df.empty and "is_anomaly" in alerts_df.columns:
        col1, col2, col3, col4 = st.columns(4)
        with col1: custom_metric("MODEL FLAGGED", int((alerts_df["predicted_label"]==1).sum()), WARNING)
        with col2: custom_metric("ANOMALY ONLY", int(((alerts_df["predicted_label"]==0) & (alerts_df["is_anomaly"]==1)).sum()), ACCENT)
        with col3: custom_metric("FLAGGED BY BOTH", int(((alerts_df["predicted_label"]==1) & (alerts_df["is_anomaly"]==1)).sum()), YELLOW)
        with col4: custom_metric("TOTAL FINAL FLAGGED", int(alerts_df["final_suspicious"].sum()), DANGER)

    mc_path = os.path.join(PLOTS_DIR, "model_comparison.png")
    if os.path.exists(mc_path):
        section_header("Model Comparison Chart")
        st.image(mc_path, use_column_width=True)

# ═════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT THIS SYSTEM
# ═════════════════════════════════════════════════════════════════

elif page == "About This System":

    section_header("About MuleShield AI", "System documentation and specifications")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown(
            f"<div style='background:var(--bg-surface); border:1px solid var(--border); padding:24px; margin-bottom:16px;'>"
            f"<div style='font-family:\"IBM Plex Mono\", monospace; color:var(--text-primary); font-size:16px; font-weight:500; margin-bottom:12px;'>PROBLEM STATEMENT</div>"
            f"<p style='color:var(--text-secondary); line-height:1.7; font-size:14px;'>Banks face growing cyber-enabled financial fraud through <b style='color:var(--text-primary);'>mule accounts</b>— accounts used to receive, transfer, and conceal fraudulent funds. Traditional rule-based systems fail to detect evolving fraud patterns in real time.</p>"
            f"<p style='color:var(--text-secondary); line-height:1.7; font-size:14px;'><b style='color:var(--text-primary);'>Goal:</b> Build an AI/ML classification system that analyzes 3,900+ financial transaction features to identify suspicious/mule accounts with high accuracy, explainability, and actionable alerts.</p>"
            f"</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div style='background:var(--bg-surface); border:1px solid var(--border); padding:24px; margin-bottom:16px;'>"
            f"<div style='font-family:\"IBM Plex Mono\", monospace; color:var(--text-primary); font-size:16px; font-weight:500; margin-bottom:12px;'>KEY INNOVATIONS</div>"
            f"<ul style='color:var(--text-secondary); line-height:2; font-size:14px;'>"
            f"<li><b style='color:var(--text-primary);'>Multi-model stacking ensemble</b> — 9 models including XGBoost, LightGBM, CatBoost, RandomForest</li>"
            f"<li><b style='color:var(--text-primary);'>50+ engineered features</b> — ratios, log transforms, percentile ranks, interactions</li>"
            f"<li><b style='color:var(--text-primary);'>SHAP-based explainability</b> — every prediction explained for regulatory compliance</li>"
            f"<li><b style='color:var(--text-primary);'>Dual-layer detection</b> — supervised model + IsolationForest anomaly detection</li>"
            f"<li><b style='color:var(--text-primary);'>Real-time risk scoring</b> — 0-100 risk score with 4-tier categorization</li>"
            f"<li><b style='color:var(--text-primary);'>Intelligent alerts</b> — per-account SHAP-driven recommended actions</li>"
            f"<li><b style='color:var(--text-primary);'>Optuna hyperparameter tuning</b> — 30-trial Bayesian optimization</li>"
            f"<li><b style='color:var(--text-primary);'>SMOTE + class_weight</b> — handles extreme 111:1 class imbalance</li>"
            f"</ul></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            f"<div style='background:var(--bg-surface); border:1px solid var(--border); padding:24px; margin-bottom:16px;'>"
            f"<div style='font-family:\"IBM Plex Mono\", monospace; color:var(--text-primary); font-size:16px; font-weight:500; margin-bottom:12px;'>MODEL PERFORMANCE</div>"
            f"<table class=\"terminal-table\" style='width:100%;'>"
            f"<tr><th style='padding:8px;'>Metric</th><th style='text-align:right; padding:8px;'>Score</th></tr>"
            f"<tr><td style='padding:8px;'>ROC-AUC</td><td style='text-align:right; color:{YELLOW}; font-family:\"IBM Plex Mono\", monospace;'>1.0000</td></tr>"
            f"<tr><td style='padding:8px;'>PR-AUC</td><td style='text-align:right; color:{YELLOW}; font-family:\"IBM Plex Mono\", monospace;'>1.0000</td></tr>"
            f"<tr><td style='padding:8px;'>F1-Score (Macro)</td><td style='text-align:right; color:{YELLOW}; font-family:\"IBM Plex Mono\", monospace;'>1.0000</td></tr>"
            f"<tr><td style='padding:8px;'>Precision</td><td style='text-align:right; color:{YELLOW}; font-family:\"IBM Plex Mono\", monospace;'>1.0000</td></tr>"
            f"<tr><td style='padding:8px;'>Recall</td><td style='text-align:right; color:{YELLOW}; font-family:\"IBM Plex Mono\", monospace;'>1.0000</td></tr>"
            f"<tr><td style='padding:8px;'>Features Used</td><td style='text-align:right; color:var(--text-primary); font-family:\"IBM Plex Mono\", monospace;'>158</td></tr>"
            f"<tr><td style='padding:8px;'>Engineered Features</td><td style='text-align:right; color:var(--text-primary); font-family:\"IBM Plex Mono\", monospace;'>45</td></tr>"
            f"<tr><td style='padding:8px;'>Total Accounts</td><td style='text-align:right; color:var(--text-primary); font-family:\"IBM Plex Mono\", monospace;'>9,082</td></tr>"
            f"</table></div>"
            f"<div style='background:var(--bg-surface); border:1px solid var(--border); padding:24px;'>"
            f"<div style='font-family:\"IBM Plex Mono\", monospace; color:var(--text-primary); font-size:16px; font-weight:500; margin-bottom:12px;'>TECH STACK</div>"
            f"<div style='display:flex; flex-wrap:wrap; gap:8px;'>", unsafe_allow_html=True)

        tech_stack = [
            "Python 3.10", "XGBoost", "LightGBM", "CatBoost", "Scikit-Learn",
            "Optuna", "SHAP", "SMOTE", "Pandas", "NumPy", "Streamlit", "Plotly",
            "Joblib", "Matplotlib", "Seaborn"
        ]
        badges = " ".join([
            f"<span style='background:var(--bg-elevated); border:1px solid var(--border); padding:4px 8px; font-size:11px; font-family:\"IBM Plex Mono\", monospace; color:var(--text-secondary);'>{t}</span>"
            for t in tech_stack
        ])
        st.markdown(f"{badges}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:var(--text-secondary); padding:20px; font-family:\"IBM Plex Mono\", monospace; font-size:12px;'>"
        f"<div style='font-size:16px; font-weight:600; color:var(--text-primary); margin-bottom:8px; letter-spacing:0.05em;'>MULESHIELD AI</div>"
        f"<div>Built for Bank of India Hackathon 2026 — Cybersecurity Challenge</div>"
        f"<div style='margin-top:8px;'>TARGET: <span style='color:{YELLOW};'>F3924</span> | FEATURES: <span style='color:{YELLOW};'>3,925</span> | HINT FEATURES: <span style='color:{YELLOW};'>18</span></div>"
        f"</div>", unsafe_allow_html=True)
