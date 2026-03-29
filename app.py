"""
Main Streamlit Application - Energy Consumption Analysis & Forecasting Platform.

Lightweight orchestrator for:
- Synthetic data generation (RS/RP profiles)
- Customer clustering and behavioral analysis
- Classification (RS vs RP prediction)
- Forecasting (consumption prediction)

Architecture: Modular UI tabs, session state management, efficient caching.
"""

import streamlit as st
from src.ui_tabs import (
    render_generation_tab,
    render_clustering_tab,
    render_classification_tab,
    render_forecasting_tab,
    render_info_tab,
)


st.set_page_config(
    page_title="Energy Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

if "raw_data" not in st.session_state:
    st.session_state.raw_data = None

if "features" not in st.session_state:
    st.session_state.features = None

if "clusters" not in st.session_state:
    st.session_state.clusters = None

if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None

if "forecast_results" not in st.session_state:
    st.session_state.forecast_results = None


# Header
st.markdown("---")
col_h1, col_h2, col_h3 = st.columns([0.2, 0.6, 0.2])

with col_h2:
    st.title("⚡ Energy Analytics Platform")
    st.caption("Energy consumption analysis, classification, and forecasting system")


# Sidebar navigation
st.sidebar.markdown("### 🗂️ NAVIGATION")
st.sidebar.markdown("---")

pages = {
    "🏠 Home": "home",
    "🔄 Synthetic Generation": "generation",
    "🎯 Clustering": "clustering",
    "🏷️ Classification": "classification",
    "🔮 Forecasting": "forecasting",
    "ℹ️ Help": "info",
}

for page_name, page_key in pages.items():
    if st.sidebar.button(page_name, use_container_width=True, key=f"nav_{page_key}"):
        st.session_state.page = page_name


st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Session Status")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    data_ok = "✅" if st.session_state.raw_data is not None else "❌"
    st.metric("Data", data_ok)
with col_s2:
    features_ok = "✅" if st.session_state.features is not None else "❌"
    st.metric("Features", features_ok)

col_s3, col_s4 = st.sidebar.columns(2)
with col_s3:
    cluster_ok = "✅" if st.session_state.clusters is not None else "❌"
    st.metric("Clustering", cluster_ok)
with col_s4:
    synth_ok = "✅" if st.session_state.synthetic_data is not None else "❌"
    st.metric("Synthetic", synth_ok)


# Main content
if st.session_state.page == "🏠 Home":
    st.markdown("---")
    st.header("🏠 Welcome")

    st.markdown("""
    ### About This Platform
    
    This comprehensive machine learning platform analyzes energy consumption data to:
    
    ✅ **Analyze customer behavior** using statistical and ML methods  
    ✅ **Generate synthetic profiles** matching real consumption patterns  
    ✅ **Classify customers** (RS: Standard, RP: Premium)  
    ✅ **Forecast consumption** with ARIMA and deep learning  
    
    ### Getting Started
    
    1. **Synthetic Data**: Generate realistic RS/RP consumption profiles
    2. **Clustering**: Upload and analyze customer consumption data
    3. **Classification**: Train models to predict customer class
    4. **Forecasting**: Predict future energy consumption
    
    ### Data Requirements
    
    CSV files must include:
    - `timestamp`: Date/time (ISO 8601)
    - `power_kw` or `puissance_kw`: Power in kW
    - `customer_id`: Customer identifier
    - `energy_kwh`: Energy in kWh (optional)
    
    **Note:** 30-minute sampling intervals (Enedis standard)
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔄 Synthetic Data\nGenerate profiles")
        if st.button("Start →", key="start_gen"):
            st.session_state.page = "🔄 Synthetic Generation"
            st.rerun()

    with col2:
        st.markdown("### 🎯 Clustering\nGroup customers")
        if st.button("Start →", key="start_clust"):
            st.session_state.page = "🎯 Clustering"
            st.rerun()

    with col3:
        st.markdown("### 🔮 Forecasting\nPredict consumption")
        if st.button("Start →", key="start_fore"):
            st.session_state.page = "🔮 Forecasting"
            st.rerun()

elif st.session_state.page == "🔄 Synthetic Generation":
    st.markdown("---")
    render_generation_tab()

elif st.session_state.page == "🎯 Clustering":
    st.markdown("---")
    render_clustering_tab()

elif st.session_state.page == "🏷️ Classification":
    st.markdown("---")
    render_classification_tab()

elif st.session_state.page == "🔮 Forecasting":
    st.markdown("---")
    render_forecasting_tab()

elif st.session_state.page == "ℹ️ Help":
    st.markdown("---")
    render_info_tab()


# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("⚡ Energy Analytics Platform v1.0")
with col_f2:
    st.caption("Streamlit • Scikit-Learn • PyTorch")
with col_f3:
    st.caption("© 2026 Data Science Project")
