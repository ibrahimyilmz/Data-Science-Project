"""Info/Help tab."""

import streamlit as st


def render_info_tab():
    """Render help and documentation interface."""
    st.header("ℹ️ Platform Information")

    st.markdown("""
    ### 📊 Energy Analytics Platform
    
    Comprehensive ML system for energy consumption analysis.
    
    ### 🎯 Features
    - Generate synthetic RS/RP profiles
    - Cluster customers by behavior
    - Classify customer types
    - Forecast consumption trends
    """)

    st.divider()
    st.subheader("📖 Workflow Guide")

    with st.expander("🔄 Synthetic Generation"):
        st.write("""
        Generate realistic energy consumption profiles for testing.
        - Select profile type: RS (Standard) or RP (Premium)
        - Configure number of profiles and random seed
        - Download as CSV for further analysis
        """)

    with st.expander("🎯 Clustering"):
        st.write("""
        Analyze and segment customers by consumption patterns.
        - Upload consumption CSV or use generated data
        - Automatic feature extraction (15+ features)
        - K-Means clustering with optional PCA
        - View 2D visualization and statistics
        """)

    with st.expander("🏷️ Classification"):
        st.write("""
        Train models to predict customer type.
        - Requires features from Clustering tab
        - Choose: Logistic Regression or Neural Network
        - View confusion matrix and metrics
        - Accuracy, Precision, Recall, F1-Score
        """)

    with st.expander("🔮 Forecasting"):
        st.write("""
        Predict future energy consumption.
        - Upload timeseries data or use generated profiles
        - Choose: ARIMA (statistical) or LSTM (deep learning)
        - Set forecast horizon (1-168 steps)
        - View predictions and download results
        """)

    st.divider()
    st.subheader("📋 Data Format")

    st.write("Required CSV columns:")
    st.code("""
    timestamp,power_kw,customer_id,energy_kwh
    2024-01-01 00:00:00,0.45,CUST_001,0.225
    2024-01-01 00:30:00,0.52,CUST_001,0.260
    """)

    st.divider()
    st.subheader("🔧 Troubleshooting")

    with st.expander("Data not loading?"):
        st.write("- Check CSV has required columns: timestamp, power_kw, customer_id")
        st.write("- Verify timestamps are ISO 8601 format")
        st.write("- Ensure no missing values in power column")

    with st.expander("Clustering failed?"):
        st.write("- Try with fewer clusters (3-5)")
        st.write("- Ensure 50+ records available")
        st.write("- Check data quality and ranges")

    with st.expander("Classification low accuracy?"):
        st.write("- Use more training data")
        st.write("- Try Neural Network instead of Logistic Regression")
        st.write("- Adjust test/train ratio")

    with st.expander("Forecast not generating?"):
        st.write("- Minimum 50 samples required for ARIMA")
        st.write("- Minimum 100 samples for LSTM")
        st.write("- Check power column is numeric")
