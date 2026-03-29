"""Clustering tab."""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from src.clustering import extract_consumption_features, perform_clustering, apply_pca_2d
from src.data_loader import load_consumption_data


def render_clustering_tab():
    """Render clustering workflow interface."""
    st.header("🎯 Clustering")

    uploaded_file = st.file_uploader("Upload CSV with consumption data", type=["csv", "xlsx"])

    if uploaded_file or st.session_state.synthetic_data is not None:
        if uploaded_file:
            # Support both CSV and Excel
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            # Auto-map common column names (Excel format support)
            column_mapping = {
                'id': 'customer_id',
                'horodate': 'timestamp',
                'valeur': 'power_kw',
                'ID': 'customer_id',
                'Horodate': 'timestamp',
                'Valeur': 'power_kw',
                'HORODATE': 'timestamp',
            }
            df = df.rename(columns=column_mapping)
            st.session_state.raw_data = df
        else:
            df = st.session_state.synthetic_data

        st.subheader("Data Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Date Range", f"{len(df) // 48} days approx")

        st.write(df.head())
        
        st.info("📋 Supports: CSV or Excel files\n\nColumn names (auto-mapped):\n- `id` → customer_id\n- `horodate` → timestamp\n- `valeur` → power_kw")

        if st.button("⚡ Extract Features", key="extract_btn"):
            with st.spinner("Extracting features..."):
                try:
                    # Validate required columns
                    required_cols = {'customer_id', 'timestamp', 'power_kw'}
                    missing_cols = required_cols - set(df.columns)
                    if missing_cols:
                        st.error(f"❌ Missing columns: {', '.join(missing_cols)}\n\nExpected: customer_id, timestamp, power_kw")
                    else:
                        # Validate data types
                        df_check = df.copy()
                        
                        # Check power_kw is numeric
                        try:
                            pd.to_numeric(df_check['power_kw'], errors='coerce')
                            if pd.to_numeric(df_check['power_kw'], errors='coerce').isna().sum() > len(df_check) * 0.5:
                                st.error("❌ Error: More than 50% of 'power_kw' values are non-numeric. Please check your data.")
                        except:
                            st.error("❌ Error: 'power_kw' column must contain numeric values.")
                        
                        # Try datetime conversion to validate format
                        try:
                            pd.to_datetime(df_check['timestamp'], errors='coerce')
                            if pd.to_datetime(df_check['timestamp'], errors='coerce').isna().sum() > len(df_check) * 0.5:
                                st.error("❌ Error: More than 50% of 'timestamp' values have invalid datetime format.\n\nExpected format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
                        except:
                            st.error("❌ Error: 'timestamp' column has invalid format.")
                        
                        features_df = extract_consumption_features(df)
                        st.session_state.features = features_df
                        st.success(f"✅ Extracted {len(features_df)} customer profiles")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}\n\n💡 Hint: Check that your data has correct column formats:\n- customer_id: numeric\n- timestamp: date/datetime\n- power_kw: numeric")

        if st.session_state.features is not None:
            st.divider()
            st.subheader("Clustering Parameters")

            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            with col2:
                use_pca = st.checkbox("Use PCA", value=True)

            if st.button("🎯 Cluster", key="cluster_btn"):
                with st.spinner("Clustering..."):
                    labels, scaler, kmeans = perform_clustering(
                        st.session_state.features, n_clusters=n_clusters, use_pca=use_pca
                    )
                    st.session_state.clusters = labels

                    X_pca, pca = apply_pca_2d(st.session_state.features)

                    df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
                    df_plot["Cluster"] = labels
                    df_plot["Customer"] = st.session_state.features["customer_id"].values

                    fig = px.scatter(
                        df_plot,
                        x="PC1",
                        y="PC2",
                        color="Cluster",
                        hover_name="Customer",
                        title="Customer Clusters (PCA 2D)",
                        labels={"Cluster": "Cluster ID"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.success(f"Clustered {len(labels)} customers into {n_clusters} clusters")

                    cluster_stats = pd.DataFrame(
                        {"Cluster": range(n_clusters), "Count": np.bincount(labels)}
                    )
                    st.write("Cluster Statistics:")
                    st.dataframe(cluster_stats)

                    csv = st.session_state.features.copy()
                    csv["cluster"] = labels
                    st.download_button(
                        label="📥 Download Results",
                        data=csv.to_csv(index=False),
                        file_name="clustering_results.csv",
                        mime="text/csv",
                    )
