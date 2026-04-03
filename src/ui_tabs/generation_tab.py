"""Synthetic data generation tab."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from src.generator import SyntheticProfileGenerator, VAEGenerator, TORCH_AVAILABLE


def clean_data_for_generation(df):
    """Clean and validate data before generation - same as clustering."""
    df = df.copy()
    
    # Convert timestamp to datetime with error handling
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].astype(str).str.strip()
        
        def fix_date(d):
            if isinstance(d, str) and '-' in d:
                parts = d.split('-')
                if len(parts) == 3 and len(parts[2]) == 1:
                    return f"{parts[0]}-{parts[1]}-0{parts[2]}"
            return d
        
        df['timestamp'] = df['timestamp'].apply(fix_date)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    # Convert power_kw to numeric
    if 'power_kw' in df.columns:
        df['power_kw'] = pd.to_numeric(df['power_kw'], errors='coerce')
    
    # Convert customer_id to numeric
    if 'customer_id' in df.columns:
        df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
    
    # Drop rows with NaN in critical columns
    df = df.dropna(subset=['customer_id', 'timestamp', 'power_kw'])
    
    return df


def render_generation_tab():
    """Render synthetic data generation interface with real data input."""
    st.header("🔄 Synthetic Data Generation")

    # Generate mode selection
    if not TORCH_AVAILABLE:
        mode_options = ["📊 Statistical Method", "🎲 Default Profiles"]
        st.warning("⚠️ PyTorch not installed. VAE not available. Install with: `pip install torch`")
    else:
        mode_options = ["📊 Statistical Method", "🧠 VAE Deep Learning", "🎲 Default Profiles"]

    mode = st.radio("Choose generation method:", mode_options)

    if mode in ["📊 Statistical Method", "🧠 VAE Deep Learning"]:
        st.subheader("Upload Real Consumption Data")
        uploaded_file = st.file_uploader(
            "Upload CSV with consumption data", 
            type=["csv", "xlsx"],
            key="gen_uploader"
        )

        if uploaded_file:
            try:
                # Load and clean data
                if uploaded_file.name.endswith('.xlsx'):
                    df_real = pd.read_excel(uploaded_file)
                else:
                    df_real = pd.read_csv(uploaded_file)

                # Auto-map column names
                column_mapping = {
                    'id': 'customer_id',
                    'horodate': 'timestamp',
                    'valeur': 'power_kw',
                    'ID': 'customer_id',
                    'Horodate': 'timestamp',
                    'Valeur': 'power_kw',
                    'HORODATE': 'timestamp',
                }
                df_real = df_real.rename(columns=column_mapping)
                
                st.subheader("📊 Real Data Preview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Raw Records", len(df_real))
                with col2:
                    st.metric("Columns", len(df_real.columns))
                with col3:
                    st.metric("Customers", df_real['customer_id'].nunique() if 'customer_id' in df_real.columns else "N/A")
                with col4:
                    st.metric("Avg Power (kW)", f"{df_real['power_kw'].mean():.2f}" if 'power_kw' in df_real.columns else "N/A")

                st.write(df_real.head(10))

                col_gen_left, col_gen_mid, col_gen_right = st.columns([1.5, 1, 1])
                with col_gen_left:
                    n_synthetic = st.slider("Generate N synthetic profiles:", 10, 500, 100)
                with col_gen_mid:
                    seed = st.number_input("Random Seed", 0, 10000, 42)
                with col_gen_right:
                    if mode == "🧠 VAE Deep Learning":
                        epochs = st.number_input("Training Epochs", 10, 200, 50)

                button_label = f"🚀 Generate ({mode.split()[0]})"
                if st.button(button_label, key="gen_from_real"):
                    with st.spinner(f"Processing real data and generating with {mode}..."):
                        try:
                            # Clean real data
                            df_clean = clean_data_for_generation(df_real)
                            
                            if len(df_clean) == 0:
                                st.error("❌ No valid data after cleaning. Check your column formats.")
                            else:
                                st.info(f"✅ Cleaned: {len(df_real)} → {len(df_clean)} rows")

                                # Choose generation method
                                if mode == "📊 Statistical Method":
                                    gen = SyntheticProfileGenerator(
                                        profile_class="RP",
                                        seed=int(seed)
                                    )
                                    df_synthetic = gen.generate_from_real_data(
                                        n_profiles=n_synthetic,
                                        real_df=df_clean
                                    )
                                    method_name = "Statistical"
                                
                                elif mode == "🧠 VAE Deep Learning":
                                    try:
                                        vae_gen = VAEGenerator(seed=int(seed), device="cpu")
                                        with st.spinner(f"Training VAE ({int(epochs)} epochs)..."):
                                            vae_gen.train_on_data(
                                                df_clean,
                                                epochs=int(epochs),
                                                batch_size=16,
                                                learning_rate=1e-3,
                                                latent_dim=8,
                                            )
                                        with st.spinner("Generating synthetic profiles..."):
                                            df_synthetic = vae_gen.generate(n_profiles=n_synthetic)
                                        method_name = "VAE"
                                    except Exception as e:
                                        st.error(f"❌ VAE Error: {str(e)}")
                                        st.info("Make sure PyTorch is installed: `pip install torch`")
                                        return

                                # Store in session state
                                st.session_state.synthetic_data = df_synthetic
                                st.session_state.real_data_for_generation = df_clean

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Method", method_name)
                                with col2:
                                    st.metric("Profiles", n_synthetic)
                                with col3:
                                    st.metric("Mean Power", f"{df_synthetic['power_kw'].mean():.2f} kW")
                                with col4:
                                    st.metric("Std Dev", f"{df_synthetic['power_kw'].std():.2f} kW")

                                # Visualization
                                st.subheader("📈 Generation Results")
                                
                                # Extract hour from timestamp for visualization
                                df_clean_viz = df_clean.copy()
                                df_clean_viz['timestamp'] = pd.to_datetime(df_clean_viz['timestamp'])
                                df_clean_viz['hour'] = df_clean_viz['timestamp'].dt.hour
                                
                                df_synth_viz = df_synthetic.copy()
                                df_synth_viz['timestamp'] = pd.to_datetime(df_synth_viz['timestamp'])
                                df_synth_viz['hour'] = df_synth_viz['timestamp'].dt.hour
                                
                                hourly_avg_synth = df_synth_viz.groupby("hour")["power_kw"].agg(["mean", "std"])
                                hourly_avg_real = df_clean_viz.groupby("hour")["power_kw"].agg(["mean", "std"])

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=hourly_avg_real.index,
                                    y=hourly_avg_real["mean"],
                                    error_y=dict(type="data", array=hourly_avg_real["std"], visible=True),
                                    name="Real Data",
                                    mode="lines+markers",
                                    line=dict(color="blue", dash="solid"),
                                ))
                                fig.add_trace(go.Scatter(
                                    x=hourly_avg_synth.index,
                                    y=hourly_avg_synth["mean"],
                                    error_y=dict(type="data", array=hourly_avg_synth["std"], visible=True),
                                    name="Synthetic Data",
                                    mode="lines+markers",
                                    line=dict(color="orange", dash="dash"),
                                ))
                                fig.update_layout(
                                    title="Real vs Synthetic Data (Hourly Average)",
                                    xaxis_title="Hour of Day",
                                    yaxis_title="Power (kW)",
                                    hovermode="x unified"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                st.info("✅ **Output Format**: Synthetic data is in clustering-compatible format (customer_id, timestamp, power_kw). You can directly upload it to the 🎯 Clustering tab!")

                                col_dl1, col_dl2 = st.columns(2)
                                with col_dl1:
                                    st.download_button(
                                        label="📥 Download Synthetic Data (CSV)",
                                        data=df_synthetic.to_csv(index=False),
                                        file_name=f"synthetic_data_{n_synthetic}.csv",
                                        mime="text/csv",
                                    )
                                with col_dl2:
                                    st.download_button(
                                        label="📥 Download Real Data (CSV)",
                                        data=df_clean.to_csv(index=False),
                                        file_name=f"real_data_cleaned.csv",
                                        mime="text/csv",
                                    )

                                # Evaluation section
                                st.divider()
                                st.subheader("📊 Data Evaluation & Comparison")

                                col_eval_left, col_eval_right = st.columns(2)

                                with col_eval_left:
                                    st.write("**Real Data Statistics**")
                                    real_stats = {
                                        "Mean": f"{df_clean['power_kw'].mean():.4f} kW",
                                        "Std Dev": f"{df_clean['power_kw'].std():.4f} kW",
                                        "Min": f"{df_clean['power_kw'].min():.4f} kW",
                                        "Max": f"{df_clean['power_kw'].max():.4f} kW",
                                        "Median": f"{df_clean['power_kw'].median():.4f} kW",
                                        "Q1": f"{df_clean['power_kw'].quantile(0.25):.4f} kW",
                                        "Q3": f"{df_clean['power_kw'].quantile(0.75):.4f} kW",
                                    }
                                    st.dataframe(pd.DataFrame(real_stats.items(), columns=["Metric", "Value"]))

                                with col_eval_right:
                                    st.write("**Synthetic Data Statistics**")
                                    synth_stats = {
                                        "Mean": f"{df_synthetic['power_kw'].mean():.4f} kW",
                                        "Std Dev": f"{df_synthetic['power_kw'].std():.4f} kW",
                                        "Min": f"{df_synthetic['power_kw'].min():.4f} kW",
                                        "Max": f"{df_synthetic['power_kw'].max():.4f} kW",
                                        "Median": f"{df_synthetic['power_kw'].median():.4f} kW",
                                        "Q1": f"{df_synthetic['power_kw'].quantile(0.25):.4f} kW",
                                        "Q3": f"{df_synthetic['power_kw'].quantile(0.75):.4f} kW",
                                    }
                                    st.dataframe(pd.DataFrame(synth_stats.items(), columns=["Metric", "Value"]))

                                # Statistical comparison
                                st.write("**Statistical Similarity Metrics**")
                                metrics = gen.calculate_similarity_metrics(
                                    df_synthetic['power_kw'].values,
                                    df_clean['power_kw'].values
                                )

                                col_m1, col_m2, col_m3 = st.columns(3)
                                with col_m1:
                                    st.metric("Mean Difference", f"{metrics['mean_diff']:.4f} kW")
                                with col_m2:
                                    st.metric("Std Difference", f"{metrics['std_diff']:.4f} kW")
                                with col_m3:
                                    ks_quality = "✅ Good" if metrics['ks_pvalue'] > 0.05 else "⚠️ Different"
                                    st.metric("KS p-value", f"{metrics['ks_pvalue']:.4f}", ks_quality)

                                col_m4, col_m5 = st.columns(2)
                                with col_m4:
                                    st.metric("KS Statistic", f"{metrics['ks_statistic']:.4f}")
                                with col_m5:
                                    st.metric("Wasserstein Distance", f"{metrics['wasserstein_distance']:.4f}")

                                # Distribution comparison
                                st.write("**Distribution Comparison**")
                                fig_dist = go.Figure()
                                fig_dist.add_trace(go.Histogram(
                                    x=df_clean['power_kw'],
                                    name="Real Data",
                                    opacity=0.6,
                                    nbinsx=50
                                ))
                                fig_dist.add_trace(go.Histogram(
                                    x=df_synthetic['power_kw'],
                                    name="Synthetic Data",
                                    opacity=0.6,
                                    nbinsx=50
                                ))
                                fig_dist.update_layout(
                                    title="Distribution Comparison",
                                    xaxis_title="Power (kW)",
                                    yaxis_title="Frequency",
                                    barmode="overlay"
                                )
                                st.plotly_chart(fig_dist, use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

            except Exception as e:
                st.error(f"❌ Failed to load file: {str(e)}")

    else:
        # Default mode: Generate with predefined profiles
        st.subheader("Generate with Default Profiles")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            profile_class = st.selectbox("Profile Class", ["RS", "RP"])
        with col2:
            n_profiles = st.slider("Number of Profiles", 1, 1000, 100)
        with col3:
            seed = st.number_input("Seed", 0, 10000, 42)
        with col4:
            frequency = st.selectbox("Sampling (minutes)", [15, 30, 60])

        if st.button("🚀 Generate", key="gen_button_default"):
            with st.spinner("Generating profiles..."):
                gen = SyntheticProfileGenerator(profile_class=profile_class, seed=int(seed))
                df = gen.generate_multiple_profiles(n_profiles=n_profiles)
                st.session_state.synthetic_data = df

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Profiles", n_profiles)
                with col2:
                    st.metric("Mean Power", f"{df['power_kw'].mean():.2f} kW")
                with col3:
                    st.metric("Max Power", f"{df['power_kw'].max():.2f} kW")
                with col4:
                    st.metric("Std Dev", f"{df['power_kw'].std():.2f} kW")

                if n_profiles <= 5:
                    fig = px.line(
                        df,
                        x="hour",
                        y="power_kw",
                        color="customer_id",
                        title="Generated Profiles",
                        labels={"hour": "Hour of Day", "power_kw": "Power (kW)"},
                    )
                else:
                    hourly_avg = df.groupby("hour")["power_kw"].mean()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hourly_avg.index, y=hourly_avg.values, mode="lines+markers", name="Average"))
                    fig.update_layout(title="Aggregate Profile", xaxis_title="Hour", yaxis_title="Power (kW)")

                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    label="📥 Download CSV",
                    data=df.to_csv(index=False),
                    file_name=f"synthetic_{profile_class}_{n_profiles}.csv",
                    mime="text/csv",
                )

    if st.session_state.synthetic_data is not None:
        st.divider()
        st.subheader("📊 Generated Data Preview")
        st.write(st.session_state.synthetic_data.head(20))
