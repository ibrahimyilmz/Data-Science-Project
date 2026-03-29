"""Synthetic data generation tab."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from src.generator import SyntheticProfileGenerator


def render_generation_tab():
    """Render synthetic data generation interface."""
    st.header("🔄 Synthetic Data Generation")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        profile_class = st.selectbox("Profile Class", ["RS", "RP"])
    with col2:
        n_profiles = st.slider("Number of Profiles", 1, 1000, 100)
    with col3:
        seed = st.number_input("Seed", 0, 10000, 42)
    with col4:
        frequency = st.selectbox("Sampling (minutes)", [15, 30, 60])

    if st.button("🚀 Generate", key="gen_button"):
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
        st.subheader("📊 Stored Data")
        st.write(st.session_state.synthetic_data.head())
