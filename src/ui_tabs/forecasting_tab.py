"""Forecasting tab."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.models_ml import ForecastingModels


def render_forecasting_tab():
    """Render forecasting interface."""
    st.header("🔮 Forecasting")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Model", ["ARIMA", "LSTM"])
    with col2:
        forecast_steps = st.slider("Forecast Steps", 1, 168, 24)

    uploaded_file = st.file_uploader("Upload timeseries CSV", type="csv", key="forecast_upload")

    if uploaded_file or st.session_state.synthetic_data is not None:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = st.session_state.synthetic_data.groupby("hour")["power_kw"].mean().reset_index()

        # Find power column intelligently
        power_cols = [col for col in df.columns if 'power' in col.lower() or 'consumption' in col.lower() or 'valeur' in col.lower()]
        
        if power_cols:
            power_col = power_cols[0]
        elif "power_kw" in df.columns:
            power_col = "power_kw"
        else:
            power_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Extract timeseries values (handle both Series and arrays)
        power_data = df[power_col]
        if isinstance(power_data, np.ndarray):
            timeseries = power_data
        else:
            timeseries = power_data.values

        if st.button("🚀 Forecast", key="forecast_btn"):
            with st.spinner("Generating forecast..."):
                try:
                    if model_type == "ARIMA":
                        results = ForecastingModels.train_arima(
                            timeseries, forecast_steps=forecast_steps
                        )
                    else:
                        results = ForecastingModels.train_lstm(
                            timeseries, forecast_steps=forecast_steps
                        )

                    forecast = results["forecast"]
                    mae = results.get("mae", 0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("MAE", f"{mae:.3f}")
                    with col2:
                        st.metric("RMSE", f"{np.sqrt(mae):.3f}")

                    # Plot
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(y=timeseries[-48:], mode="lines", name="Historical", line=dict(color="blue"))
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=forecast,
                            mode="lines+markers",
                            name="Forecast",
                            line=dict(color="red", dash="dash"),
                        )
                    )
                    fig.update_layout(
                        title=f"{model_type} Forecast",
                        xaxis_title="Time",
                        yaxis_title="Power (kW)",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    forecast_df = pd.DataFrame(
                        {"step": range(1, len(forecast) + 1), "forecast": forecast}
                    )
                    st.download_button(
                        label="📥 Download Forecast",
                        data=forecast_df.to_csv(index=False),
                        file_name=f"forecast_{model_type}.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
