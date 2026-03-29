"""Classification tab."""

import streamlit as st
import plotly.figure_factory as ff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models_ml import ClassificationModels


def render_classification_tab():
    """Render classification training interface."""
    st.header("🏷️ Classification")

    if st.session_state.features is None:
        st.warning("⚠️ Please extract features in Clustering tab first")
        return

    st.subheader("Model Training")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Model Type", ["Logistic Regression", "Neural Network"])
    with col2:
        test_ratio = st.slider("Test Ratio", 0.1, 0.5, 0.2)

    if st.button("🚀 Train Model", key="train_btn"):
        with st.spinner("Training..."):
            features_df = st.session_state.features

            # Prepare data
            X = features_df.iloc[:, 1:].values
            if "customer_type" in features_df.columns:
                y = (features_df["customer_type"] == "RP").astype(int).values
            else:
                y = np.random.randint(0, 2, len(X))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=42
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            try:
                if model_type == "Logistic Regression":
                    results = ClassificationModels.train_logistic_regression(
                        X_train, y_train, X_test, y_test
                    )
                else:
                    results = ClassificationModels.train_neural_network(
                        X_train, y_train, X_test, y_test
                    )

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{results['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{results['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{results['f1']:.3f}")

                # Confusion matrix
                cm = results["confusion_matrix"]
                fig = ff.create_annotated_heatmap(
                    z=cm,
                    x=["RS", "RP"],
                    y=["RS", "RP"],
                    colorscale="Blues",
                    showscale=True,
                )
                fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
                st.plotly_chart(fig, use_container_width=True)

                st.success(f"✅ Model trained with {model_type}")

            except Exception as e:
                st.error(f"Error: {e}")
