# Data Science Project - Energy Consumption Analysis and Forecasting

---

## 📋 Project Overview

By analyzing energy consumption data from Enedis:
1. **Build a data foundation**: Ingest raw metering data, convert physical units (kW -> kWh), engineer behavioral features, and generate unsupervised RS/RP labels
2. **Classify houses**: Determine whether they are primary (RP) or secondary (RS) residences
3. **Forecast future consumption**: Predict energy consumption 24 hours ahead
4. **Interactive Dashboard**: Provide a Streamlit-based user interface

The workflow starts with a dedicated data preprocessing and unsupervised labeling stage that produces the ground-truth labels consumed by downstream supervised models.

---

## Project Structure

```
Data-Science-Project/
├── src/                          # Source code
│   ├── data_loader.py            # Raw data ingestion and kW to kWh conversion
│   ├── features.py               # Behavioral feature extraction and Fourier analysis
│   ├── clustering_engine.py      # PCA and K-Means implementation
│   ├── clustering.py             # Automated labeling logic based on occupancy patterns
│   ├── model_prep.py            # Data preparation and balancing
│   ├── classification.py        # Classification models (RS/RP)
│   ├── forecasting.py           # Forecasting models
│   ├── evaluator.py             # Model evaluation tools
│   └── integration_logic.py      # Integration layer
├── models/                       # Trained models (pickle/pt)
├── data/                         # Datasets
├── logs/                         # Log files
├── requirements.txt              # Python libraries
└── README.md                     # This file
```

---

## 🛠️ Installation

### 1. Setting up Python Environment

```bash
# Python 3.9+ required
python --version

# Create virtual environment
python -m venv venv

# Activation
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Libraries

```bash
pip install -r requirements.txt
```

**Core Libraries:**
- `numpy`, `pandas`: Data processing
- `scikit-learn`: Machine learning
- `torch`: Deep learning (PyTorch)
- `statsmodels`: Time series (ARIMA)
- `prophet`: Forecasting (Facebook Prophet)
- `joblib`: Model serialization

---

## Modeling and Integration Tasks

### 1. **Model Preparation** (`src/model_prep.py`)

Prepares labeled and engineered data for training.

#### Classes:
- **`DataPreprocessor`**: Data preprocessing
  - `balance_dataset()`: Balance RS/RP in test set
  - `train_test_split_timeseries()`: Time series splitting
  - `normalize_features()`: Feature scaling
  - `handle_missing_values()`: Missing data handling
  - `remove_outliers()`: Outlier removal

#### Usage Example:
```python
from src.model_prep import DataPreprocessor

preprocessor = DataPreprocessor(test_size=0.2, random_state=42)

# Data balancing
df_balanced = preprocessor.balance_dataset(df, label_col='label')

# Train/Test split
X_train, X_test, y_train, y_test, feature_names = preprocessor.train_test_split_timeseries(
    df_balanced,
    time_col='timestamp',
    label_col='label'
)

# Normalization
X_train_norm, X_test_norm = preprocessor.normalize_features(X_train, X_test, method='standard')
```

---

### 2. **Data Preprocessing and Unsupervised Labeling** (`src/data_loader.py`, `src/features.py`, `src/clustering_engine.py`, `src/clustering.py`)

This stage establishes the data foundation of the full pipeline. It transforms raw Enedis measurements into robust RS/RP labels used as ground truth by subsequent supervised classification tasks.

#### 2.1 Data Ingestion and Physical Conversion (`src/data_loader.py`)

- **Raw input schema**: Enedis CSV files commonly include:
  - `id_pdl` (customer identifier)
  - `horodate` (timestamp)
  - `valeur` (power values)
- **Smart column detection**: The loader resolves time/power/id columns from multiple candidate names, allowing ingestion of datasets with varying headers.
- **Physical conversion (kW to kWh)**: For 30-minute intervals, energy is computed by:

$$
E_{kWh} = P_{kW} \times 0.5
$$

This converts instantaneous power values into interval energy before feature engineering.

#### 2.2 Behavioral Feature Engineering (`src/features.py`)

Thousands of 30-minute readings are aggregated into 11 customer-level behavioral features that summarize occupancy and lifestyle signatures.

- **Occupancy signal**:

$$
occupancy\_rate = 1 - \text{low\_consumption\_day\_ratio}
$$

`low_consumption_day_ratio` is derived from a daily-energy threshold of `0.5` kWh to flag potentially uninhabited days.
- **Weekend behavior**: `weekend_weekday_ratio` compares weekend and weekday mean consumption, which helps identify weekend-only holiday homes.
- **Fourier rhythms**:
  - `fft_daily_amp`: strength of the 24-hour cycle
  - `fft_weekly_amp`: strength of the 7-day cycle
  These are extracted with FFT to quantify periodic consumption structure.
- **Statistical profiling**:
  - `mean_daily_kwh` captures consumption scale
  - `std_daily_kwh` captures variability/volatility

#### 2.3 Unsupervised Machine Learning Pipeline (`src/clustering_engine.py` and `src/clustering.py`)

The project constructs RS/RP ground-truth labels through the following steps:

1. **Feature normalization** with `StandardScaler` so each feature has comparable influence.
2. **PCA reduction** to two principal components (`pca_1`, `pca_2`) to improve cluster separability and enable 2D visualization.
3. **K-Means clustering** to group households by behavioral similarity.
4. **Automated labeling logic**:
   - Cluster with lower mean `occupancy_rate` -> `RS` (Secondary Residence)
   - Cluster with higher mean `occupancy_rate` -> `RP` (Primary Residence)

The resulting labels are exported in `labeled_customers.csv` and become the reference targets for the classification models.

---

### 3. **Classification** (`src/classification.py`)

Classify houses as RS (Secondary Residence) or RP (Primary Residence).

#### Classes:
- **`BaseClassifier`** (Abstract): Base classifier interface

- **`LogisticRegressionClassifier`**: Fast baseline model
  - Hyperparameters: `C=1.0, solver='lbfgs', max_iter=1000`
  - Advantage: Fast, interpretable

- **`NeuralNetworkClassifier`**: Deep neural network (PyTorch)
  - Architecture: 128 → 64 → 32 → 2 neurons
  - Dropout: 0.3 (overfitting prevention)
  - Optimizer: Adam (lr=0.001)
  - Epochs: 50-100

#### Usage Example:
```python
from src.classification import LogisticRegressionClassifier, NeuralNetworkClassifier

# Logistic Regression
lr_classifier = LogisticRegressionClassifier(C=1.0, solver='lbfgs', max_iter=1000)
lr_classifier.train(X_train, y_train)
lr_metrics = lr_classifier.evaluate(X_test, y_test)
lr_classifier.save('models/lr_classifier.pkl')

# Neural Network
nn_classifier = NeuralNetworkClassifier(
    input_size=10,
    hidden_sizes=[128, 64, 32],
    dropout_rate=0.3,
    learning_rate=0.001
)
nn_classifier.train(X_train, y_train, epochs=50, batch_size=32, X_val=X_val, y_val=y_val)
nn_classifier.evaluate(X_test, y_test)
nn_classifier.save('models/nn_classifier.pt')

# Predictions
predictions = lr_classifier.predict(X_test)
probabilities = lr_classifier.predict_proba(X_test)  # Probabilities
```

---

### 4. **Forecasting** (`src/forecasting.py`)

Predict future energy consumption.

#### Classes:
- **`BaseForecaster`** (Abstract): Base forecaster interface

- **`LinearForecaster`**: Multivariate linear regression
  - Lookback: 24 hours
  - Polynomial support (optional)

- **`ARIMAForecaster`**: ARIMA (AutoRegressive Integrated Moving Average)
  - Order: (1, 1, 1) [p=1, d=1, q=1]
  - Confidence intervals for prediction

- **`ProphetForecaster`**: Facebook Prophet
  - Trend + seasonality analysis
  - Weekly/daily periodicity capture
  - 95% confidence intervals

#### Time Series Features:
- Lag features: `create_lag_features(data, lags=[1, 7, 24])`
  - Lag-1: One hour history
  - Lag-7: Seven hour history (weekly rhythm)
  - Lag-24: 24 hour history (daily rhythm)

#### Usage Example:
```python
from src.forecasting import LinearForecaster, ARIMAForecaster, ProphetForecaster

# Linear Forecaster
lf = LinearForecaster(lookback=24, degree=1)
lf.fit(y_train)
forecast = lf.forecast(steps=24, last_sequence=y_train[-24:])

# ARIMA
af = ARIMAForecaster(order=(1, 1, 1))
af.fit(y_train)  # pandas Series or numpy array
forecast = af.forecast(steps=24)
lower, upper = af.get_confidence_intervals(steps=24, alpha=0.05)

# Prophet
import pandas as pd
df_prophet = pd.DataFrame({
    'ds': pd.date_range('2024-01-01', periods=len(y_train), freq='h'),
    'y': y_train
})
pf = ProphetForecaster(interval_width=0.95, seasonality_mode='additive')
pf.fit(df_prophet)
forecast, lower_bound, upper_bound = pf.forecast(steps=24, freq='h')

# Save models
lf.save('models/linear_forecaster.pkl')
af.save('models/arima_forecaster.pkl')
pf.save('models/prophet_forecaster.pkl')
```

---

### 5. **Evaluator** (`src/evaluator.py`)

Tools for model performance evaluation.

#### Classes:
- **`ClassificationEvaluator`**: Classification metrics
  - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
  - Visualization: Confusion Matrix, ROC Curve

- **`ForecastingEvaluator`**: Forecasting metrics
  - Metrics: MAE, RMSE, MAPE, R²
  - Visualization: Predictions vs Actuals, Residuals

- **`ModelComparator`**: Model comparison

#### Usage Example:
```python
from src.evaluator import ClassificationEvaluator, ForecastingEvaluator, ModelComparator

# Classification Evaluation
clf_eval = ClassificationEvaluator()
metrics = clf_eval.evaluate(y_test, y_pred, y_pred_proba)

# Visualizations
fig_cm = clf_eval.plot_confusion_matrix(labels=['RS', 'RP'])
fig_roc = clf_eval.plot_roc_curve()

# Report
report = clf_eval.get_classification_report(labels=['RS', 'RP'])
print(report)

# Forecasting Evaluation
fcst_eval = ForecastingEvaluator()
fcst_metrics = fcst_eval.evaluate(y_test, y_pred)

print(f"MAE: {fcst_metrics['mae']:.2f}")
print(f"RMSE: {fcst_metrics['rmse']:.2f}")
print(f"R2: {fcst_metrics['r2']:.4f}")
```

### Model Comparison
```python
comparator = ModelComparator()
comparator.add_result('Model A', metrics_a)
comparator.add_result('Model B', metrics_b)

comparison_df = comparator.compare_forecasting_models()
fig = comparator.plot_model_comparison(model_type='forecasting', metric='mae')
```

---

### 6. **Integration Logic** (`src/integration_logic.py`)

Integration layer that combines data processing, model inference, and result formatting.

#### Classes:
- **`ModelIntegrator`**: Load and run all models
  - Load models
  - Coordinate pipeline
  - Return results

- **`DataPipeline`**: Prepare input data
  - `process_input_data()`: Process labeled data
  - `process_timeseries_for_forecast()`: Prepare time series

- **`ResultsFormatter`**: Format results for Streamlit

#### Usage Example:
```python
from src.integration_logic import ModelIntegrator, DataPipeline, ResultsFormatter

# 1. Initialize Integrator
integrator = ModelIntegrator(models_dir='models')

# 2. Load models
integrator.load_classification_model('models/lr_classifier.pkl', model_type='logistic')
integrator.load_forecasting_model('arima', 'models/arima_forecaster.pkl')
integrator.load_feature_scaler('models/scaler.pkl')

# 3. Process labeled input data
pipeline = DataPipeline()
X_features, y_labels, feature_names = pipeline.process_input_data(df_labeled_input)

# 4. Classify residences
clf_results = integrator.classify_residence(X_features, feature_names)
clf_df = ResultsFormatter.format_classification_results(clf_results)

# 5. Forecast consumption
timeseries = pipeline.process_timeseries_for_forecast(df_timeseries)
fcst_results = integrator.forecast_consumption(timeseries, steps=24, model_name='arima')
fcst_df = ResultsFormatter.format_forecast_results(fcst_results)

# 6. Send to Streamlit for dashboard display
print(clf_df)
print(fcst_df)
```

---

## Data Flow

```
RAW DATA
    ↓
    → Extract features (PCA, Fourier, etc.)
    → Add RS/RP labels
    ↓
LABELED DATA → MODEL TRAINING
                ↓
                1. DataPreprocessor: Balancing + Normalization
                2. Classification: RS/RP Classification
                3. Forecasting: 24-hour ahead forecasting
                4. Evaluator: Performance evaluation
                5. Save models (models/)
                ↓
TRAINED MODELS → STREAMLIT DASHBOARD
                ↓
                → Interactive UI (Tabs)
                → Dynamic Visualization
                → User Input
```

---

## Quick Start

### Minimal Example

```python
# 1. Import libraries
from src.model_prep import DataPreprocessor
from src.classification import LogisticRegressionClassifier
from src.evaluator import ClassificationEvaluator

import numpy as np
import pandas as pd

# 2. Create synthetic data (replace with real project data)
np.random.seed(42)
X_data = np.random.randn(1000, 15)
y_data = np.random.choice(['RS', 'RP'], 1000)

df = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(15)])
df['label'] = y_data

# 3. Prepare data
prep = DataPreprocessor(test_size=0.2, random_state=42)
df_balanced = prep.balance_dataset(df)
X_train, X_test, y_train, y_test, features = prep.train_test_split_timeseries(df_balanced)
X_train, X_test = prep.normalize_features(X_train, X_test)

# 4. Train model
clf = LogisticRegressionClassifier(C=1.0)
clf.train(X_train, y_train)

# 5. Evaluate
evaluator = ClassificationEvaluator()
metrics = evaluator.evaluate(y_test, clf.predict(X_test))

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")

# 6. Save model
clf.save('models/classifier.pkl')
```

---

## Technical Notes

### Hyperparameters (Optimized)

| Model | Parameter | Value | Reason |
|-------|-----------|-------|--------|
| Logistic Regression | C | 1.0 | Default regularization |
| | solver | lbfgs | Suitable for binary classification |
| | max_iter | 1000 | Convergence assurance |
| Neural Network | Hidden Sizes | [128, 64, 32] | Progressive bottleneck |
| | Dropout | 0.3 | Overfitting prevention |
| | Learning Rate | 0.001 | Stable learning |
| | Epochs | 50 | Overfitting balance |
| | Batch Size | 32 | Memory/convergence balance |

| ARIMA | Order | (1,1,1) |
| Prophet | Changepoint Prior | 0.05 | Sensible trend changes |
| | Seasonality Scale | 10 | 
