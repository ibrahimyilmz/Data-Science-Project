# 🏗️ PERSON 2 ARCHITECTURE DIAGRAM

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PERSON 2 (MODEL ARCHITECT)             │
│                                                                   │
│  INPUT: Etiketlenmiş Veri (Person 1)                            │
│  OUTPUT: Eğitilmiş Modeller → Person 3 (Streamlit)             │
└─────────────────────────────────────────────────────────────────┘

                            ↓
                            
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION LAYER                      │
│                     (model_prep.py)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  INPUT: Raw Data (CSV)                                           │
│    └─ features_1..N: Features from Person 1                     │
│    └─ label: RS/RP classification                               │
│    └─ timestamp: (optional) Time series                          │
│                                                                   │
│  PROCESSING:                                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Data Cleaning                                         │  │
│  │    - Missing values (interpolate/forward fill)           │  │
│  │    - Outlier detection (IQR/ZScore)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. Dataset Balancing                                     │  │
│  │    - RS: 1000 → 400 (undersampling)                      │  │
│  │    - RP: 400 → 400 (keep)                                │  │
│  │    - Result: 50/50 split                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. Train/Test Split (Chronological for Time Series)      │  │
│  │    - Training: 80% (sorted by time)                      │  │
│  │    - Testing: 20% (never seen during training)           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. Feature Normalization                                 │  │
│  │    - Method: StandardScaler (mean=0, std=1)              │  │
│  │    - Applied: fit(X_train) → transform(X_train, X_test)  │  │
│  │    - Lag Features: [t-1, t-7, t-24] for forecasting      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  OUTPUT: (X_train_norm, X_test_norm, y_train, y_test)           │
└─────────────────────────────────────────────────────────────────┘

                            ↓
                    
        ┌───────────────────────────────────────┐
        │                                       │
        ↓                                       ↓
        
┌──────────────────────────┐    ┌──────────────────────────┐
│  CLASSIFICATION BRANCH   │    │  FORECASTING BRANCH      │
│  (classification.py)     │    │  (forecasting.py)        │
└──────────────────────────┘    └──────────────────────────┘
        ↓                                       ↓

    MODEL 1: LR         MODEL 2: NN       MODEL 3: Linear
    ┌─────────┐        ┌──────────┐      ┌──────────┐
    │ Logistic│        │ PyTorch  │      │ Multiple │
    │Regress. │        │ NN       │      │ Linear   │
    │         │        │ [128,64] │      │ Regression
    │ C=1.0   │        │ Dropout  │      │ Lookback:24
    │ Fast    │        │ 0.3      │      │
    │ Baseline│        │ Epochs:50│      │
    └────┬────┘        └────┬─────┘      └────┬─────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                    MODEL 4: ARIMA
                    ┌──────────────┐
                    │ Order (1,1,1)│
                    │ Confidence   │
                    │ Intervals    │
                    └────┬─────────┘
                         │
                    MODEL 5: Prophet
                    ┌──────────────┐
                    │ Trend+Season │
                    │ Weekly/Daily │
                    │ 95% CI       │
                    └────┬─────────┘
                         │
                         ↓

            ┌──────────────────────────┐
            │   EVALUATION LAYER       │
            │   (evaluator.py)         │
            ├──────────────────────────┤
            │                          │
            │ CLASSIFICATION METRICS:  │
            │ ├─ Accuracy             │
            │ ├─ Precision/Recall     │
            │ ├─ F1 Score             │
            │ ├─ ROC-AUC              │
            │ └─ Confusion Matrix     │
            │                          │
            │ FORECASTING METRICS:    │
            │ ├─ MAE                  │
            │ ├─ RMSE                 │
            │ ├─ MAPE                 │
            │ └─ R² Score             │
            │                          │
            │ MODEL COMPARISON        │
            │ └─ Performance ranking  │
            │                          │
            └────┬─────────────────────┘
                 │
                 ↓
         
        ┌─────────────────────────┐
        │ SAVE TRAINED MODELS     │
        │                         │
        │ models/                 │
        │ ├─ classification_      │
        │ │  logistic.pkl         │
        │ ├─ classification_      │
        │ │  neural_network.pt    │
        │ ├─ forecasting_         │
        │ │  linear.pkl           │
        │ ├─ forecasting_         │
        │ │  arima.pkl            │
        │ └─ forecasting_         │
        │    prophet.pkl          │
        │                         │
        └────┬────────────────────┘
             │
             ↓

        ┌──────────────────────────┐
        │ INTEGRATION LAYER        │
        │ (integration_logic.py)   │
        ├──────────────────────────┤
        │                          │
        │ ModelIntegrator:         │
        │ • Load models             │
        │ • classify_residence()    │
        │ • forecast_consumption()  │
        │ • get_model_info()        │
        │                          │
        │ DataPipeline:            │
        │ • process_person1_data()  │
        │ • process_timeseries()    │
        │                          │
        │ ResultsFormatter:        │
        │ • Format for Streamlit    │
        │ • DataFrame output        │
        │                          │
        └────┬─────────────────────┘
             │
             ↓

        ┌──────────────────────────┐
        │ OUTPUT: Ready for Person3│
        │                          │
        │ models/ (trained)        │
        │ └─ 5 files ready         │
        │                          │
        │ Integration Interface:   │
        │ • ModelIntegrator        │
        │ • ResultsFormatter       │
        │                          │
        └──────────────────────────┘
```

---

## Data Flow Diagram

```
┌──────────────────┐
│  PERSON 1        │
│  (Data Architect)│
│                  │
│ CSV:             │
│ - features_1..N  │
│ - label (RS/RP)  │
│ - timestamp      │
└────────┬─────────┘
         │
         ↓
    ┌─────────────────┐
    │ DataPreprocessor│
    │ - Balance       │
    │ - Split         │
    │ - Normalize     │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │                 │
    ↓                 ↓
[Classification] [Forecasting]
    │                 │
    │  5 Models       │  3 Models
    │                 │
    └────────┬────────┘
             │
        ┌────┴─────┐
        │ Evaluate  │
        │ Compare   │
        │ Metrics   │
        └────┬─────┘
             │
        ┌────┴──────────────┐
        │  Save Models      │
        │  (models/)        │
        │                   │
        │  training_report  │
        │  (.json)          │
        └────┬──────────────┘
             │
             ↓
      ┌────────────────┐
      │  PERSON 3      │
      │ (Integrator)   │
      │                │
      │ Streamlit      │
      │ Dashboard      │
      │ - Tabs         │
      │ - Charts       │
      │ - User Input   │
      └────────────────┘
```

---

## Class Hierarchy

```
BASE CLASSES (Abstract)
│
├─ BaseClassifier (ABC)
│  └─ `train()`
│  └─ `predict()`
│  └─ `evaluate()`
│  └─ `save()`
│  └─ `load()`
│
├─ BaseForecaster (ABC)
│  └─ `fit()`
│  └─ `forecast()`
│  └─ `evaluate()`
│  └─ `save()`
│  └─ `load()`
│
└─ BaseEvaluator (Mixin)
   └─ `evaluate()`
   └─ `plot_*()`


CONCRETE IMPLEMENTATIONS
│
├─ LogisticRegressionClassifier ← BaseClassifier
│  └─ Scikit-learn wrapper
│  
├─ NeuralNetworkClassifier ← BaseClassifier
│  └─ PyTorch model
│
├─ LinearForecaster ← BaseForecaster
│  └─ Sklearn regression
│
├─ ARIMAForecaster ← BaseForecaster
│  └─ Statsmodels ARIMA
│
└─ ProphetForecaster ← BaseForecaster
   └─ Facebook Prophet


EVALUATORS & UTILITIES
│
├─ ClassificationEvaluator
│  ├─ evaluate()
│  ├─ plot_confusion_matrix()
│  ├─ plot_roc_curve()
│  └─ get_classification_report()
│
├─ ForecastingEvaluator
│  ├─ evaluate()
│  ├─ plot_predictions()
│  ├─ plot_residuals()
│  └─ get_performance_summary()
│
├─ ModelComparator
│  ├─ add_result()
│  ├─ compare_forecasting_models()
│  ├─ compare_classification_models()
│  └─ plot_model_comparison()
│
├─ DataPreprocessor
│  ├─ balance_dataset()
│  ├─ train_test_split_timeseries()
│  ├─ normalize_features()
│  ├─ handle_missing_values()
│  └─ remove_outliers()
│
├─ ModelIntegrator
│  ├─ load_classification_model()
│  ├─ load_forecasting_model()
│  ├─ classify_residence()
│  ├─ forecast_consumption()
│  └─ get_model_info()
│
├─ DataPipeline
│  ├─ process_person1_data()
│  └─ process_timeseries_for_forecast()
│
└─ ResultsFormatter
   ├─ format_classification_results()
   └─ format_forecast_results()
```

---

## Training Pipeline

```
train_all_models.py

Step 1: Load Data
        ↓
Step 2: Create Fake Data (Replace with Person1)
        ↓
Step 3: Data Preprocessing
        ├─ Balance
        ├─ Split
        └─ Normalize
        ↓
Step 4: Train Classification Models
        ├─ Model 1: LogisticRegression
        │  └─ evaluate() → metrics
        │  └─ save() → models/
        │
        └─ Model 2: NeuralNetwork
           └─ evaluate() → metrics
           └─ save() → models/
        ↓
Step 5: Train Forecasting Models
        ├─ Model 3: Linear
        ├─ Model 4: ARIMA
        └─ Model 5: Prophet
           (same pattern as step 4)
        ↓
Step 6: Generate Report
        ├─ training_report.json
        ├─ Model paths
        └─ Performance metrics
        ↓
DONE ✓ Models ready for Person 3
```

---

## Performance Expectations

### Classification (RS vs RP)
```
Model                 Accuracy  Precision  Recall   F1
────────────────────────────────────────────────────────
Logistic Regression    0.75      0.73      0.77    0.75
Neural Network        0.82      0.81      0.83    0.82
────────────────────────────────────────────────────────
Expected (Production) > 0.80
```

### Forecasting (24-hour ahead)
```
Model                 MAE    RMSE   MAPE%   R²
──────────────────────────────────────────────
Linear Forecaster     9.2    11.5   15.2   0.68
ARIMA(1,1,1)         8.1     9.8    13.1   0.75
Prophet              7.5     9.2    12.0   0.78
──────────────────────────────────────────────
Expected (Production) MAE < 8, RMSE < 10, R² > 0.75
```

---

**Generated by Person 2 (Model Architect)**  
**Date**: March 29, 2026  
**Status**: ✅ Complete
