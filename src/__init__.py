"""
Data Science Project - Unified Package
=====================================

Person 1 (Data Architect):
    - clustering: Evleri RS/RP olarak etiketleme
    - clustering_engine: K-means ve PCA
    - data_loader: Veri yükleme
    - features: Davranışsal özellikler

Person 2 (Model Architect):
    - model_prep: Veri hazırlığı ve dengeleme
    - classification: Sınıflandırma modelleri (RS/RP)
    - forecasting: Tahminleme modelleri (Linear, ARIMA, Prophet)
    - evaluator: Model değerlendirme araçları
    - integration_logic: Entegrasyon katmanı
"""

__version__ = "1.0.0"
__author__ = "Person 1 & Person 2"

# Person 1 - Data Preparation
try:
    from .clustering import assign_residence_labels
    from .clustering_engine import reduce_and_cluster
    from .data_loader import load_consumption_data
    from .features import build_behavioral_features
except ImportError:
    pass

# Person 2 - Model Architect
try:
    from .model_prep import DataPreprocessor, create_lag_features
    from .classification import BaseClassifier, LogisticRegressionClassifier, NeuralNetworkClassifier
    from .forecasting import BaseForecaster, LinearForecaster, ARIMAForecaster, ProphetForecaster
    from .evaluator import ClassificationEvaluator, ForecastingEvaluator, ModelComparator
    from .integration_logic import ModelIntegrator, DataPipeline, ResultsFormatter
except ImportError:
    pass

__all__ = [
    # Person 1 - Data Preparation
    'assign_residence_labels',
    'reduce_and_cluster',
    'load_consumption_data',
    'build_behavioral_features',
    
    # Person 2 - Data Preparation
    'DataPreprocessor',
    'create_lag_features',
    
    # Person 2 - Classification
    'BaseClassifier',
    'LogisticRegressionClassifier',
    'NeuralNetworkClassifier',
    
    # Person 2 - Forecasting
    'BaseForecaster',
    'LinearForecaster',
    'ARIMAForecaster',
    'ProphetForecaster',
    
    # Person 2 - Evaluation
    'ClassificationEvaluator',
    'ForecastingEvaluator',
    'ModelComparator',
    
    # Person 2 - Integration
    'ModelIntegrator',
    'DataPipeline',
    'ResultsFormatter',
]
