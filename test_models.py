"""
Quick Test Script - Kişi 2 Modüllerinin Temel Testi
===================================================

Bu script tüm modüllerin doğru çalıştığını doğrular.
Python: python test_models.py
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Kütüphaneleri import et
try:
    from src.model_prep import DataPreprocessor, create_lag_features
    from src.classification import LogisticRegressionClassifier, NeuralNetworkClassifier
    from src.forecasting import LinearForecaster, ARIMAForecaster, ProphetForecaster
    from src.evaluator import ClassificationEvaluator, ForecastingEvaluator
    from src.integration_logic import ModelIntegrator, DataPipeline, ResultsFormatter
    print("✓ Tüm modüller başarıyla import edildi")
except ImportError as e:
    print(f"✗ Import hatası: {e}")
    sys.exit(1)


def test_data_preparation():
    """Test: Veri Hazırlığı"""
    print("\n" + "="*60)
    print("TEST 1: DATA PREPARATION")
    print("="*60)
    
    try:
        # Sahte veri oluştur
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'label': np.random.choice(['RS', 'RP'], 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h')
        })
        
        # DataPreprocessor
        preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
        
        # Dataset dengeleme
        df_balanced = preprocessor.balance_dataset(df, label_col='label')
        print(f"✓ Dataset dengelendi: {df_balanced.shape}")
        
        # Train/Test split
        X_train, X_test, y_train, y_test, features = preprocessor.train_test_split_timeseries(
            df_balanced,
            time_col='timestamp',
            label_col='label'
        )
        print(f"✓ Train/Test split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        
        # Normalizasyon
        X_train_norm, X_test_norm = preprocessor.normalize_features(X_train, X_test, method='standard')
        print(f"✓ Normalizasyon uygulandı: Mean={X_train_norm.mean():.4f}, Std={X_train_norm.std():.4f}")
        
        return X_train_norm, X_test_norm, y_train, y_test, True
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        return None, None, None, None, False


def test_classification(X_train, X_test, y_train, y_test):
    """Test: Sınıflandırma"""
    print("\n" + "="*60)
    print("TEST 2: CLASSIFICATION")
    print("="*60)
    
    try:
        # Logistic Regression
        print("\n[Logistic Regression]")
        lr = LogisticRegressionClassifier(C=1.0, solver='lbfgs', max_iter=1000)
        lr.train(X_train, y_train)
        print(f"✓ Model eğitildi")
        
        lr_pred = lr.predict(X_test)
        print(f"✓ Tahmin yapıldı: {lr_pred[:5]}")
        
        lr_proba = lr.predict_proba(X_test)
        print(f"✓ Olasılıklar: {lr_proba[:1]}")
        
        # Neural Network
        print("\n[Neural Network]")
        nn = NeuralNetworkClassifier(
            input_size=X_train.shape[1],
            hidden_sizes=[64, 32],
            dropout_rate=0.3,
            learning_rate=0.001
        )
        nn.train(X_train, y_train, epochs=10, batch_size=16)
        print(f"✓ Model eğitildi (10 epoch)")
        
        nn_pred = nn.predict(X_test)
        print(f"✓ Tahmin yapıldı: {nn_pred[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False


def test_forecasting():
    """Test: Tahminleme"""
    print("\n" + "="*60)
    print("TEST 3: FORECASTING")
    print("="*60)
    
    try:
        # Sahte zaman serisi
        np.random.seed(42)
        t = np.arange(0, 100)
        y = 50 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 2, len(t))
        
        train_idx = int(0.8 * len(y))
        y_train = y[:train_idx]
        
        # Linear Forecaster
        print("\n[Linear Forecaster]")
        lf = LinearForecaster(lookback=12, degree=1)
        lf.fit(y_train)
        forecast_lf = lf.forecast(steps=24, last_sequence=y_train[-12:])
        print(f"✓ Tahmin: {forecast_lf[:5]}")
        
        # ARIMA
        print("\n[ARIMA Forecaster]")
        af = ARIMAForecaster(order=(1, 1, 1))
        af.fit(y_train)
        forecast_af = af.forecast(steps=24)
        print(f"✓ Tahmin: {forecast_af[:5]}")
        
        # Prophet
        print("\n[Prophet Forecaster]")
        df_prophet = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=len(y_train), freq='h'),
            'y': y_train
        })
        pf = ProphetForecaster(interval_width=0.95)
        pf.fit(df_prophet)
        forecast_pf, lower, upper = pf.forecast(steps=24, freq='h')
        print(f"✓ Tahmin: {forecast_pf[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False


def test_evaluation(X_test, y_test):
    """Test: Değerlendirme"""
    print("\n" + "="*60)
    print("TEST 4: EVALUATION")
    print("="*60)
    
    try:
        # Sınıflandırma Evaluator
        print("\n[Classification Evaluator]")
        clf_eval = ClassificationEvaluator()
        y_pred = np.random.choice(['RS', 'RP'], len(y_test))
        metrics = clf_eval.evaluate(y_test, y_pred)
        print(f"✓ Accuracy: {metrics['accuracy']:.4f}")
        print(f"✓ F1: {metrics['f1']:.4f}")
        
        # Tahminleme Evaluator
        print("\n[Forecasting Evaluator]")
        y_true_fcst = np.random.randn(50) * 10 + 100
        y_pred_fcst = y_true_fcst + np.random.randn(50) * 5
        
        fcst_eval = ForecastingEvaluator()
        fcst_metrics = fcst_eval.evaluate(y_true_fcst, y_pred_fcst)
        print(f"✓ MAE: {fcst_metrics['mae']:.4f}")
        print(f"✓ RMSE: {fcst_metrics['rmse']:.4f}")
        print(f"✓ R2: {fcst_metrics['r2']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False


def test_integration():
    """Test: Entegrasyon"""
    print("\n" + "="*60)
    print("TEST 5: INTEGRATION")
    print("="*60)
    
    try:
        # DataPipeline
        print("\n[DataPipeline]")
        df_data = pd.DataFrame({
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50),
            'label': np.random.choice(['RS', 'RP'], 50)
        })
        
        pipeline = DataPipeline()
        X, y, feature_names = pipeline.process_person1_data(df_data)
        print(f"✓ Veri işlendi: X shape={X.shape}, Features={len(feature_names)}")
        
        # ModelIntegrator
        print("\n[ModelIntegrator]")
        integrator = ModelIntegrator(models_dir='./test_models')
        info = integrator.get_model_info()
        print(f"✓ Integrator başlatıldı: {info}")
        
        # ResultsFormatter
        print("\n[ResultsFormatter]")
        mock_results = {
            'predictions': np.array(['RS', 'RP', 'RS']),
            'probabilities': np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        }
        formatted = ResultsFormatter.format_classification_results(mock_results)
        print(f"✓ Sonuçlar formatlandı: {formatted.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False


def main():
    """Ana test fonksiyonu"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║  PERSON 2 (MODEL ARCHITECT) - MODULE TEST SUITE        ║")
    print("╚" + "="*58 + "╝")
    
    results = {}
    
    # Test 1: Data Preparation
    X_train, X_test, y_train, y_test, success = test_data_preparation()
    results['Data Preparation'] = success
    
    # Test 2: Classification
    if success:
        results['Classification'] = test_classification(X_train, X_test, y_train, y_test)
    
    # Test 3: Forecasting
    results['Forecasting'] = test_forecasting()
    
    # Test 4: Evaluation
    if success:
        results['Evaluation'] = test_evaluation(X_test, y_test)
    
    # Test 5: Integration
    results['Integration'] = test_integration()
    
    # Özet
    print("\n" + "="*60)
    print("TEST ÖZETİ")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:30} {status}")
    
    print("-"*60)
    print(f"TOPLAM: {passed}/{total} testler geçti")
    
    if passed == total:
        print("\n🎉 Tüm testler başarılı!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test başarısız")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
