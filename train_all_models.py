"""
TRAINING GUIDE - Kişi 2'nin Adım Adım Training Kılavuzu
=====================================================

Bu script Kişi 1'in verilerini alarak sınıflandırma ve 
tahminleme modellerini eğiten eksiksiz bir pipeline'ı gösterir.

Kullanım: python train_all_models.py
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path

# Kütüphaneleri import et
from src.model_prep import DataPreprocessor, create_lag_features
from src.classification import LogisticRegressionClassifier, NeuralNetworkClassifier
from src.forecasting import LinearForecaster, ARIMAForecaster, ProphetForecaster
from src.evaluator import ClassificationEvaluator, ForecastingEvaluator, ModelComparator
from src.integration_logic import ModelIntegrator, DataPipeline, ResultsFormatter

# Logger kurulumu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_person1_data(data_path: str) -> pd.DataFrame:
    """
    Kişi 1'den gelen veriyi yükle.
    
    Beklenen format:
    - CSV dosyası
    - Sütunlar: feature_1, feature_2, ..., label (RS/RP)
    - İsteğe bağlı: timestamp (zaman serisi için)
    
    Args:
        data_path: CSV dosya yolu
        
    Returns:
        pd.DataFrame: Yüklü veri
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Veri yüklendi: {data_path} | Shape: {df.shape}")
        logger.info(f"Sütunlar: {df.columns.tolist()}")
        logger.info(f"Label dağılımı:\n{df['label'].value_counts()}")
        return df
    except Exception as e:
        logger.error(f"Veri yükleme errorsı: {e}")
        raise


def train_classification_models(X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray,
                               models_dir: str = 'models') -> dict:
    """
    Sınıflandırma modellerini eğit ve karşılaştır.
    
    Eğitilecek Modeller:
    1. Logistic Regression (Baseline)
    2. Neural Network (Deep Learning)
    
    Args:
        X_train, X_test: Training ve test özellikleri
        y_train, y_test: Training ve test etiketleri
        models_dir: Model kayıt dizini
        
    Returns:
        dict: Tüm modellerle ilgili infoler
    """
    logger.info("="*60)
    logger.info("SINIFLAMA MODELLERİ EĞİTİMİ BAŞLANIYOR")
    logger.info("="*60)
    
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    results = {}
    
    # 1. LOGISTIC REGRESSION
    logger.info("\n[1/2] Logistic Regression eğitiliyor...")
    try:
        lr_clf = LogisticRegressionClassifier(C=1.0, solver='lbfgs', max_iter=1000)
        lr_clf.train(X_train, y_train)
        
        lr_eval = ClassificationEvaluator()
        lr_metrics = lr_eval.evaluate(y_test, lr_clf.predict(X_test))
        
        # Model kaydet
        lr_model_path = models_path / 'classification_logistic_regression.pkl'
        lr_clf.save(str(lr_model_path))
        
        results['logistic_regression'] = {
            'model_path': str(lr_model_path),
            'metrics': lr_metrics,
            'trained': True
        }
        
        logger.info(f"✓ Logistic Regression başarıyla eğitildi")
        logger.info(f"  Accuracy: {lr_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {lr_metrics['precision']:.4f}")
        logger.info(f"  Recall: {lr_metrics['recall']:.4f}")
        logger.info(f"  F1: {lr_metrics['f1']:.4f}")
        
    except Exception as e:
        logger.error(f"Logistic Regression eğitim errorsı: {e}")
        results['logistic_regression'] = {'trained': False, 'error': str(e)}
    
    # 2. NEURAL NETWORK
    logger.info("\n[2/2] Neural Network eğitiliyor...")
    try:
        nn_clf = NeuralNetworkClassifier(
            input_size=X_train.shape[1],
            hidden_sizes=[128, 64, 32],
            dropout_rate=0.3,
            learning_rate=0.001
        )
        
        # Validation split
        split_idx = int(0.8 * len(X_train))
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train_nn = X_train[:split_idx]
        y_train_nn = y_train[:split_idx]
        
        nn_clf.train(X_train_nn, y_train_nn, epochs=50, batch_size=32,
                    X_val=X_val, y_val=y_val)
        
        nn_eval = ClassificationEvaluator()
        nn_metrics = nn_eval.evaluate(y_test, nn_clf.predict(X_test))
        
        # Model kaydet
        nn_model_path = models_path / 'classification_neural_network.pt'
        nn_clf.save(str(nn_model_path))
        
        results['neural_network'] = {
            'model_path': str(nn_model_path),
            'metrics': nn_metrics,
            'trained': True
        }
        
        logger.info(f"✓ Neural Network başarıyla eğitildi")
        logger.info(f"  Accuracy: {nn_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {nn_metrics['precision']:.4f}")
        logger.info(f"  Recall: {nn_metrics['recall']:.4f}")
        logger.info(f"  F1: {nn_metrics['f1']:.4f}")
        
    except Exception as e:
        logger.error(f"Neural Network eğitim errorsı: {e}")
        results['neural_network'] = {'trained': False, 'error': str(e)}
    
    return results


def train_forecasting_models(timeseries_data: np.ndarray,
                            test_size: float = 0.2,
                            models_dir: str = 'models') -> dict:
    """
    Tahminleme modellerini eğit ve karşılaştır.
    
    Eğitilecek Modeller:
    1. Linear Forecaster
    2. ARIMA
    3. Prophet
    
    Args:
        timeseries_data: 1D zaman serisi
        test_size: Test seti oranı
        models_dir: Model kayıt dizini
        
    Returns:
        dict: Tüm modellerle ilgili infoler
    """
    logger.info("\n" + "="*60)
    logger.info("TAHMİNLEME MODELLERİ EĞİTİMİ BAŞLANIYOR")
    logger.info("="*60)
    
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    # Train/Test split
    split_idx = int(len(timeseries_data) * (1 - test_size))
    y_train = timeseries_data[:split_idx]
    y_test = timeseries_data[split_idx:]
    
    results = {}
    
    # 1. LINEAR FORECASTER
    logger.info("\n[1/3] Linear Forecaster eğitiliyor...")
    try:
        lf = LinearForecaster(lookback=24, degree=1)
        lf.fit(y_train)
        
        # Tahmin yap
        forecast_lf = lf.forecast(steps=len(y_test), last_sequence=y_train[-24:])
        
        # Evaluate
        lf_eval = ForecastingEvaluator()
        lf_metrics = lf_eval.evaluate(y_test, forecast_lf[:len(y_test)])
        
        # Model kaydet
        lf_model_path = models_path / 'forecasting_linear.pkl'
        lf.save(str(lf_model_path))
        
        results['linear'] = {
            'model_path': str(lf_model_path),
            'metrics': lf_metrics,
            'trained': True
        }
        
        logger.info(f"✓ Linear Forecaster başarıyla eğitildi")
        logger.info(f"  MAE: {lf_metrics['mae']:.4f}")
        logger.info(f"  RMSE: {lf_metrics['rmse']:.4f}")
        logger.info(f"  R2: {lf_metrics['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Linear Forecaster eğitim errorsı: {e}")
        results['linear'] = {'trained': False, 'error': str(e)}
    
    # 2. ARIMA
    logger.info("\n[2/3] ARIMA eğitiliyor...")
    try:
        af = ARIMAForecaster(order=(1, 1, 1))
        af.fit(y_train)
        
        # Tahmin yap
        forecast_af = af.forecast(steps=len(y_test))
        
        # Evaluate
        af_eval = ForecastingEvaluator()
        af_metrics = af_eval.evaluate(y_test, forecast_af[:len(y_test)])
        
        # Model kaydet
        af_model_path = models_path / 'forecasting_arima.pkl'
        af.save(str(af_model_path))
        
        results['arima'] = {
            'model_path': str(af_model_path),
            'metrics': af_metrics,
            'trained': True
        }
        
        logger.info(f"✓ ARIMA başarıyla eğitildi")
        logger.info(f"  MAE: {af_metrics['mae']:.4f}")
        logger.info(f"  RMSE: {af_metrics['rmse']:.4f}")
        logger.info(f"  R2: {af_metrics['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"ARIMA eğitim errorsı: {e}")
        results['arima'] = {'trained': False, 'error': str(e)}
    
    # 3. PROPHET
    logger.info("\n[3/3] Prophet eğitiliyor...")
    try:
        df_prophet = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=len(y_train), freq='h'),
            'y': y_train
        })
        
        pf = ProphetForecaster(interval_width=0.95, seasonality_mode='additive')
        pf.fit(df_prophet)
        
        # Tahmin yap
        forecast_pf, lower_pf, upper_pf = pf.forecast(steps=len(y_test), freq='h')
        
        # Evaluate
        pf_eval = ForecastingEvaluator()
        pf_metrics = pf_eval.evaluate(y_test, forecast_pf[:len(y_test)])
        
        # Model kaydet
        pf_model_path = models_path / 'forecasting_prophet.pkl'
        pf.save(str(pf_model_path))
        
        results['prophet'] = {
            'model_path': str(pf_model_path),
            'metrics': pf_metrics,
            'trained': True
        }
        
        logger.info(f"✓ Prophet başarıyla eğitildi")
        logger.info(f"  MAE: {pf_metrics['mae']:.4f}")
        logger.info(f"  RMSE: {pf_metrics['rmse']:.4f}")
        logger.info(f"  R2: {pf_metrics['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Prophet eğitim errorsı: {e}")
        results['prophet'] = {'trained': False, 'error': str(e)}
    
    return results


def generate_training_report(clf_results: dict, fcst_results: dict,
                            output_file: str = 'training_report.json'):
    """
    Training raporunu oluştur.
    
    Args:
        clf_results: Sınıflandırma sonuçları
        fcst_results: Tahminleme sonuçları
        output_file: Rapor dosya adı
    """
    logger.info("\n" + "="*60)
    logger.info("EĞITIM RAPORU OLUŞTURULUYOR")
    logger.info("="*60)
    
    report = {
        'training_date': pd.Timestamp.now().isoformat(),
        'classification_models': clf_results,
        'forecasting_models': fcst_results,
        'summary': {
            'total_classification_models': len([m for m in clf_results.values() if m.get('trained', False)]),
            'total_forecasting_models': len([m for m in fcst_results.values() if m.get('trained', False)])
        }
    }
    
    # JSON olarak kaydet
    with open(output_file, 'w') as f:
        # Metrics içindeki numpy arrays'i serializable yap
        report_str = json.dumps(report, indent=2, default=str)
        f.write(report_str)
    
    logger.info(f"✓ Rapor kaydedildi: {output_file}")
    
    # Konsola yazdır
    logger.info("\nÖZET:")
    logger.info(f"  Sınıflandırma Modelleri: {report['summary']['total_classification_models']}")
    logger.info(f"  Forecasting Models: {report['summary']['total_forecasting_models']}")


def main():
    """Ana eğitim fonksiyonu"""
    logger.info("╔" + "="*58 + "╗")
    logger.info("║  PERSON 2 - COMPLETE TRAINING PIPELINE              ║")
    logger.info("╚" + "="*58 + "╝")
    
    try:
        # 1. SAHTe VERİ OLUŞTUR (Kişi 1'in verisi geldiğinde düzenlenecek)
        logger.info("\n[STEP 1] Sahte veri oluşturuluyor...")
        np.random.seed(42)
        
        # Sınıflandırma verisi
        X_clf_data = np.random.randn(500, 15)
        y_clf_data = np.random.choice(['RS', 'RP'], 500)
        
        clf_df = pd.DataFrame(X_clf_data, columns=[f'feature_{i}' for i in range(15)])
        clf_df['label'] = y_clf_data
        clf_df['timestamp'] = pd.date_range('2024-01-01', periods=500, freq='h')
        
        # Tahminleme verisi (zaman serisi)
        t = np.arange(0, 500)
        timeseries = 50 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 3, len(t))
        
        logger.info("✓ Sahte veri oluşturuldu")
        
        # 2. VERİ HAZIRLIĞI
        logger.info("\n[STEP 2] Veri hazırlanıyor...")
        preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
        
        # Dengeleme
        clf_df_balanced = preprocessor.balance_dataset(clf_df, label_col='label')
        
        # Train/Test split
        X_train, X_test, y_train, y_test, features = preprocessor.train_test_split_timeseries(
            clf_df_balanced,
            time_col='timestamp',
            label_col='label'
        )
        
        # Normalization
        X_train_norm, X_test_norm = preprocessor.normalize_features(X_train, X_test)
        
        logger.info("✓ Veri hazırlandı")
        
        # 3. SINIFLAMA MODELLERİ
        clf_results = train_classification_models(X_train_norm, X_test_norm, y_train, y_test)
        
        # 4. TAHMİNLEME MODELLERİ
        fcst_results = train_forecasting_models(timeseries, test_size=0.2)
        
        # 5. RAPOR OLUŞTUR
        generate_training_report(clf_results, fcst_results)
        
        logger.info("\n" + "="*60)
        logger.info("🎉 EĞITIM TAMAMLANDI")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline errorsı: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
