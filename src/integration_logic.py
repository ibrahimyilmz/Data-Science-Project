"""
Integration Logic Module - Person 1, 2, 3 Integration
====================================================
Middle layer that combines modules from three team members.

Connection Points:
    - Person 1 (Data Architect): Clustering results (RS/RP labels)
    - Person 2 (Model Architect): Classification and forecasting models
    - Person 3 (Integrator): Streamlit Dashboard interface

Tasks:
    - Prepare Person 1's features for models
    - Coordinate classification and forecasting operations
    - Format results for Streamlit
"""

import numpy as np
import pandas as pd
import pickle
import torch
import logging
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path
import json

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelIntegrator:
    """
    All modelleri entegre eden ana sınıf.
    
    Sorumluluğu:
    - Modelleri yükle/başlat
    - Pipeline'ı koordine et
    - Sonuçları döndür
    """
    
    def __init__(self, models_dir: str = 'models', config_file: Optional[str] = None):
        """
        ModelIntegrator Initialize.
        
        Args:
            models_dir (str): Kayıtlı modellerin dizini
            config_file (str, optional): Konfigürasyon JSON dosyası
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.config = self._load_config(config_file) if config_file else {}
        
        # Model containers
        self.classification_model = None
        self.forecasting_models = {}
        self.feature_scaler = None
        self.label_encoder = None
        
        self.logger = logger
        self.logger.info(f"ModelIntegrator initialized | Models Dir: {self.models_dir}")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Konfigürasyon dosyasını yükle.
        
        Args:
            config_file (str): JSON config dosyası yolu
            
        Returns:
            Dict: Konfigürasyon parametreleri
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Konfigürasyon loaded: {config_file}")
            return config
        except Exception as e:
            self.logger.warning(f"Config yükleme errorsı: {str(e)} - Varsayılanlar kullanılacak")
            return {}
    
    def load_classification_model(self, model_path: str, model_type: str = 'logistic'):
        """
        Sınıflandırma modelini yükle.
        
        Args:
            model_path (str): Model dosya yolu
            model_type (str): 'logistic' veya 'neural_network'
        """
        try:
            if model_type == 'logistic':
                import joblib
                self.classification_model = joblib.load(model_path)
            elif model_type == 'neural_network':
                # PyTorch model yükleme
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                state_dict = torch.load(model_path, map_location=device)
                # Model architecturesı bilindiğini varsayarak (şu an placeholder)
                self.classification_model = state_dict
            else:
                raise ValueError(f"Bilinmeyen model tipi: {model_type}")
            
            self.logger.info(f"Sınıflandırma modeli loaded: {model_path} ({model_type})")
            
        except Exception as e:
            self.logger.error(f"Model yükleme errorsı: {str(e)}")
            raise
    
    def load_forecasting_model(self, model_name: str, model_path: str):
        """
        Forecasting modelini yükle.
        
        Args:
            model_name (str): Model adı ('linear', 'arima', 'prophet')
            model_path (str): Model dosya yolu
        """
        try:
            import joblib
            self.forecasting_models[model_name] = joblib.load(model_path)
            self.logger.info(f"Forecasting modeli loaded: {model_name} - {model_path}")
            
        except Exception as e:
            self.logger.error(f"Model yükleme errorsı: {str(e)}")
            raise
    
    def load_feature_scaler(self, scaler_path: str):
        """
        Özellik normalizasyon aracını yükle.
        
        Args:
            scaler_path (str): Scaler dosya yolu (joblib format)
        """
        try:
            import joblib
            self.feature_scaler = joblib.load(scaler_path)
            self.logger.info(f"Feature scaler loaded: {scaler_path}")
            
        except Exception as e:
            self.logger.error(f"Scaler yükleme errorsı: {str(e)}")
            raise
    
    def classify_residence(self, features: np.ndarray, 
                          feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evleri RS/RP olarak sınıflandır.
        
        Args:
            features (np.ndarray): Ev özellikleri (n_samples x n_features)
            feature_names (list, optional): Özellik adları
            
        Returns:
            Dict: Sınıflandırma sonuçları
        """
        if self.classification_model is None:
            raise ValueError("Sınıflandırma modeli henüz yüklenmedi")
        
        try:
            # Features normalize et (eğer scaler varsa)
            if self.feature_scaler is not None:
                features_scaled = self.feature_scaler.transform(features)
            else:
                features_scaled = features
            
            # Make predictions
            predictions = self.classification_model.predict(features_scaled)
            
            # Olasılıkları al (varsa)
            try:
                probabilities = self.classification_model.predict_proba(features_scaled)
            except:
                probabilities = None
            
            results = {
                'predictions': predictions,
                'probabilities': probabilities,
                'n_samples': features.shape[0],
                'feature_names': feature_names,
                'rs_count': np.sum(predictions == 'RS') if isinstance(predictions[0], str) else np.sum(predictions == 0),
                'rp_count': np.sum(predictions == 'RP') if isinstance(predictions[0], str) else np.sum(predictions == 1)
            }
            
            self.logger.info(f"Sınıflandırma tamamlandı: {results['rs_count']} RS, {results['rp_count']} RP")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Sınıflandırma errorsı: {str(e)}")
            raise
    
    def forecast_consumption(self, timeseries_data: np.ndarray, 
                            steps: int = 24, 
                            model_name: str = 'arima') -> Dict[str, Any]:
        """
        Enerji tüketimini tahmin et.
        
        Args:
            timeseries_data (np.ndarray): Geçmiş tüketim verileri
            steps (int): Kaç adım ileri tahmin et
            model_name (str): Kullanılacak model adı
            
        Returns:
            Dict: Tahmin sonuçları
        """
        if model_name not in self.forecasting_models:
            raise ValueError(f"Model '{model_name}' yüklenmedi. "
                           f"Mevcut modeller: {list(self.forecasting_models.keys())}")
        
        try:
            model = self.forecasting_models[model_name]
            
            # Model tipine göre tahmin yap
            if model_name == 'linear':
                # Linear model için son sequence'i al
                if len(timeseries_data) < 24:
                    raise ValueError("Linear model için en az 24 adım gerekli")
                forecast = model.forecast(steps=steps, last_sequence=timeseries_data[-24:])
                
            elif model_name == 'arima':
                forecast = model.forecast(steps=steps)
                
            elif model_name == 'prophet':
                forecast, lower_bound, upper_bound = model.forecast(steps=steps)
                
                return {
                    'model': model_name,
                    'forecast': forecast,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'steps': steps,
                    'timestamps': pd.date_range(start=pd.Timestamp.now(), 
                                              periods=steps, freq='h')
                }
            
            results = {
                'model': model_name,
                'forecast': forecast,
                'steps': steps,
                'timestamps': pd.date_range(start=pd.Timestamp.now(), 
                                          periods=steps, freq='h')
            }
            
            self.logger.info(f"Forecasting tamamlandı: {model_name} modeli | "
                           f"{steps} adım ileri tahmin")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Forecasting error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Yüklü modeller hakkında info al.
        
        Returns:
            Dict: Model infoleri
        """
        return {
            'classification_model': 'Yüklü' if self.classification_model else 'Yüklenmedi',
            'forecasting_models': list(self.forecasting_models.keys()),
            'feature_scaler': 'Yüklü' if self.feature_scaler else 'Yüklenmedi',
            'models_directory': str(self.models_dir)
        }


class DataPipeline:
    """
    Kişi 1'den gelen veriyi Kişi 2'nin modelleri için hazırlayan pipeline.
    
    İş akışı:
    1. Kişi 1'den etiketlenmiş veri al
    2. Features ayıkla
    3. Normalization yap
    4. Modellere gönder
    """
    
    def __init__(self):
        self.logger = logger
        self.feature_names = None
        self.scaler = None
    
    def process_person1_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Kişi 1'in hazırladığı veriyi işle.
        
        Beklenen sütunlar:
        - Özzellik sütunları (PCA sonuçları veya ham özellikler)
        - 'label': RS/RP etiketleri
        
        Args:
            df (pd.DataFrame): Kişi 1'in veri seti
            
        Returns:
            Tuple: (X_features, y_labels, feature_names)
        """
        try:
            # Label sütununu ayır
            if 'label' not in df.columns:
                raise ValueError("DataFrame 'label' sütununu içermeli")
            
            y = df['label'].values
            X = df.drop(columns=['label']).values
            self.feature_names = df.drop(columns=['label']).columns.tolist()
            
            self.logger.info(f"Veri işlendi: {X.shape[0]} örnek, {X.shape[1]} özellik")
            
            return X, y, self.feature_names
            
        except Exception as e:
            self.logger.error(f"Veri işleme errorsı: {str(e)}")
            raise
    
    def process_timeseries_for_forecast(self, df: pd.DataFrame, 
                                       value_col: str = 'consumption',
                                       time_col: str = 'timestamp') -> np.ndarray:
        """
        Forecasting için zaman serisini hazırla.
        
        Args:
            df (pd.DataFrame): Time series dataset
            value_col (str): Value column name
            time_col (str): Time column name
            
        Returns:
            np.ndarray: İşlenmiş zaman serisi
        """
        try:
            df = df.sort_values(by=time_col).reset_index(drop=True)
            timeseries = df[value_col].values
            
            self.logger.info(f"Time series hazırlandı: {len(timeseries)} adım")
            
            return timeseries
            
        except Exception as e:
            self.logger.error(f"Time series işleme errorsı: {str(e)}")
            raise


class ResultsFormatter:
    """
    Model sonuçlarını Streamlit Dashboard'a hazır format'a dönüştür.
    """
    
    @staticmethod
    def format_classification_results(results: Dict[str, Any]) -> pd.DataFrame:
        """
        Sınıflandırma sonuçlarını tablo format'ına çevir.
        
        Args:
            results (dict): classification_model sonuçları
            
        Returns:
            pd.DataFrame: Formatlanmış sonuçlar
        """
        try:
            df = pd.DataFrame({
                'House_ID': np.arange(len(results['predictions'])),
                'Predicted_Type': results['predictions'],
                'Confidence': results['probabilities'].max(axis=1) if results['probabilities'] is not None else None
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Sonuç biçimlendirme errorsı: {str(e)}")
            raise
    
    @staticmethod
    def format_forecast_results(results: Dict[str, Any]) -> pd.DataFrame:
        """
        Forecasting sonuçlarını tablo format'ına çevir.
        
        Args:
            results (dict): forecast_consumption sonuçları
            
        Returns:
            pd.DataFrame: Formatlanmış tahmin
        """
        try:
            df = pd.DataFrame({
                'Timestamp': results['timestamps'],
                'Forecasted_Consumption': results['forecast']
            })
            
            # Prophet modelinden güven aralıkları varsa ekle
            if 'lower_bound' in results:
                df['Lower_Bound'] = results['lower_bound']
                df['Upper_Bound'] = results['upper_bound']
            
            return df
            
        except Exception as e:
            logger.error(f"Sonuç biçimlendirme errorsı: {str(e)}")
            raise


# Test code
if __name__ == "__main__":
    print("Integration Logic Module - Test")
    print("=" * 50)
    
    # ModelIntegrator test
    print("\n--- ModelIntegrator Test ---")
    integrator = ModelIntegrator(models_dir='./models')
    
    print(f"Integrator initialized: {integrator.get_model_info()}")
    
    # DataPipeline test
    print("\n--- DataPipeline Test ---")
    pipeline = DataPipeline()
    
    # Sahte veri oluştur
    df_test = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100),
        'label': np.random.choice(['RS', 'RP'], 100)
    })
    
    X, y, feature_names = pipeline.process_person1_data(df_test)
    print(f"İşlenmiş veri: X shape={X.shape}, y shape={y.shape}")
    print(f"Özellik adları: {feature_names}")
    
    # ResultsFormatter test
    print("\n--- ResultsFormatter Test ---")
    mock_classification_results = {
        'predictions': np.array(['RS', 'RP', 'RS']),
        'probabilities': np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    }
    
    formatted_clf = ResultsFormatter.format_classification_results(mock_classification_results)
    print(f"Formatlanmış sınıflandırma sonuçları:\n{formatted_clf}")
