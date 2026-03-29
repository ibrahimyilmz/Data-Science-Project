"""
Forecasting Module - Time Series Forecasting Models
=======================================================
Contains models that predict future energy consumption.

Models:
    1. LinearForecaster: Simple Linear Regression (baseline)
    2. ARIMAForecaster: ARIMA (AutoRegressive Integrated Moving Average)
    3. ProphetForecaster: Facebook Prophet (trend + seasonality)
    
Hyperparameters:
    - ARIMA: (p=1, d=1, q=1) - Optimized experimentally
    - Prophet: changepoint_prior_scale=0.05, seasonality_scale=10
    - Linear: degree=1 (simple linear relationship)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
import joblib
import logging
from typing import Tuple, Dict, Optional, List, Union
from abc import ABC, abstractmethod

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """
    All tahmin modellerinin temel sınıfı.
    
    Ortak metodlar:
    - fit(): Train model
    - forecast(): Make predictions
    - evaluate(): Modeli değerlendir
    - save(): Save model
    - load(): Load model
    """
    
    def __init__(self, name: str = "BaseForecaster"):
        self.name = name
        self.model = None
        self.is_trained = False
        self.logger = logger
        
    @abstractmethod
    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], 
           y_train: Union[np.ndarray, pd.Series], **kwargs):
        """Train model (abstract metod)"""
        pass
    
    @abstractmethod
    def forecast(self, steps: int) -> np.ndarray:
        """Make predictions (abstract metod)"""
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Forecasting performansını değerlendir.
        
        Metrikler:
        - MAE (Mean Absolute Error): Ortalama mutlak error
        - RMSE (Root Mean Squared Error): Karekök ortalama kare error
        - R2 Score: Modelin varyans açıklama oranı
        
        Returns:
            Dict: MAE, RMSE, R2 skorları
        """
        try:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            self.logger.info(f"{self.name} Evaluateme Sonuçları:")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluateme errorsı: {str(e)}")
            raise
    
    @abstractmethod
    def save(self, filepath: str):
        """Save model"""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load model"""
        pass


class LinearForecaster(BaseForecaster):
    """
    Çok değişkenli Linear Regression tabanlı tahmin modeli.
    
    Features:
    - Fast training and prediction
    - Use as baseline model
    - Açık ve anlaşılır sonuçlar
    
    Yöntem:
    - Geçmiş 24 saati (lag) kullanarak gelecek saati tahmin et
    - İsteğe bağlı Polynomial Features desteği
    
    NOT: Time series doğası göz önüne alınarak lag özellikler kullanılır
    """
    
    def __init__(self, lookback: int = 24, degree: int = 1):
        """
        LinearForecaster Initialize.
        
        Args:
            lookback (int): Geçmiş kaç adımı göz önüne alalım (default: 24 saat)
            degree (int): Polinom derecesi (1=doğrusal, 2=kuadratik)
        """
        super().__init__(name="LinearForecaster")
        self.lookback = lookback
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree) if degree > 1 else None
        self.model = LinearRegression()
        
        self.logger.info(f"LinearForecaster initialized | Lookback: {lookback}, Degree: {degree}")
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time seriesnden gecikmeli (lag) özellikleri oluştur.
        
        Args:
            data (np.ndarray): 1D zaman serisi
            
        Returns:
            Tuple: (X_sequences, y_sequences) - Girdi ve hedef çiftleri
        """
        X, y = [], []
        
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])
        
        return np.array(X), np.array(y)
    
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None, **kwargs):
        """
        Train model.
        
        Args:
            X_train (np.ndarray): Training serisi veya özellik matrisi
            y_train (np.ndarray, optional): Hedef değerler
            
        NOT: Eğer y_train None ise, X_train'in son sütunu hedef olarak kullan
        """
        try:
            # Eğer 1D zaman serisi verilmişse sequence oluştur
            if X_train.ndim == 1:
                X_sequences, y_sequences = self._create_sequences(X_train)
            else:
                X_sequences = X_train
                y_sequences = y_train if y_train is not None else None
            
            if y_sequences is None:
                raise ValueError("Hedef değerler (y_train) belirtilmesi gerekir")
            
            # Polinom özelliklerini uygula
            if self.poly_features is not None:
                X_sequences = self.poly_features.fit_transform(X_sequences)
            
            self.model.fit(X_sequences, y_sequences)
            self.is_trained = True
            
            self.logger.info(f"LinearForecaster eğitimi tamamlandı | "
                           f"Training boyutu: {X_sequences.shape[0]}, "
                           f"Number of features: {X_sequences.shape[1]}")
            
        except Exception as e:
            self.logger.error(f"Training errorsı: {str(e)}")
            raise
    
    def forecast(self, steps: int = 24, last_sequence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Gelecek değerleri tahmin et.
        
        Args:
            steps (int): Kaç adım ileri tahmin et (default: 24 saat)
            last_sequence (np.ndarray, optional): Son bilinilen serilerin başlangıç noktası
            
        Returns:
            np.ndarray: Tahmin edilen değerler
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        if last_sequence is None:
            raise ValueError("last_sequence belirtilmesi gerekir")
        
        try:
            forecast = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # Şu anki sequence'i özellik olarak kullan
                X_input = current_sequence.reshape(1, -1)
                
                if self.poly_features is not None:
                    X_input = self.poly_features.transform(X_input)
                
                # Make predictions
                next_value = self.model.predict(X_input)[0]
                forecast.append(next_value)
                
                # Sequence'i güncelle (en eski değeri çıkar, tahmin edilen yeni değeri ekle)
                current_sequence = np.append(current_sequence[1:], next_value)
            
            return np.array(forecast)
            
        except Exception as e:
            self.logger.error(f"Tahmin errorsı: {str(e)}")
            raise
    
    def save(self, filepath: str):
        """Modeli joblib formatında kaydet."""
        try:
            joblib.dump({
                'model': self.model,
                'lookback': self.lookback,
                'degree': self.degree,
                'poly_features': self.poly_features
            }, filepath)
            self.logger.info(f"Model saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Model kayıt errorsı: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """Modeli joblib formatından yükle."""
        try:
            loaded = joblib.load(filepath)
            self.model = loaded['model']
            self.lookback = loaded['lookback']
            self.degree = loaded['degree']
            self.poly_features = loaded['poly_features']
            self.is_trained = True
            self.logger.info(f"Model loaded: {filepath}")
        except Exception as e:
            self.logger.error(f"Model yükleme errorsı: {str(e)}")
            raise


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA tabanlı zaman serisi tahmin modeli.
    
    ARIMA = AutoRegressive Integrated Moving Average
    
    Parametreler:
    - p (AutoRegressive): Geçmiş değerlerin bağımlılığı
    - d (Integrated): Farklandırma derecesi (trend kaldırma)
    - q (Moving Average): Error geçmiş değerlerinin bağımlılığı
    
    Seçilen parametreler (1,1,1):
    - p=1: 1. mertebeden otoregresyon
    - d=1: 1 kez farklandırma
    - q=1: 1. mertebeden hareketli ortalama
    
    NOT: Bu parametreler pilot veri üzerinde AIC/BIC kriterleri
    kullanılarak optimize edilmelidir.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        ARIMAForecaster Initialize.
        
        Args:
            order (tuple): (p, d, q) parametreleri
        """
        super().__init__(name="ARIMAForecaster")
        self.order = order
        self.model = None
        self.fitted_model = None
        
        self.logger.info(f"ARIMAForecaster initialized | Order: {order}")
    
    def fit(self, X_train: Union[np.ndarray, pd.Series], y_train: Optional[np.ndarray] = None, **kwargs):
        """
        ARIMA modelini eğit.
        
        Args:
            X_train (np.ndarray or pd.Series): Time series verisi
            y_train: Kullanılmaz (ARIMA univariate model)
        """
        try:
            # Pandas Series'e dönüştür
            if isinstance(X_train, np.ndarray):
                X_train = pd.Series(X_train)
            
            # ARIMA modeli kur ve eğit
            self.model = ARIMA(X_train, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_trained = True
            
            self.logger.info(f"ARIMA eğitimi tamamlandı | "
                           f"Train boyutu: {len(X_train)}")
            
            # Model özeti
            if hasattr(self.fitted_model, 'aic'):
                self.logger.info(f"  AIC: {self.fitted_model.aic:.2f}, "
                               f"BIC: {self.fitted_model.bic:.2f}")
            
        except Exception as e:
            self.logger.error(f"Training errorsı: {str(e)}")
            raise
    
    def forecast(self, steps: int = 24) -> np.ndarray:
        """
        ARIMA ile tahmin yap.
        
        Args:
            steps (int): Kaç adım ileri tahmin et
            
        Returns:
            np.ndarray: Tahmin edilen değerler
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        try:
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean.values
            
            self.logger.info(f"ARIMA tahminlemesi: {steps} adım")
            return forecast
            
        except Exception as e:
            self.logger.error(f"Tahmin errorsı: {str(e)}")
            raise
    
    def get_confidence_intervals(self, steps: int = 24, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecasting için güven aralıklarını al.
        
        Args:
            steps (int): Tahmin adımları
            alpha (float): Anlamlılık düzeyi (default: 0.05 → %95 güven)
            
        Returns:
            Tuple: (alt_sınır, üst_sınır) dizileri
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        try:
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            conf_int = forecast_result.conf_int(alpha=alpha)
            
            return conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values
            
        except Exception as e:
            self.logger.error(f"Güven aralığı errorsı: {str(e)}")
            raise
    
    def save(self, filepath: str):
        """Modeli joblib formatında kaydet."""
        try:
            joblib.dump({
                'fitted_model': self.fitted_model,
                'order': self.order
            }, filepath)
            self.logger.info(f"Model saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Model kayıt errorsı: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """Modeli joblib formatından yükle."""
        try:
            loaded = joblib.load(filepath)
            self.fitted_model = loaded['fitted_model']
            self.order = loaded['order']
            self.is_trained = True
            self.logger.info(f"Model loaded: {filepath}")
        except Exception as e:
            self.logger.error(f"Model yükleme errorsı: {str(e)}")
            raise


class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet tabanlı zaman serisi tahmin modeli.
    
    Prophet özellikleri:
    - Trend decomposition (trend analizi)
    - Sezonallik (günlük, haftalık, yıllık)
    - Missing values handled usinge ve aykırı değerlere dayanıklı
    - Otomatik hyperparameter tuning
    
    Hyperparameters:
    - changepoint_prior_scale: 0.05 (Trend değişim noktası duyarlılığı)
    - seasonality_scale: 10 (Sezonallik gücü)
    - seasonality_mode: 'additive' (Trend + Sezonallik)
    """
    
    def __init__(self, interval_width: float = 0.95, seasonality_mode: str = 'additive'):
        """
        ProphetForecaster Initialize.
        
        Args:
            interval_width (float): Tahmin aralığı genişliği (default: 0.95 → %95)
            seasonality_mode (str): 'additive' veya 'multiplicative'
        """
        super().__init__(name="ProphetForecaster")
        self.interval_width = interval_width
        self.seasonality_mode = seasonality_mode
        self.model = Prophet(
            interval_width=interval_width,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=0.05,
            seasonality_scale=10,
            yearly_seasonality=False,  # Günlük ve haftalık sezonallik yeterli
            weekly_seasonality=True,
            daily_seasonality=True
        )
        
        self.logger.info(f"ProphetForecaster initialized | "
                       f"Interval width: {interval_width}, "
                       f"Seasonality mode: {seasonality_mode}")
    
    def fit(self, X_train: pd.DataFrame, y_train: Optional[np.ndarray] = None, **kwargs):
        """
        Prophet modelini eğit.
        
        Args:
            X_train (pd.DataFrame): 'ds' (datetime) ve 'y' (value) sütunları içermeli
            y_train: Kullanılmaz
            
        NOT: X_train DataFrame olmalı ve şu sütunları içermeli:
             - 'ds': Zaman (datetime)
             - 'y': Değer (tüketim)
        """
        try:
            if not isinstance(X_train, pd.DataFrame):
                raise ValueError("X_train DataFrame olmalı ve 'ds' ve 'y' sütunlarını içermeli")
            
            if 'ds' not in X_train.columns or 'y' not in X_train.columns:
                raise ValueError("DataFrame 'ds' (datetime) ve 'y' (value) sütunlarını içermeli")
            
            # Train model (Prophet suppress_stdout parameter'ı kullanır)
            with open('/dev/null', 'w') as devnull:
                import sys
                old_stdout = sys.stdout
                sys.stdout = devnull
                self.model.fit(X_train)
                sys.stdout = old_stdout
            
            self.is_trained = True
            
            self.logger.info(f"Prophet eğitimi tamamlandı | Train boyutu: {len(X_train)}")
            
        except Exception as e:
            self.logger.error(f"Training errorsı: {str(e)}")
            raise
    
    def forecast(self, steps: int = 24, freq: str = 'h') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prophet ile tahmin yap.
        
        Args:
            steps (int): Kaç adım ileri tahmin et
            freq (str): Frekans ('h' = saat, 'd' = gün)
            
        Returns:
            Tuple: (forecast, lower_bound, upper_bound) - Tahmin ve güven aralıkları
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        try:
            # Gelecek tarihleri oluştur
            future = self.model.make_future_dataframe(periods=steps, freq=freq)
            
            # Make predictions
            forecast_df = self.model.predict(future)
            
            # Son 'steps' satırı al
            forecast_df = forecast_df.tail(steps)
            
            forecast = forecast_df['yhat'].values
            lower_bound = forecast_df['yhat_lower'].values
            upper_bound = forecast_df['yhat_upper'].values
            
            self.logger.info(f"Prophet tahminlemesi: {steps} adım")
            return forecast, lower_bound, upper_bound
            
        except Exception as e:
            self.logger.error(f"Tahmin errorsı: {str(e)}")
            raise
    
    def save(self, filepath: str):
        """Modeli joblib formatında kaydet."""
        try:
            joblib.dump({
                'model': self.model,
                'interval_width': self.interval_width,
                'seasonality_mode': self.seasonality_mode
            }, filepath)
            self.logger.info(f"Model saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Model kayıt errorsı: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """Modeli joblib formatından yükle."""
        try:
            loaded = joblib.load(filepath)
            self.model = loaded['model']
            self.interval_width = loaded['interval_width']
            self.seasonality_mode = loaded['seasonality_mode']
            self.is_trained = True
            self.logger.info(f"Model loaded: {filepath}")
        except Exception as e:
            self.logger.error(f"Model yükleme errorsı: {str(e)}")
            raise


# Test code
if __name__ == "__main__":
    print("Forecasting Module - Test")
    print("=" * 50)
    
    # Örnek zaman serisi oluştur (güneş döngüsü + gürültü)
    np.random.seed(42)
    t = np.arange(0, 100, 1)
    y = 50 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 5, len(t))
    
    print(f"\nTime series boyutu: {len(y)}")
    
    # Train/Test split
    split_idx = int(0.8 * len(y))
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train boyutu: {len(y_train)}, Test boyutu: {len(y_test)}")
    
    # Linear Forecaster Test
    print("\n--- Linear Forecaster ---")
    lf = LinearForecaster(lookback=12, degree=1)
    lf.fit(y_train)
    forecast_lf = lf.forecast(steps=24, last_sequence=y_train[-12:])
    print(f"Tahmin (ilk 5): {forecast_lf[:5]}")
    
    # ARIMA Test
    print("\n--- ARIMA Forecaster ---")
    af = ARIMAForecaster(order=(1, 1, 1))
    af.fit(y_train)
    forecast_af = af.forecast(steps=24)
    print(f"Tahmin (ilk 5): {forecast_af[:5]}")
    
    # Prophet Test
    print("\n--- Prophet Forecaster ---")
    df_prophet = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=len(y_train), freq='h'),
        'y': y_train
    })
    pf = ProphetForecaster(interval_width=0.95, seasonality_mode='additive')
    pf.fit(df_prophet)
    forecast_pf, lower_pf, upper_pf = pf.forecast(steps=24, freq='h')
    print(f"Tahmin (ilk 5): {forecast_pf[:5]}")
