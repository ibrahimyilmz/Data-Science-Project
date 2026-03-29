"""
Model Preparation Module - Veri Hazırlığı ve Dengeleme
========================================================
Kişi 1 (Veri Mimarı) tarafından sağlanan etiketlenmiş veriyi,
Kişi 2'nin sınıflandırma ve tahminleme modelleri için hazırlar.

Fonksiyonlar:
    - balance_dataset(): Test setinde eşit RS/RP dağılımı sağlar
    - train_test_split_timeseries(): Zaman serisi yapısını korur
    - prepare_features(): Özellik vektorünü modele hazır hale getirir
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from typing import Tuple, Dict, Optional, List

# Logger konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Veri ön işleme ve hazırlama sınıfı.
    
    Özellikleri:
    - Dataset dengeleme (RS/RP eşit dağılımı)
    - Zaman serisi yapısını koruyarak train/test split
    - Eksik veri işleme
    - Özellik normalizasyonu
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        DataPreprocessor başlatma.
        
        Args:
            test_size (float): Test seti oranı (0-1)
            random_state (int): Reproducibility için seed değeri
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.logger = logger
        
    def balance_dataset(self, df: pd.DataFrame, label_col: str = 'label') -> pd.DataFrame:
        """
        Test veri setinde eşit sayıda RS ve RP örneği sağlamak için 
        veri setini dengeler.
        
        Yöntem: Minorite sınıfı oversampling, Majority sınıfı undersampling
        
        Args:
            df (pd.DataFrame): Orijinal veri seti
            label_col (str): Etiket sütun adı ('RS' veya 'RP' içerir)
            
        Returns:
            pd.DataFrame: Dengeli veri seti
        """
        try:
            # Sınıf dağılımını kontrol et
            class_distribution = df[label_col].value_counts()
            self.logger.info(f"Orijinal sınıf dağılımı:\n{class_distribution}")
            
            # Her sınıftan aynı sayıda örnek al
            min_samples = class_distribution.min()
            
            balanced_dfs = []
            for class_label in df[label_col].unique():
                class_df = df[df[label_col] == class_label]
                balanced_dfs.append(class_df.sample(n=min_samples, 
                                                     random_state=self.random_state))
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            balanced_df = balanced_df.sample(frac=1, 
                                            random_state=self.random_state).reset_index(drop=True)
            
            self.logger.info(f"Dengeli sınıf dağılımı:\n{balanced_df[label_col].value_counts()}")
            return balanced_df
            
        except Exception as e:
            self.logger.error(f"Dataset dengeleme hatası: {str(e)}")
            raise
    
    def train_test_split_timeseries(self, 
                                    df: pd.DataFrame, 
                                    time_col: Optional[str] = None,
                                    features_cols: Optional[List[str]] = None,
                                    label_col: str = 'label') -> Tuple:
        """
        Zaman serisi yapısını koruyarak train/test split yapar.
        
        Yöntem: TimeSeriesSplit kullanarak data leakage'ı önler
        
        Args:
            df (pd.DataFrame): Veri seti
            time_col (str, optional): Zaman sütun adı
            features_cols (list, optional): Özellik sütunları listesi
            label_col (str): Etiket sütun adı
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            if time_col:
                # Zaman sütununa göre sıralama
                df = df.sort_values(by=time_col).reset_index(drop=True)
                
            # Eğer features_cols belirtilmemişse tüm sayısal sütunları kullan
            if features_cols is None:
                features_cols = [col for col in df.columns 
                               if col not in [label_col, time_col] and df[col].dtype in [np.float64, np.int64]]
            
            X = df[features_cols].values
            y = df[label_col].values
            
            # Zaman serisi bölünmesi
            split_point = int(len(df) * (1 - self.test_size))
            
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            self.logger.info(f"Train seti boyutu: {X_train.shape[0]}")
            self.logger.info(f"Test seti boyutu: {X_test.shape[0]}")
            self.logger.info(f"Özellik sayısı: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test, features_cols
            
        except Exception as e:
            self.logger.error(f"Train/Test split hatası: {str(e)}")
            raise
    
    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray, 
                          method: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Özellik normalizasyonu uygular.
        
        Args:
            X_train (np.ndarray): Eğitim özelikleri
            X_test (np.ndarray): Test özellikleri
            method (str): Normalizasyon yöntemi ('standard' veya 'minmax')
            
        Returns:
            Tuple: (X_train_normalized, X_test_normalized)
        """
        try:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")
            
            X_train_normalized = self.scaler.fit_transform(X_train)
            X_test_normalized = self.scaler.transform(X_test)
            
            self.logger.info(f"{method.upper()} normalizasyonu uygulandı")
            return X_train_normalized, X_test_normalized
            
        except Exception as e:
            self.logger.error(f"Normalizasyon hatası: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Eksik verileri işler.
        
        Args:
            df (pd.DataFrame): Veri seti
            method (str): İşleme yöntemi ('interpolate', 'forward_fill', 'backward_fill')
            
        Returns:
            pd.DataFrame: Eksik veri işlenmiş veri seti
        """
        try:
            if method == 'interpolate':
                df_filled = df.interpolate(method='linear', limit_direction='both')
            elif method == 'forward_fill':
                df_filled = df.fillna(method='ffill').fillna(method='bfill')
            elif method == 'backward_fill':
                df_filled = df.fillna(method='bfill').fillna(method='ffill')
            else:
                raise ValueError(f"Bilinmeyen eksik veri işleme yöntemi: {method}")
            
            self.logger.info(f"Eksik veriler {method} yöntemi ile işlendi")
            return df_filled
            
        except Exception as e:
            self.logger.error(f"Eksik veri işleme hatası: {str(e)}")
            raise
    
    def remove_outliers(self, X: np.ndarray, method: str = 'iqr', threshold: float = 3.0) -> np.ndarray:
        """
        Aykırı değerleri tespit ve çıkarır.
        
        Args:
            X (np.ndarray): Veri (n_samples x n_features)
            method (str): Yöntem ('iqr' veya 'zscore')
            threshold (float): Eşik değeri
            
        Returns:
            np.ndarray: Aykırı değerler çıkarılmış veri
        """
        try:
            if method == 'iqr':
                Q1 = np.percentile(X, 25, axis=0)
                Q3 = np.percentile(X, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
                
            elif method == 'zscore':
                z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
                mask = np.all(z_scores < threshold, axis=1)
            else:
                raise ValueError(f"Bilinmeyen aykırı değer yöntemi: {method}")
            
            removed_count = X.shape[0] - mask.sum()
            self.logger.info(f"Aykırı değerler çıkarıldı: {removed_count} örnek")
            
            return X[mask]
            
        except Exception as e:
            self.logger.error(f"Aykırı değer çıkarma hatası: {str(e)}")
            raise


def create_lag_features(data: pd.DataFrame, 
                        value_col: str, 
                        lags: List[int] = [1, 7, 24]) -> pd.DataFrame:
    """
    Tahminleme modeli için gecikmeli (lag) özellikleri oluşturur.
    
    Örn: Dünün tüketimi (lag=24), Geçen haftanın (lag=7) vb.
    
    Args:
        data (pd.DataFrame): Zaman serisi veri seti
        value_col (str): Değer sütun adı
        lags (list): Oluşturulacak lag değerleri (saat cinsinden)
        
    Returns:
        pd.DataFrame: Lag özelikleri eklenen veri seti
    """
    try:
        df = data.copy()
        
        for lag in lags:
            df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
        
        # İlk satırların NaN değerlerini kaldır
        df = df.dropna()
        
        logger.info(f"Lag özelikleri oluşturuldu: {lags}")
        return df
        
    except Exception as e:
        logger.error(f"Lag özelikleri oluşturma hatası: {str(e)}")
        raise


# Test kodu
if __name__ == "__main__":
    print("Model Preparation Module - Test")
    print("=" * 50)
    
    # Örnek veri seti oluştur
    np.random.seed(42)
    n_samples = 1000
    
    # Sahte veri üret
    X_data = np.random.randn(n_samples, 10)
    y_data = np.random.choice(['RS', 'RP'], n_samples)
    
    df_test = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(10)])
    df_test['label'] = y_data
    df_test['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='h')
    
    print(f"\nOrijinal veri seti şekli: {df_test.shape}")
    print(f"Sınıf dağılımı:\n{df_test['label'].value_counts()}")
    
    # DataPreprocessor'ı test et
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    
    # Dataset dengeleme
    df_balanced = preprocessor.balance_dataset(df_test)
    print(f"\nDengeli veri seti şekli: {df_balanced.shape}")
    
    # Train/Test split
    X_train, X_test, y_train, y_test, feature_names = preprocessor.train_test_split_timeseries(
        df_balanced, 
        time_col='timestamp',
        label_col='label'
    )
    
    print(f"\nX_train şekli: {X_train.shape}")
    print(f"X_test şekli: {X_test.shape}")
    
    # Normalizasyon
    X_train_norm, X_test_norm = preprocessor.normalize_features(X_train, X_test, method='standard')
    print(f"\nX_train (normalized) ortalaması: {X_train_norm.mean():.4f}")
    print(f"X_train (normalized) standart sapması: {X_train_norm.std():.4f}")
