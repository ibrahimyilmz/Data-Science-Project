"""
Classification Module - Sınıflandırma Modelleri (RS/RP)
========================================================
Evlerin birincil (RP) veya ikincil (RS) konut olup olmadığını
sınıflandıran modelller içerir.

Modeller:
    1. BaselineClassifier: Hızlı başlangıç için Logistic Regression
    2. DeepClassifier: PyTorch tabanlı derin sinir ağı
    
Hiperparametreler:
    - Logistic Regression: C=1.0, solver='lbfgs', max_iter=1000
    - PyTorch NN: 3 katman, ReLU aktivasyon, Dropout=0.3
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from typing import Tuple, Dict, Optional, List
from abc import ABC, abstractmethod

# Logger konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """
    Tüm sınıflandırıcıların temel sınıfı.
    
    Ortak metodlar:
    - train(): Modeli eğit
    - predict(): Tahmin yap
    - evaluate(): Modeli değerlendir
    - save(): Modeli kaydet
    - load(): Modeli yükle
    """
    
    def __init__(self, name: str = "BaseClassifier"):
        self.name = name
        self.model = None
        self.is_trained = False
        self.logger = logger
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Modeli eğit (abstract metod)"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Tahmin yap (abstract metod)"""
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Modeli değerlendir ve metrikleri hesapla.
        
        Returns:
            Dict: Accuracy, Precision, Recall, F1 skorları
        """
        try:
            if not self.is_trained:
                raise ValueError("Model henüz eğitilmedi")
            
            y_pred = self.predict(X_test)
            
            # String etiketleri encode et
            if isinstance(y_test[0], str):
                label_map = {label: idx for idx, label in enumerate(np.unique(y_test))}
                y_test_encoded = np.array([label_map[y] for y in y_test])
                y_pred_encoded = np.array([label_map.get(y, 0) for y in y_pred])
            else:
                y_test_encoded = y_test
                y_pred_encoded = y_pred
            
            metrics = {
                'accuracy': accuracy_score(y_test_encoded, y_pred_encoded),
                'precision': precision_score(y_test_encoded, y_pred_encoded, average='weighted'),
                'recall': recall_score(y_test_encoded, y_pred_encoded, average='weighted'),
                'f1': f1_score(y_test_encoded, y_pred_encoded, average='weighted')
            }
            
            self.logger.info(f"{self.name} Değerlendirme Sonuçları:")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Değerlendirme hatası: {str(e)}")
            raise
    
    @abstractmethod
    def save(self, filepath: str):
        """Modeli kaydet"""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Modeli yükle"""
        pass


class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression tabanlı sınıflandırıcı.
    
    Özellikler:
    - Hızlı eğitim ve tahmin
    - Baseline model olarak kullanım
    - Explainability (Açıklanabilirlik) yüksek
    
    Hiperparametreler:
    - C: Regularizasyon gücü (default: 1.0)
    - solver: Optimizasyon algoritması (default: 'lbfgs')
    - max_iter: Maximum iterasyon sayısı (default: 1000)
    """
    
    def __init__(self, C: float = 1.0, solver: str = 'lbfgs', max_iter: int = 1000):
        """
        LogisticRegressionClassifier başlatma.
        
        Args:
            C (float): Regularizasyon gücü (C değeri küçüldükçe daha kuvvetli regularizasyon)
            solver (str): Optimizasyon algoritması
            max_iter (int): Maximum iterasyon sayısı
        """
        super().__init__(name="LogisticRegression")
        self.model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
        self.label_mapping = None
        
        self.logger.info(f"LogisticRegression başlatıldı | C={C}, solver={solver}, max_iter={max_iter}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Modeli eğit.
        
        Args:
            X_train (np.ndarray): Eğitim özellikleri (n_samples x n_features)
            y_train (np.ndarray): Eğitim etiketleri
            **kwargs: Ekstra parametreler (kullanılmaz)
        """
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            self.label_mapping = {label: idx for idx, label in enumerate(np.unique(y_train))}
            
            self.logger.info(f"LogisticRegression eğitimi tamamlandı | Train boyutu: {X_train.shape[0]}")
            
        except Exception as e:
            self.logger.error(f"Eğitim hatası: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin yap.
        
        Args:
            X (np.ndarray): Tahmin özellikleri
            
        Returns:
            np.ndarray: Tahmin edilen etiketler
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin olasılıklarını hesapla.
        
        Args:
            X (np.ndarray): Tahmin özellikleri
            
        Returns:
            np.ndarray: Her sınıf için olasılıklar
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """Modeli joblib formatında kaydet."""
        try:
            joblib.dump(self.model, filepath)
            self.logger.info(f"Model kaydedildi: {filepath}")
        except Exception as e:
            self.logger.error(f"Model kayıt hatası: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """Modeli joblib formatından yükle."""
        try:
            self.model = joblib.load(filepath)
            self.is_trained = True
            self.logger.info(f"Model yüklendi: {filepath}")
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {str(e)}")
            raise


class NeuralNetworkClassifier(BaseClassifier):
    """
    PyTorch tabanlı Derin Sinir Ağı sınıflandırıcısı.
    
    Mimari:
    - Input Layer: n_features girişi
    - Hidden Layer 1: 128 nöron + ReLU + Dropout(0.3)
    - Hidden Layer 2: 64 nöron + ReLU + Dropout(0.3)
    - Hidden Layer 3: 32 nöron + ReLU
    - Output Layer: 2 sınıf (RS/RP) - Softmax ile
    
    Eğitim:
    - Loss: CrossEntropyLoss
    - Optimizer: Adam (lr=0.001)
    - Batch Size: 32
    - Epochs: 50-100
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = None, 
                 dropout_rate: float = 0.3, learning_rate: float = 0.001):
        """
        NeuralNetworkClassifier başlatma.
        
        Args:
            input_size (int): Giriş özellik sayısı
            hidden_sizes (list): Hidden layer boyutları (default: [128, 64, 32])
            dropout_rate (float): Dropout oranı
            learning_rate (float): Öğrenme oranı
        """
        super().__init__(name="NeuralNetworkClassifier")
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model mimarisi
        self.model = self._build_network(input_size, hidden_sizes, dropout_rate)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []
        
        self.logger.info(f"NeuralNetworkClassifier başlatıldı | "
                        f"Input: {input_size}, Hidden: {hidden_sizes}, "
                        f"Device: {self.device}, LR: {learning_rate}")
    
    def _build_network(self, input_size: int, hidden_sizes: List[int], 
                      dropout_rate: float) -> nn.Module:
        """
        Sinir ağını oluştur.
        
        Args:
            input_size (int): Giriş boyutu
            hidden_sizes (list): Hidden layer boyutları
            dropout_rate (float): Dropout oranı
            
        Returns:
            nn.Module: PyTorch sinir ağı
        """
        layers = []
        
        # İlk gizli layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Ara gizli layerler
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 2:  # Son layerden önce dropout
                layers.append(nn.Dropout(dropout_rate))
        
        # Çıktı layer (2 sınıf için)
        layers.append(nn.Linear(hidden_sizes[-1], 2))
        
        return nn.Sequential(*layers)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             epochs: int = 50, batch_size: int = 32, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Modeli eğit.
        
        Args:
            X_train (np.ndarray): Eğitim özellikleri
            y_train (np.ndarray): Eğitim etiketleri
            epochs (int): Eğitim devresi sayısı
            batch_size (int): Batch boyutu
            X_val (np.ndarray, optional): Doğrulama özellikleri
            y_val (np.ndarray, optional): Doğrulama etiketleri
            
        NOT: Hiperparametreler seçiminde dikkat edildi:
            - epochs=50: Overfitting'e karşı koruma
            - batch_size=32: Gradyan güncellemesinin düzgün olması
            - Dropout=0.3: Overfitting'i kontrol etmek için
        """
        try:
            # String etiketleri sayıya dönüştür
            if isinstance(y_train[0], str):
                unique_labels = np.unique(y_train)
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                y_train_encoded = np.array([label_map[y] for y in y_train])
                if y_val is not None:
                    y_val_encoded = np.array([label_map[y] for y in y_val])
            else:
                y_train_encoded = y_train
                y_val_encoded = y_val if y_val is not None else None
            
            # PyTorch tensor'e dönüştür
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Doğrulama seti varsa
            if X_val is not None and y_val is not None:
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Eğitim döngüsü
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(train_loader)
                self.train_losses.append(avg_train_loss)
                
                # Doğrulama
                if X_val is not None:
                    self.model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            outputs = self.model(batch_X)
                            loss = self.criterion(outputs, batch_y)
                            val_loss += loss.item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    self.val_losses.append(avg_val_loss)
                    
                    if (epoch + 1) % 10 == 0:
                        self.logger.info(f"Epoch [{epoch+1}/{epochs}] | "
                                       f"Train Loss: {avg_train_loss:.4f} | "
                                       f"Val Loss: {avg_val_loss:.4f}")
                    
                    self.model.train()
                else:
                    if (epoch + 1) % 10 == 0:
                        self.logger.info(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}")
            
            self.is_trained = True
            self.logger.info(f"NeuralNetworkClassifier eğitimi tamamlandı | "
                           f"Epochs: {epochs}, Train boyutu: {X_train.shape[0]}")
            
        except Exception as e:
            self.logger.error(f"Eğitim hatası: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin yap.
        
        Args:
            X (np.ndarray): Tahmin özellikleri
            
        Returns:
            np.ndarray: Tahmin edilen sınıflar (0 veya 1)
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin olasılıklarını hesapla.
        
        Args:
            X (np.ndarray): Tahmin özellikleri
            
        Returns:
            np.ndarray: Her sınıf için olasılıklar
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
        
        return probas.cpu().numpy()
    
    def save(self, filepath: str):
        """Modeli PyTorch formatında kaydet."""
        try:
            torch.save(self.model.state_dict(), filepath)
            self.logger.info(f"Model kaydedildi: {filepath}")
        except Exception as e:
            self.logger.error(f"Model kayıt hatası: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """Modeli PyTorch formatından yükle."""
        try:
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.model.to(self.device)
            self.is_trained = True
            self.logger.info(f"Model yüklendi: {filepath}")
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {str(e)}")
            raise


# Test kodu
if __name__ == "__main__":
    print("Classification Module - Test")
    print("=" * 50)
    
    # Örnek veri seti oluştur
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X_data = np.random.randn(n_samples, n_features)
    y_data = np.random.choice(['RS', 'RP'], n_samples)
    
    # Train/Test split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X_data[:split_idx], X_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
    print(f"\nEğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    
    # Logistic Regression Test
    print("\n--- Logistic Regression ---")
    lr_classifier = LogisticRegressionClassifier(C=1.0, solver='lbfgs', max_iter=1000)
    lr_classifier.train(X_train, y_train)
    lr_metrics = lr_classifier.evaluate(X_test, y_test)
    
    # Neural Network Test
    print("\n--- Neural Network ---")
    nn_classifier = NeuralNetworkClassifier(input_size=n_features, 
                                           hidden_sizes=[128, 64, 32],
                                           dropout_rate=0.3,
                                           learning_rate=0.001)
    nn_classifier.train(X_train, y_train, epochs=30, batch_size=16)
    nn_metrics = nn_classifier.evaluate(X_test, y_test)
