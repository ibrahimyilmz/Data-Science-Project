"""
Evaluator Module - Model Değerlendirme ve Metrikleme
====================================================
Sınıflandırma ve tahminleme modelleri için kapsamlı 
değerlendirme araçları sağlar.

İçerik:
    - Confusion Matrix görselleştirmesi
    - Sınıflandırma metrikleri (Precision, Recall, F1)
    - Tahminleme metrikleri (MAE, RMSE, R2)
    - Model karşılaştırma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score
)
import logging
from typing import Dict, List, Tuple, Optional, Union

# Logger konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """
    Sınıflandırma modelleri için değerlendirme sınıfı.
    
    Metrikler:
    - Confusion Matrix
    - Accuracy, Precision, Recall, F1
    - ROC-AUC
    - Classification Report
    """
    
    def __init__(self):
        self.metrics = {}
        self.logger = logger
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Sınıflandırma modelini değerlendir.
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred (np.ndarray): Tahmin edilen etiketler
            y_pred_proba (np.ndarray, optional): Tahmin olasılıkları (ROC-AUC için)
            
        Returns:
            Dict: Tüm metrikler
        """
        try:
            # String etiketleri encode et
            if isinstance(y_true[0], str):
                unique_labels = np.unique(y_true)
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                y_true_encoded = np.array([label_map[y] for y in y_true])
                y_pred_encoded = np.array([label_map.get(y, 0) for y in y_pred])
            else:
                y_true_encoded = y_true
                y_pred_encoded = y_pred
            
            # Temel metrikler
            accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
            precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
            recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
            f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
            
            # Confusion Matrix
            cm = confusion_matrix(y_true_encoded, y_pred_encoded)
            
            self.metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'y_true': y_true,
                'y_pred': y_pred
            }
            
            # ROC-AUC (binary classification için)
            if len(np.unique(y_true_encoded)) == 2 and y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_true_encoded, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                self.metrics['roc_auc'] = roc_auc
                self.metrics['fpr'] = fpr
                self.metrics['tpr'] = tpr
            
            self.logger.info(f"Sınıflandırma Değerlendirmesi:")
            self.logger.info(f"  Accuracy: {accuracy:.4f}")
            self.logger.info(f"  Precision: {precision:.4f}")
            self.logger.info(f"  Recall: {recall:.4f}")
            self.logger.info(f"  F1: {f1:.4f}")
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Değerlendirme hatası: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, labels: Optional[List[str]] = None, 
                             figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Confusion Matrix'i görselleştir.
        
        Args:
            labels (list, optional): Etiket isimleri
            figsize (tuple): Şekil boyutu
            
        Returns:
            plt.Figure: Matplotlib figure nesnesi
        """
        if 'confusion_matrix' not in self.metrics:
            raise ValueError("Önce evaluate() metodu çağrılmalı")
        
        try:
            cm = self.metrics['confusion_matrix']
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Heatmap çiz
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=labels, yticklabels=labels)
            
            ax.set_xlabel('Tahmin Edilen')
            ax.set_ylabel('Gerçek')
            ax.set_title('Confusion Matrix')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Görselleştirme hatası: {str(e)}")
            raise
    
    def plot_roc_curve(self, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        ROC eğrisini görselleştir.
        
        Args:
            figsize (tuple): Şekil boyutu
            
        Returns:
            plt.Figure: Matplotlib figure nesnesi
        """
        if 'roc_auc' not in self.metrics:
            raise ValueError("ROC-AUC metrikleri mevcut değil (binary classification gerekli)")
        
        try:
            fpr = self.metrics['fpr']
            tpr = self.metrics['tpr']
            roc_auc = self.metrics['roc_auc']
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc="lower right")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Görselleştirme hatası: {str(e)}")
            raise
    
    def get_classification_report(self, labels: Optional[List[str]] = None) -> str:
        """
        Detaylı sınıflandırma raporu al.
        
        Args:
            labels (list, optional): Etiket isimleri
            
        Returns:
            str: Rapor metni
        """
        if 'y_true' not in self.metrics:
            raise ValueError("Önce evaluate() metodu çağrılmalı")
        
        try:
            y_true = self.metrics['y_true']
            y_pred = self.metrics['y_pred']
            
            report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Rapor oluşturma hatası: {str(e)}")
            raise


class ForecastingEvaluator:
    """
    Tahminleme modelleri için değerlendirme sınıfı.
    
    Metrikler:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R2 Score
    """
    
    def __init__(self):
        self.metrics = {}
        self.logger = logger
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Tahminleme modelini değerlendir.
        
        Args:
            y_true (np.ndarray): Gerçek değerler
            y_pred (np.ndarray): Tahmin edilen değerler
            
        Returns:
            Dict: MAE, RMSE, MAPE, R2 metrikleri
        """
        try:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            # Sıfıra bölmeden kaçınmak için epsilon ekle
            epsilon = np.finfo(float).eps
            mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
            
            self.metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'y_true': y_true,
                'y_pred': y_pred
            }
            
            self.logger.info(f"Tahminleme Değerlendirmesi:")
            self.logger.info(f"  MAE: {mae:.4f}")
            self.logger.info(f"  RMSE: {rmse:.4f}")
            self.logger.info(f"  MAPE: {mape:.2f}%")
            self.logger.info(f"  R2: {r2:.4f}")
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Değerlendirme hatası: {str(e)}")
            raise
    
    def plot_predictions(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Gerçek ve tahmin edilen değerleri karşılaştırmalı çiz.
        
        Args:
            figsize (tuple): Şekil boyutu
            
        Returns:
            plt.Figure: Matplotlib figure nesnesi
        """
        if 'y_true' not in self.metrics:
            raise ValueError("Önce evaluate() metodu çağrılmalı")
        
        try:
            y_true = self.metrics['y_true']
            y_pred = self.metrics['y_pred']
            
            fig, ax = plt.subplots(figsize=figsize)
            
            time_index = np.arange(len(y_true))
            
            ax.plot(time_index, y_true, label='Gerçek', marker='o', linewidth=2)
            ax.plot(time_index, y_pred, label='Tahmin', marker='s', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Zaman')
            ax.set_ylabel('Değer')
            ax.set_title('Gerçek vs Tahmin Edilen Değerler')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Görselleştirme hatası: {str(e)}")
            raise
    
    def plot_residuals(self, figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Hataları (residuals) görselleştir.
        
        Args:
            figsize (tuple): Şekil boyutu
            
        Returns:
            plt.Figure: Matplotlib figure nesnesi
        """
        if 'y_true' not in self.metrics:
            raise ValueError("Önce evaluate() metodu çağrılmalı")
        
        try:
            y_true = self.metrics['y_true']
            y_pred = self.metrics['y_pred']
            residuals = y_true - y_pred
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Residuals zaman serisi
            axes[0].plot(residuals, marker='o', linestyle='-', linewidth=2)
            axes[0].axhline(y=0, color='r', linestyle='--')
            axes[0].set_xlabel('Zaman')
            axes[0].set_ylabel('Hata (Residual)')
            axes[0].set_title('Tahminleme Hataları')
            axes[0].grid(True, alpha=0.3)
            
            # Residuals dağılımı
            axes[1].hist(residuals, bins=30, edgecolor='black')
            axes[1].set_xlabel('Hata')
            axes[1].set_ylabel('Frekans')
            axes[1].set_title('Hata Dağılımı')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Görselleştirme hatası: {str(e)}")
            raise
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Performans özetini DataFrame olarak döndür.
        
        Returns:
            pd.DataFrame: Metrikler tablosu
        """
        if not self.metrics:
            raise ValueError("Önce evaluate() metodu çağrılmalı")
        
        try:
            summary = pd.DataFrame({
                'Metrik': ['MAE', 'RMSE', 'MAPE (%)', 'R2'],
                'Değer': [
                    self.metrics['mae'],
                    self.metrics['rmse'],
                    self.metrics['mape'],
                    self.metrics['r2']
                ]
            })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Özet oluşturma hatası: {str(e)}")
            raise


class ModelComparator:
    """
    Farklı modelleri karşılaştırma sınıfı.
    
    Özellikler:
    - Birden fazla modelin metrikleri
    - Modeller arası karşılaştırma
    - Performans raporları
    """
    
    def __init__(self):
        self.results = {}
        self.logger = logger
    
    def add_result(self, model_name: str, metrics: Dict[str, float]):
        """
        Model sonuçlarını ekle.
        
        Args:
            model_name (str): Model adı
            metrics (dict): Metrikler
        """
        self.results[model_name] = metrics
        self.logger.info(f"'{model_name}' modeli sonuçlara eklendi")
    
    def compare_forecasting_models(self) -> pd.DataFrame:
        """
        Tahminleme modellerini karşılaştır.
        
        Returns:
            pd.DataFrame: Karşılaştırma tablosu
        """
        if not self.results:
            raise ValueError("Karşılaştırılacak sonuç yok")
        
        try:
            comparison_data = {
                'Model': [],
                'MAE': [],
                'RMSE': [],
                'R2': []
            }
            
            for model_name, metrics in self.results.items():
                comparison_data['Model'].append(model_name)
                comparison_data['MAE'].append(metrics.get('mae', np.nan))
                comparison_data['RMSE'].append(metrics.get('rmse', np.nan))
                comparison_data['R2'].append(metrics.get('r2', np.nan))
            
            comparison_df = pd.DataFrame(comparison_data)
            
            self.logger.info("Model Karşılaştırması:")
            self.logger.info(f"\n{comparison_df.to_string()}")
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Karşılaştırma hatası: {str(e)}")
            raise
    
    def compare_classification_models(self) -> pd.DataFrame:
        """
        Sınıflandırma modellerini karşılaştır.
        
        Returns:
            pd.DataFrame: Karşılaştırma tablosu
        """
        if not self.results:
            raise ValueError("Karşılaştırılacak sonuç yok")
        
        try:
            comparison_data = {
                'Model': [],
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1': []
            }
            
            for model_name, metrics in self.results.items():
                comparison_data['Model'].append(model_name)
                comparison_data['Accuracy'].append(metrics.get('accuracy', np.nan))
                comparison_data['Precision'].append(metrics.get('precision', np.nan))
                comparison_data['Recall'].append(metrics.get('recall', np.nan))
                comparison_data['F1'].append(metrics.get('f1', np.nan))
            
            comparison_df = pd.DataFrame(comparison_data)
            
            self.logger.info("Model Karşılaştırması:")
            self.logger.info(f"\n{comparison_df.to_string()}")
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Karşılaştırma hatası: {str(e)}")
            raise
    
    def plot_model_comparison(self, model_type: str = 'forecasting', 
                             metric: str = 'mae', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Model karşılaştırmasını görselleştir.
        
        Args:
            model_type (str): 'forecasting' veya 'classification'
            metric (str): Karşılaştırılacak metrik
            figsize (tuple): Şekil boyutu
            
        Returns:
            plt.Figure: Matplotlib figure nesnesi
        """
        try:
            models = list(self.results.keys())
            values = [self.results[m].get(metric, np.nan) for m in models]
            
            fig, ax = plt.subplots(figsize=figsize)
            
            bars = ax.bar(models, values, color='steelblue', edgecolor='black')
            
            # En iyi performans için renk değiştir
            if metric in ['mae', 'rmse', 'mape']:
                best_idx = np.argmin(values)  # Düşük değer iyi
            else:
                best_idx = np.argmax(values)  # Yüksek değer iyi
            
            bars[best_idx].set_color('green')
            
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Model Karşılaştırması - {metric.upper()}')
            ax.grid(True, alpha=0.3, axis='y')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Görselleştirme hatası: {str(e)}")
            raise


# Test kodu
if __name__ == "__main__":
    print("Evaluator Module - Test")
    print("=" * 50)
    
    # Sınıflandırma test
    print("\n--- Classification Evaluator Test ---")
    np.random.seed(42)
    y_true = np.random.choice([0, 1], 100)
    y_pred = np.random.choice([0, 1], 100)
    
    clf_eval = ClassificationEvaluator()
    clf_metrics = clf_eval.evaluate(y_true, y_pred)
    print(f"Accuracy: {clf_metrics['accuracy']:.4f}")
    print(f"F1: {clf_metrics['f1']:.4f}")
    
    # Tahminleme test
    print("\n--- Forecasting Evaluator Test ---")
    y_true_fcst = np.random.randn(50) * 10 + 100
    y_pred_fcst = y_true_fcst + np.random.randn(50) * 5
    
    fcst_eval = ForecastingEvaluator()
    fcst_metrics = fcst_eval.evaluate(y_true_fcst, y_pred_fcst)
    print(f"MAE: {fcst_metrics['mae']:.4f}")
    print(f"RMSE: {fcst_metrics['rmse']:.4f}")
    
    # Model Comparator test
    print("\n--- Model Comparator Test ---")
    comparator = ModelComparator()
    comparator.add_result('Model A', fcst_metrics)
    
    # Farklı metrikleri simüle et
    fcst_metrics_b = fcst_eval.evaluate(y_true_fcst + 10, y_pred_fcst)
    comparator.add_result('Model B', fcst_metrics_b)
    
    comparison_df = comparator.compare_forecasting_models()
    print(f"\nKarşılaştırma:\n{comparison_df}")
