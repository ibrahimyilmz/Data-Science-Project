# Data Science Project - Energy Consumption Analysis and Forecasting

**Team Division:**
- **Person 1**: Data Architect and Clustering Expert
- **Person 2**: Model Architect (Classification & Forecasting) ← **You are here**
- **Person 3**: Integrator and Generative AI Expert

---

## 📋 Project Overview

By analyzing energy consumption data from Enedis:
1. **Classify houses**: Determine whether they are primary (RP) or secondary (RS) residences
2. **Forecast future consumption**: Predict energy consumption 24 hours ahead
3. **Interactive Dashboard**: Provide a Streamlit-based user interface

---

## 🏗️ Project Structure

```
Data-Science-Project/
├── src/                          # Source code
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

### 2. Kütüphaneleri Yükle

```bash
pip install -r requirements.txt
```

**Temel Kütüphaneler:**
- `numpy`, `pandas`: Veri işleme
- `scikit-learn`: Makine öğrenmesi
- `torch`: Derin öğrenme (PyTorch)
- `statsmodels`: Zaman serisi (ARIMA)
- `prophet`: Tahminleme (Facebook Prophet)
- `joblib`: Model kayıt/yükleme

---

## 📚 Person 2 (Sizin) Görevleriniz

### 1. **Model Preparation** (`src/model_prep.py`)

Person 1'den gelen veriyi eğitim için hazırlar.

#### Sınıflar:
- **`DataPreprocessor`**: Veri ön işleme
  - `balance_dataset()`: Test setinde RS/RP dengeleme
  - `train_test_split_timeseries()`: Zaman serisi bölünmesi
  - `normalize_features()`: Özellikleri ölçeklendirme
  - `handle_missing_values()`: Eksik veri işleme
  - `remove_outliers()`: Aykırı değer çıkarma

#### Kullanım Örneği:
```python
from src.model_prep import DataPreprocessor

preprocessor = DataPreprocessor(test_size=0.2, random_state=42)

# Veri dengeleme
df_balanced = preprocessor.balance_dataset(df, label_col='label')

# Train/Test split
X_train, X_test, y_train, y_test, feature_names = preprocessor.train_test_split_timeseries(
    df_balanced,
    time_col='timestamp',
    label_col='label'
)

# Normalizasyon
X_train_norm, X_test_norm = preprocessor.normalize_features(X_train, X_test, method='standard')
```

---

### 2. **Classification** (`src/classification.py`)

Evleri RS (Secondary Residence) veya RP (Primary Residence) olarak sınıflandırır.

#### Sınıflar:
- **`BaseClassifier`** (Abstract): Temel sınıflandırıcı arayüzü

- **`LogisticRegressionClassifier`**: Hızlı baseline model
  - Hiperparametreler: `C=1.0, solver='lbfgs', max_iter=1000`
  - Avantaj: Hızlı, explainable

- **`NeuralNetworkClassifier`**: Derin sinir ağı (PyTorch)
  - Mimari: 128 → 64 → 32 → 2 nöron
  - Dropout: 0.3 (overfitting önleme)
  - Optimizer: Adam (lr=0.001)
  - Epochs: 50-100

#### Kullanım Örneği:
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

# Tahmin
predictions = lr_classifier.predict(X_test)
probabilities = lr_classifier.predict_proba(X_test)  # Olasılıklar
```

---

### 3. **Forecasting** (`src/forecasting.py`)

Gelecekteki enerji tüketimini tahmin eder.

#### Sınıflar:
- **`BaseForecaster`** (Abstract): Temel tahmin arayüzü

- **`LinearForecaster`**: Çok değişkenli linear regression
  - Lookback: 24 saat
  - Polinom desteği (opsiyonel)

- **`ARIMAForecaster`**: ARIMA (AutoRegressive Integrated Moving Average)
  - Order: (1, 1, 1) [p=1, d=1, q=1]
  - Güven aralıkları ile tahmin

- **`ProphetForecaster`**: Facebook Prophet
  - Trend + sezonallik analizi
  - Haftalık/günlük periyodisiteleri yakalar
  - % 95 güven aralıkları

#### Zaman Serisi Özellikleri:
- Lag özelikleri: `create_lag_features(data, lags=[1, 7, 24])`
  - Lag-1: Bir saatlik geçmiş
  - Lag-7: 7 saatlik geçmiş (haftalık ritim)
  - Lag-24: 24 saatlik geçmiş (günlük ritim)

#### Kullanım Örneği:
```python
from src.forecasting import LinearForecaster, ARIMAForecaster, ProphetForecaster

# Linear Forecaster
lf = LinearForecaster(lookback=24, degree=1)
lf.fit(y_train)
forecast = lf.forecast(steps=24, last_sequence=y_train[-24:])

# ARIMA
af = ARIMAForecaster(order=(1, 1, 1))
af.fit(y_train)  # pd.Series veya np.array
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

# Model kayıt
lf.save('models/linear_forecaster.pkl')
af.save('models/arima_forecaster.pkl')
pf.save('models/prophet_forecaster.pkl')
```

---

### 4. **Evaluator** (`src/evaluator.py`)

Model performansını değerlendirmek için araçlar.

#### Sınıflar:
- **`ClassificationEvaluator`**: Sınıflandırma metrikleri
  - Metrikler: Accuracy, Precision, Recall, F1, ROC-AUC
  - Görselleştirme: Confusion Matrix, ROC Curve

- **`ForecastingEvaluator`**: Tahminleme metrikleri
  - Metrikler: MAE, RMSE, MAPE, R²
  - Görselleştirme: Predictions vs Actuals, Residuals

- **`ModelComparator`**: Modeller arası karşılaştırma

#### Kullanım Örneği:
```python
from src.evaluator import ClassificationEvaluator, ForecastingEvaluator, ModelComparator

# Sınıflandırma Değerlendirme
clf_eval = ClassificationEvaluator()
metrics = clf_eval.evaluate(y_test, y_pred, y_pred_proba)

# Visualizations
fig_cm = clf_eval.plot_confusion_matrix(labels=['RS', 'RP'])
fig_roc = clf_eval.plot_roc_curve()

# Report
report = clf_eval.get_classification_report(labels=['RS', 'RP'])
print(report)

# Tahminleme Değerlendirme
fcst_eval = ForecastingEvaluator()
fcst_metrics = fcst_eval.evaluate(y_test, y_pred)

print(f"MAE: {fcst_metrics['mae']:.2f}")
print(f"RMSE: {fcst_metrics['rmse']:.2f}")
print(f"R2: {fcst_metrics['r2']:.4f}")

# Model Karşılaştırma
comparator = ModelComparator()
comparator.add_result('Model A', metrics_a)
comparator.add_result('Model B', metrics_b)

comparison_df = comparator.compare_forecasting_models()
fig = comparator.plot_model_comparison(model_type='forecasting', metric='mae')
```

---

### 5. **Integration Logic** (`src/integration_logic.py`)

Üç kişinin çalışmasını birleştiren entegrasyon katmanı.

#### Sınıflar:
- **`ModelIntegrator`**: Tüm modelleri yükle ve çalıştır
  - Modelleryi load et
  - Pipeline'ı koordine et
  - Sonuçları döndür

- **`DataPipeline`**: Person 1'in verisini hazırla
  - `process_person1_data()`: Etiketlenmiş veriyi işle
  - `process_timeseries_for_forecast()`: Zaman serisini hazırla

- **`ResultsFormatter`**: Sonuçları Streamlit'e hazırlayan format

#### Kullanım Örneği:
```python
from src.integration_logic import ModelIntegrator, DataPipeline, ResultsFormatter

# 1. Integrator başlatması
integrator = ModelIntegrator(models_dir='models')

# 2. Modelleri yükle
integrator.load_classification_model('models/lr_classifier.pkl', model_type='logistic')
integrator.load_forecasting_model('arima', 'models/arima_forecaster.pkl')
integrator.load_feature_scaler('models/scaler.pkl')

# 3. Person 1'in verisini işle
pipeline = DataPipeline()
X_features, y_labels, feature_names = pipeline.process_person1_data(df_from_person1)

# 4. Sınıflandırma yap
clf_results = integrator.classify_residence(X_features, feature_names)
clf_df = ResultsFormatter.format_classification_results(clf_results)

# 5. Tahminleme yap
timeseries = pipeline.process_timeseries_for_forecast(df_timeseries)
fcst_results = integrator.forecast_consumption(timeseries, steps=24, model_name='arima')
fcst_df = ResultsFormatter.format_forecast_results(fcst_results)

# 6. Streamlit'e gönder (Person 3 tarafından kullanılacak)
print(clf_df)
print(fcst_df)
```

---

## 📊 Veri Akışı

```
Person 1 (Ham Veri)
    ↓
    → Özellikleri çıkar (PCA, Fourier, vb.)
    → RS/RP etiketleri ekle
    ↓
LABELED DATA → Person 2 (Model Eğitimi)
                ↓
                1. DataPreprocessor: Dengeleme + Normalizasyon
                2. Classification: RS/RP Sınıflandırması
                3. Forecasting: 24 saat ileri tahminleme
                4. Evaluator: Performans değerlendirmesi
                5. Modelleri kaydet (models/)
                ↓
TRAINED MODELS → Person 3 (Streamlit Dashboard)
                ↓
                → İnteraktif UI (Tabs)
                → Dinamik Görselleştirme
                → Kullanıcı girişi
```

---

## 🚀 Hızlı Başlangıç

### Minimal Örnek

```python
# 1. Kütüphaneleri import et
from src.model_prep import DataPreprocessor
from src.classification import LogisticRegressionClassifier
from src.evaluator import ClassificationEvaluator

import numpy as np
import pandas as pd

# 2. Sahte veri oluştur (Person 1 gerçek veriyi sağlayacak)
np.random.seed(42)
X_data = np.random.randn(1000, 15)
y_data = np.random.choice(['RS', 'RP'], 1000)

df = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(15)])
df['label'] = y_data

# 3. Veriyi hazırla
prep = DataPreprocessor(test_size=0.2, random_state=42)
df_balanced = prep.balance_dataset(df)
X_train, X_test, y_train, y_test, features = prep.train_test_split_timeseries(df_balanced)
X_train, X_test = prep.normalize_features(X_train, X_test)

# 4. Model eğit
clf = LogisticRegressionClassifier(C=1.0)
clf.train(X_train, y_train)

# 5. Değerlendir
evaluator = ClassificationEvaluator()
metrics = evaluator.evaluate(y_test, clf.predict(X_test))

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")

# 6. Modeli kaydet
clf.save('models/classifier.pkl')
```

---

## 📝 Teknik Notlar

### Hiperparametreler (Optimize Edildi)

| Model | Parametre | Değer | Neden |
|-------|-----------|-------|-------|
| Logistic Regression | C | 1.0 | Varsayılan regularizasyon |
| | solver | lbfgs | Binary classification için uygun |
| | max_iter | 1000 | Convergence güvencesi |
| Neural Network | Hidden Sizes | [128, 64, 32] | Progressive bottleneck |
| | Dropout | 0.3 | Overfitting önleme |
| | Learning Rate | 0.001 | Stabil öğrenme |
| | Epochs | 50 | Overfitting dengesi |
| | Batch Size | 32 | Memory/convergence dengesi |
| ARIMA | Order | (1,1,1) | AIC/BIC optimizasyon sonucu |
| Prophet | Changepoint Prior | 0.05 | Sensible trend changes |
| | Seasonality Scale | 10 | Güçlü sezonallik |

### Dataset Dengeleme Yöntemi

```python
# Problemo: Train seti R = 70%, RP = 30% (dengesiz)
# Çözüm: Undersampling (minority class align to minority)

Train: 1000 RS, 400 RP → Balanced: 400 RS, 400 RP (eğitim)
Test: 200 RS, 100 RP → Balanced: 100 RS, 100 RP (değerlendirme)
```

### Zaman Serisi Validation

⚠️ **ÖNEMLI**: Train/Test split yapılırken data leakage'dan kaçınılır:
- Kronolojik sıra korunur
- TimeSeriesSplit kullanılır
- Gelecek verileri training'e dahil edilmez

---

## 🔍 Debugging ve Hata Ayıklama

### Sık Karşılaşılan Hatalar

| Hata | Çözüm |
|------|-------|
| `Model henüz eğitilmedi` | `.train()` veya `.fit()` çağrısından emin ol |
| `Shape mismatch` | Train ve test özellikleri sayısını kontrol et |
| `Out of Memory` | Batch size düşür, veriyi küçült |
| `NaN değerler` | `handle_missing_values()` çağır, outlier kontrol et |

### Log Dosyaları

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
```

---

## 📚 Kaynaklar

- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Statsmodels**: https://www.statsmodels.org/
- **Facebook Prophet**: https://facebook.github.io/prophet/

---

## 👥 Ekip İletişimi

### Person 1 → Person 2
- **Format**: `data/labeled_data.csv` veya `data/person1_features.pkl`
- **Sütunlar**: Feature_1 ... Feature_N + label (RS/RP)
- **Beklenti**: Etiketlenmiş, normalize edilmemiş veri

### Person 2 → Person 3
- **Format**: Trained models in `models/` directory
- **Interface**: `src/integration_logic.py` (ModelIntegrator sınıfı)
- **Output**: Classification results + Forecasting predictions

---

## 📞 İletişim Bilgileri

Soru ve problem raporları için:
- GitHub Issues
- Email: [proje-email]
- Team Chat: [slack/discord]

---

**Son Güncelleme**: 29 Mart 2026

**Versiyon**: 1.0 (Person 2 Modelleri)
