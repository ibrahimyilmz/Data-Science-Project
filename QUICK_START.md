# QUICK START - Kişi 2 (Model Mimarı) İçin Hızlı Başlangıç

## 🚀 5 Dakika İçinde Başlayın

### Step 1: Kurulum

```bash
# Virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Kütüphaneleri yükle
pip install -r requirements.txt
```

### Step 2: Modülleri Test Et

```bash
python test_models.py
```

Çıktı:
```
✓ Tüm modüller başarıyla import edildi
...
🎉 Tüm testler başarılı!
```

### Step 3: Eksiksiz Eğitim Döngüsünü Çalıştır

```bash
python train_all_models.py
```

Bu script:
1. Sahte veri oluşturur (Kişi 1'in verisi geldiğinde değiştirilecek)
2. Veriyi dengeleyip normalize eder
3. **2 sınıflandırma modeli** eğitir:
   - Logistic Regression
   - Neural Network
4. **3 tahminleme modeli** eğitir:
   - Linear Forecaster
   - ARIMA
   - Prophet
5. Tüm modelleri `models/` dizinine kaydeder
6. `training_report.json` raporu oluşturur

---

## 📚 Temel Kullanım Örnekleri

### Örnek 1: Sınıflandırma Modeli

```python
from src.classification import LogisticRegressionClassifier
import numpy as np

# Veri oluştur
X_train = np.random.randn(100, 10)
y_train = np.random.choice(['RS', 'RP'], 100)

# Model oluştur ve eğit
clf = LogisticRegressionClassifier(C=1.0)
clf.train(X_train, y_train)

# Tahmin yap
X_test = np.random.randn(20, 10)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print(predictions)  # ['RS', 'RP', 'RS', ...]
print(probabilities)  # [[0.8, 0.2], [0.3, 0.7], ...]

# Model kaydet
clf.save('models/my_classifier.pkl')
```

### Örnek 2: Tahminleme Modeli

```python
from src.forecasting import ARIMAForecaster
import numpy as np

# Zaman serisi veri
timeseries = np.random.randn(100) * 10 + 100

# Model oluştur ve eğit
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(timeseries)

# Tahmin yap
forecast = model.forecast(steps=24)
print(forecast)  # [98.5, 99.2, 100.1, ...]

# Güven aralıkları
lower, upper = model.get_confidence_intervals(steps=24, alpha=0.05)
```

### Örnek 3: Model Değerlendirmesi

```python
from src.evaluator import ClassificationEvaluator
import numpy as np

y_true = np.array(['RS', 'RP', 'RS', 'RP'])
y_pred = np.array(['RS', 'RP', 'RS', 'RS'])

evaluator = ClassificationEvaluator()
metrics = evaluator.evaluate(y_true, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

---

## 📂 Dosya Rehberi

```
Project/
├── src/
│   ├── model_prep.py          ← Veri hazırlığı
│   ├── classification.py      ← Sınıflandırma modelleri
│   ├── forecasting.py         ← Tahminleme modelleri
│   ├── evaluator.py           ← Performans değerlendirmesi
│   └── integration_logic.py   ← Entegrasyon
├── models/                     ← Eğitilmiş modeller buraya kaydedilir
├── data/                       ← Veri setleri buraya konur
├── logs/                       ← Log dosyaları
├── test_models.py              ← Modül testleri
├── train_all_models.py         ← Eksiksiz eğitim script'i
├── requirements.txt            ← Python kütüphaneleri
└── README.md                   ← Detaylı dokümantasyon
```

---

## ⚙️ Konfigürasyon

`config.json` dosyasında hiperparametreleri değiştirebilirsiniz:

```json
{
  "model_config": {
    "classification": {
      "neural_network": {
        "hidden_sizes": [128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "epochs": 50
      }
    },
    "forecasting": {
      "arima": {
        "order": [1, 1, 1]
      }
    }
  }
}
```

---

## 🤝 Ekip İntegrasyonu

### Kişi 1'den Veri Al

```python
df_from_person1 = pd.read_csv('data/person1_features.csv')
# Sütunlar: feature_1, feature_2, ..., label (RS/RP), timestamp
```

### Modelleri Kişi 3'e Gönder

```python
from src.integration_logic import ModelIntegrator

integrator = ModelIntegrator(models_dir='models')
integrator.load_classification_model('models/classification_logistic_regression.pkl')
integrator.load_forecasting_model('arima', 'models/forecasting_arima.pkl')

# Kişi 3 bunu Streamlit Dashboard'da kullanabilir
```

---

## 📊 Performans Beklentileri

| Model | Metrik | Hedef | 
|-------|--------|-------|
| Logistic Regression | F1 Score | > 0.75 |
| Neural Network | Accuracy | > 0.80 |
| ARIMA | RMSE | < 10 |
| Prophet | MAE | < 8 |

---

## 🐛 Sık Sorunlar

**Q: Model eğitimi çok yavaş**
A: Batch size'ı artırın veya epochs'u azaltın

**Q: NaN hatası alıyorum**
A: Eksik değerleri kontrol et: `preprocessor.handle_missing_values(df)`

**Q: Model kayıt çalışmıyor**
A: `models/` dizinin var olduğundan emin ol: `mkdir models`

---

## 📞 Yardım

- **README.md**: Detaylı dokümantasyon
- **Docstrings**: Fonksiyonlar `# """` açıklaması içerir
- **Logs**: `logs/training.log` dosyasında hata detayları

---

Başarılar! 🎯
