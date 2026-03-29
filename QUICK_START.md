# QUICK START - Person 2 (Model Architect) - Get Started Quickly

## 🚀 Start in 5 Minutes

### Step 1: Setup

```bash
# Virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install libraries
pip install -r requirements.txt
```

### Step 2: Test Modules

```bash
python test_models.py
```

Output:
```
✓ All modules successfully imported
...
🎉 All tests passed!
```

### Step 3: Run Complete Training Pipeline

```bash
python train_all_models.py
```

This script:
1. Creates synthetic data (will be replaced when Person 1's data arrives)
2. Balances and normalizes the data
3. **Trains 2 classification models**:
   - Logistic Regression
   - Neural Network
4. **Trains 3 forecasting models**:
   - Linear Forecaster
   - ARIMA
   - Prophet
5. Saves all models to `models/` directory
6. Generates `training_report.json` report

---

## 📚 Basic Usage Examples

### Example 1: Classification Model

```python
from src.classification import LogisticRegressionClassifier
import numpy as np

# Create data
X_train = np.random.randn(100, 10)
y_train = np.random.choice(['RS', 'RP'], 100)

# Create and train model
clf = LogisticRegressionClassifier(C=1.0)
clf.train(X_train, y_train)

# Make predictions
X_test = np.random.randn(20, 10)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print(predictions)  # ['RS', 'RP', 'RS', ...]
print(probabilities)  # [[0.8, 0.2], [0.3, 0.7], ...]

# Save model
clf.save('models/my_classifier.pkl')
```

### Example 2: Forecasting Model

```python
from src.forecasting import ARIMAForecaster
import numpy as np

# Time series data
timeseries = np.random.randn(100) * 10 + 100

# Create and train model
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(timeseries)

# Make forecast
forecast = model.forecast(steps=24)
print(forecast)  # [98.5, 99.2, 100.1, ...]

# Confidence intervals
lower, upper = model.get_confidence_intervals(steps=24, alpha=0.05)
```

### Example 3: Model Evaluation

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

## 📂 File Guide

```
Project/
├── src/
│   ├── model_prep.py          ← Data preparation
│   ├── classification.py      ← Classification models
│   ├── forecasting.py         ← Forecasting models
│   ├── evaluator.py           ← Performance evaluation
│   └── integration_logic.py   ← Integration
├── models/                     ← Trained models saved here
├── data/                       ← Datasets placed here
├── logs/                       ← Log files
├── test_models.py              ← Module tests
├── train_all_models.py         ← Complete training script
├── requirements.txt            ← Python libraries
└── README.md                   ← Detailed documentation
```

---

## ⚙️ Configuration

You can modify hyperparameters in `config.json`:

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

### Person 1'den Veri Al

```python
df_from_person1 = pd.read_csv('data/person1_features.csv')
# Sütunlar: feature_1, feature_2, ..., label (RS/RP), timestamp
```

### Modelleri Person 3'e Gönder

```python
from src.integration_logic import ModelIntegrator

integrator = ModelIntegrator(models_dir='models')
integrator.load_classification_model('models/classification_logistic_regression.pkl')
integrator.load_forecasting_model('arima', 'models/forecasting_arima.pkl')

# Person 3 bunu Streamlit Dashboard'da kullanabilir
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
