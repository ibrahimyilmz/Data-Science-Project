"""
Comprehensive Turkish to English translator for all Python files
"""

import os
import re

# More comprehensive translation mapping
translations = {
    # Module documentation phrases
    "tarafından sağlanan": "provided by",
    "için hazırlar": "prepares for",
    "dengeleme": "balancing",
    "Fonksiyonlar:": "Functions:",
    "Test setinde eşit": "Equal",
    "dağılımı sağlar": "distribution in test set",
    "Zaman serisi yapısını korur": "Preserves time series structure",
    "Özellik vektorünü modele hazır hale getirir": "Prepares feature vector for model input",
    
    # Class descriptions
    "Veri ön işleme ve hazırlama sınıfı": "Data preprocessing and preparation class",
    "Özellikleri:": "Features:",
    "Dataset dengeleme (RS/RP eşit dağılımı)": "Dataset balancing (equal RS/RP distribution)",
    "Zaman serisi yapısını koruyarak train/test split": "Train/test split preserving time series structure",
    "Eksik veri işleme": "Missing value handling",
    "Özellik normalizasyonu": "Feature normalization",
    
    # Method docstrings 
    "DataPreprocessor başlatma": "Initialize DataPreprocessor",
    "Test seti oranı (0-1)": "Test set ratio (0-1)",
    "Reproducibility için seed değeri": "Random seed for reproducibility",
    "Test veri setinde eşit sayıda RS ve RP örneği sağlamak için\n        veri setini dengeler": 
        "Balances dataset to ensure equal number of RS and RP examples in test set",
    "Test veri setinde eşit sayıda RS ve RP örneği sağlamak için veri setini dengeler": 
        "Balances dataset to ensure equal number of RS and RP examples in test set",
    "Yöntem: Minorite sınıfı oversampling, Majority sınıfı undersampling": 
        "Method: Minority class oversampling, majority class undersampling",
    "Orijinal veri seti": "Original dataset",
    "Etiket sütun adı ('RS' veya 'RP' içerir)": "Label column name (contains 'RS' or 'RP')",
    "Dengeli veri seti": "Balanced dataset",
    
    # Code comments
    "Sınıf dağılımını kontrol et": "Check class distribution",
    "Orijinal sınıf dağılımı:": "Original class distribution:",
    "Her sınıftan aynı sayıda örnek al": "Take equal number of samples from each class",
    "Dengeli sınıf dağılımı:": "Balanced class distribution:",
    "Dataset dengeleme hatası:": "Dataset balancing error:",
    
    # Time series
    "Zaman serisi yapısını koruyarak train/test split yapar": 
        "Performs train/test split while preserving time series structure",
    "TimeSeriesSplit kullanarak data leakage'ı önler": 
        "Uses TimeSeriesSplit to prevent data leakage",
    "Veri seti": "Dataset",
    "Zaman sütun adı": "Time column name",
    "Özellik sütunları listesi": "List of feature columns",
    "Zaman sütununa göre sıralama": "Sort by time column",
    "Eğer features_cols belirtilmemişse tüm sayısal sütunları kullan": 
        "If features_cols not specified, use all numeric columns",
    "Zaman serisi bölünmesi": "Time series split",
    "Train seti boyutu:": "Training set size:",
    "Test seti boyutu:": "Test set size:",
    "Özellik sayısı:": "Number of features:",
    "Train/Test split hatası:": "Train/Test split error:",
    
    # Normalization
    "Özellik normalizasyonu uygular": "Applies feature normalization",
    "Eğitim özellikleri": "Training features",
    "Test özellikleri": "Test features",
    "Normalizasyon yöntemi ('standard' veya 'minmax')": 
        "Normalization method ('standard' or 'minmax')",
    "Bilinmeyen normalizasyon yöntemi:": "Unknown normalization method:",
    "normalizasyonu uygulandı": "normalization applied",
    "Normalizasyon hatası:": "Normalization error:",
    
    # Missing values
    "Eksik verileri işler": "Handles missing values",
    "İşleme yöntemi ('interpolate', 'forward_fill', 'backward_fill')": 
        "Processing method ('interpolate', 'forward_fill', 'backward_fill')",
    "Eksik veri işlenmiş veri seti": "Dataset with missing values handled",
    "Bilinmeyen eksik veri işleme yöntemi:": "Unknown missing value handling method:",
    "Eksik veriler": "Missing values handled using",
    "yöntemi ile işlendi": "method",
    "Eksik veri işleme hatası:": "Missing value handling error:",
    
    # Outliers
    "Aykırı değerleri tespit ve çıkarır": "Detects and removes outliers",
    "Veri (n_samples x n_features)": "Data (n_samples x n_features)",
    "Yöntem ('iqr' veya 'zscore')": "Method ('iqr' or 'zscore')",
    "Eşik değeri": "Threshold value",
    "Aykırı değerler çıkarılmış veri": "Data with outliers removed",
    "Bilinmeyen aykırı değer yöntemi:": "Unknown outlier removal method:",
    "Aykırı değerler çıkarıldı:": "Outliers removed:",
    "örnekler": "samples",
    "Aykırı değer çıkarma hatası:": "Outlier removal error:",
    
    # Lag features
    "Tahminleme modeli için gecikmeli (lag) özellikleri oluşturur": 
        "Creates lagged (lag) features for forecasting model",
    "Dünün tüketimi (lag=24), Geçen haftanın (lag=7) vb": 
        "Example: Yesterday's consumption (lag=24), last week's (lag=7), etc",
    "Zaman serisi veri seti": "Time series dataset",
    "Değer sütun adı": "Value column name",
    "Oluşturulacak lag değerleri (saat cinsinden)": "Lag values to create (in hours)",
    "Lag özellikleri eklenen veri seti": "Dataset with lag features added",
    "İlk satırların NaN değerlerini kaldır": "Remove NaN values from first rows",
    "Lag özellikleri oluşturuldu:": "Lag features created:",
    "Lag özellikleri oluşturma hatası:": "Lag features creation error:",
    
    # Test code
    "Test kodu": "Test code",
    "Örnek veri seti oluştur": "Create sample dataset",
    "Sahte veri üret": "Generate fake data",
    "Orijinal veri seti şekli:": "Original dataset shape:",
    "Sınıf dağılımı:": "Class distribution:",
    "DataPreprocessor'ı test et": "Test DataPreprocessor",
    "Dataset dengeleme": "Dataset balancing",
    "Dengeli veri seti şekli:": "Balanced dataset shape:",
    "X_train şekli:": "X_train shape:",
    "X_test şekli:": "X_test shape:",
    "Normalizasyon": "Normalization",
    "X_train (normalized) ortalaması:": "X_train (normalized) mean:",
    "X_train (normalized) standart sapması:": "X_train (normalized) std:",
    
    # Classification module
    "Sınıflandırma Modelleri - RS/RP Sınıflandırması": 
        "Classification Models - RS/RP Classification",
    "RS veya RP olarak sınıflandırma yapar": "Classifies data as RS or RP",
    "İkili sınıflandırma problemi": "Binary classification problem",
    "Baz Sınıflandırıcı soyut sınıfı": "Base Classifier abstract class",
    "Lojistik Regresyon sınıflandırıcı": "Logistic Regression Classifier",
    "Sinir Ağı sınıflandırıcı (PyTorch tabanlı)": "Neural Network Classifier (PyTorch based)",
    "3 katmanlı mimari [128, 64, 32]": "3-layer architecture [128, 64, 32]",
    "Modeli eğit": "Train model",
    "Öğrenme oranı": "Learning rate",
    "Epoch sayısı": "Number of epochs",
    "Batch boyutu": "Batch size",
    "Model eğitim hatası:": "Model training error:",
    "Eğitim": "Training",
    "Test": "Test",
    
    # Forecasting
    "Tahminleme Modelleri": "Forecasting Models",
    "24 saat önceden enerji tüketimini tahmin eder": 
        "Forecasts energy consumption 24 hours ahead",
    "Zaman serisi tahminleme modelleri": "Time series forecasting models",
    "Doğrusal Tahmin Modeli": "Linear Forecasting Model",
    "ARIMA Tahmin Modeli": "ARIMA Forecasting Model",
    "Prophet Tahmin Modeli": "Prophet Forecasting Model",
    "Modeli eğit ve tahmin yap": "Train model and make predictions",
    "Tahminleme hatası:": "Forecasting error:",
    
    # Evaluator
    "Model Değerlendirilmesi ve Karşılaştırma": 
        "Model Evaluation and Comparison",
    "Sınıflandırma ve Tahminleme Değerlendirmesi": 
        "Classification and Forecasting Evaluation",
    "Model Karşılaştırması": "Model Comparison",
    "Sınıflandırma metrikleri (Accuracy, Precision, Recall, F1)": 
        "Classification metrics (Accuracy, Precision, Recall, F1)",
    "Tahminleme metrikleri (MAE, RMSE, MAPE)": 
        "Forecasting metrics (MAE, RMSE, MAPE)",
    "Metrikleri hesapla": "Calculate metrics",
    
    # Integration
    "Model Entegrasyonu ve Veri Pipeline": 
        "Model Integration and Data Pipeline",
    "Kişi 1'in verisi ve Kişi 3'ün Dashboard'u arasında ara katman": 
        "Integration layer between Person 1's data and Person 3's dashboard",
    "Modelleri yükle ve tahmin yap": "Load models and make predictions",
    "Sonuçları formatla": "Format results",
    
    # Common messages
    "başarıyla kaydedildi": "saved successfully",
    "başarıyla yüklendi": "loaded successfully",
    "hatası:": "error:",
    "uyarı": "warning",
    "bilgi": "info",
}

# Function to apply all translations
def translate_comprehensive(text):
    """Apply comprehensive Turkish to English translations"""
    result = text
    
    # Sort by length descending to replace longer phrases first
    for turkish in sorted(translations.keys(), key=len, reverse=True):
        english = translations[turkish]
        if turkish in result:
            result = result.replace(turkish, english)
    
    return result

# Get only project files (not venv)
python_files = []
project_root = r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project"

for root, dirs, files in os.walk(project_root):
    # Skip excluded directories
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.venv', 'models', 'data', 'logs', '__pycache__']]
    
    for file in files:
        if file.endswith('.py') and not file.startswith('translate'):
            filepath = os.path.join(root, file)
            # Skip venv files
            if '.venv' not in filepath:
                python_files.append(filepath)

print(f"Found {len(python_files)} project Python files:")
for f in python_files:
    print(f"  - {os.path.relpath(f, project_root)}")

# Process each file
translated_count = 0
for filepath in python_files:
    rel_path = os.path.relpath(filepath, project_root)
    print(f"\n[{rel_path}]", end=" ")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Translate content
        translated_content = translate_comprehensive(original_content)
        
        # Check if there were any changes
        if original_content != translated_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            print("✓ Translated")
            translated_count += 1
        else:
            print("Already English")
            
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*60)
print(f"Translation complete! {translated_count} files updated")
