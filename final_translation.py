"""
Automated final Turkish to English translation for all source files
"""

import os
import re

def final_translations():
    """Apply final comprehensive Turkish to English translations"""
    
    files_to_fix = [
        r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project\src\classification.py",
        r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project\src\forecasting.py",
        r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project\src\evaluator.py",
        r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project\src\integration_logic.py",
    ]
    
    turkish_patterns = [
        # Class and method descriptions
        ('tabanlı sınıflandırıcı', 'based classifier'),
        ('Özellikler:', 'Features:'),
        ('Hızlı eğitim ve tahmin', 'Fast training and prediction'),
        ('Baseline model olarak kullanım', 'Use as baseline model'),
        ('Explainability (Açıklanabilirlik) yüksek', 'High explainability'),
        ('Hiperparametreler:', 'Hyperparameters:'),
        ('Regularizasyon gücü', 'Regularization strength'),
        ('Optimizasyon algoritması', 'Optimization algorithm'),
        ('iterasyon sayısı', 'number of iterations'),
        ('başlatma', 'Initialize'),
        ('Regularizasyon gücü (C değeri küçüldükçe daha kuvvetli regularizasyon)', 
         'Regularization strength (smaller C means stronger regularization)'),
        ('başlatıldı', 'initialized'),
        ('Training etiketleri', 'Training labels'),
        ('Ekstra parametreler (kullanılmaz)', 'Extra parameters (not used)'),
        ('Modeli eğit', 'Train model'),
        ('Eğitim hatası:', 'Training error:'),
        ('başarıyla eğitildi', 'trained successfully'),
        ('Tahmin yap', 'Make predictions'),
        ('Tahmin hatası:', 'Prediction error:'),
        ('Modeli kaydet', 'Save model'),
        ('kaydedildi', 'saved'),
        ('Modeli yükle', 'Load model'),
        ('yüklendi', 'loaded'),
        ('Değerlendir', 'Evaluate'),
        ('metrikleri hesapla', 'calculate metrics'),
        ('tabanlı derin sinir ağı', 'PyTorch-based deep neural network'),
        ('katman', 'layers'),
        ('aktivasyon', 'activation'),
        ('ReLU', 'ReLU'),
        ('yapı', 'architecture'),
        ('Tahminleme', 'Forecasting'),
        ('Zaman serisi', 'Time series'),
        ('Tüm', 'All'),
        ('Değerlendir', 'Evaluate'),
        ('Karşılaştır', 'Compare'),
        ('hatası', 'error'),
        ('hatasız', 'successfully'),
    ]
    
    for filepath in files_to_fix:
        if not os.path.exists(filepath):
            print(f"✗ File not found: {filepath}")
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Apply all translations
            for turkish, english in turkish_patterns:
                if turkish in content:
                    content = content.replace(turkish, english)
            
            # Write back if changed
            if original != content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✓ {os.path.basename(filepath)} - Translated")
            else:
                print(f"- {os.path.basename(filepath)} - Already English")
                
        except Exception as e:
            print(f"✗ {os.path.basename(filepath)} - Error: {e}")

if __name__ == "__main__":
    print("Final Turkish to English Translation")
    print("=" * 60)
    final_translations()
    print("=" * 60)
    print("Complete!")
