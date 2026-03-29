"""
Translate markdown documentation files to English
"""

import os

markdown_files = {
    r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project\README.md": {
        "Enerji Tüketim Analizi ve Tahminlemesi": "Energy Consumption Analysis and Forecasting",
        "Ekip Bölümü": "Team Division",
        "Kişi 1": "Person 1",
        "Veri Mimarı ve Clustering Uzmanı": "Data Architect and Clustering Expert",
        "Kişi 2": "Person 2",
        "Model Mimarı (Classification & Forecasting)": "Model Architect (Classification & Forecasting)",
        "Siz burada": "*You are here*",
        "Kişi 3": "Person 3",
        "Entegratör ve Üretken AI Uzmanı": "Integrator and Generative AI Expert",
        "Proje Özeti": "Project Overview",
        "Enedis kaynaklı enerji tüketim verilerini analiz ederek": "By analyzing energy consumption data from Enedis:",
        "Evleri sınıflandırma": "Classify houses",
        "Birincil (RP) veya ikincil (RS) konut olup olmadığını belirlemek": "Determine whether they are primary (RP) or secondary (RS) residences",
        "Gelecek tüketimi tahminleme": "Forecast future consumption",
        "24 saat ileri enerji tüketimi öngörmek": "Predict energy consumption 24 hours ahead",
        "İnteraktif Dashboard": "Interactive Dashboard",
        "Streamlit tabanlı kullanıcı arayüzü sağlamak": "Provide a Streamlit-based user interface",
        "Proje Yapısı": "Project Structure",
        "Kaynak kod": "Source code",
        "Veri hazırlığı ve dengeleme": "Data preparation and balancing",
        "Sınıflandırma modelleri (RS/RP)": "Classification models (RS/RP)",
        "Tahminleme modelleri": "Forecasting models",
        "Model değerlendirme araçları": "Model evaluation tools",
        "Entegrasyon katmanı": "Integration layer",
        "Eğitilmiş modeller (pickle/pt)": "Trained models (pickle/pt)",
        "Veri setleri": "Datasets",
        "Log dosyaları": "Log files",
        "Python kütüphaneleri": "Python libraries",
        "Bu dosya": "This file",
        "Kurulum": "Installation",
        "Python Ortamı Kurması": "Setting up Python Environment",
        "Python 3.9+ gerekli": "Python 3.9+ required",
        "Virtual environment oluştur": "Create virtual environment",
        "Aktivasyon": "Activation",
    },
    r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project\QUICK_START.md": {
        "Hızlı Başlangıç": "Quick Start",
        "1. Veri Hazırlığı": "1. Data Preparation",
        "2. Model Eğitimi": "2. Model Training",
        "3. Model Değerlendirmesi": "3. Model Evaluation",
        "4. Sonuç Saklama": "4. Saving Results",
        "Kişi 1": "Person 1",
        "etiketlenmiş veri": "labeled data",
        "Kişi 2": "Person 2",
        "modelleri eğit": "train models",
        "Kişi 3": "Person 3",
        "dashboard": "dashboard",
    },
    r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project\ARCHITECTURE.md": {
        "Mimari Tasarım": "Architecture Design",
        "Genel Tasarım": "Overall Design",
        "Üç aşamalı pipeline": "Three-stage pipeline",
        "Veri": "Data",
        "Modeller": "Models",
        "Sonuçlar": "Results",
        "Kişi 1": "Person 1",
        "Kişi 2": "Person 2",
        "Kişi 3": "Person 3",
    },
    r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project\PERSON2_SUMMARY.md": {
        "Kişi 2 Görevleri": "Person 2 Tasks",
        "Sınıflandırma": "Classification",
        "Tahminleme": "Forecasting",
        "Değerlendirme": "Evaluation",
        "Entegrasyon": "Integration",
    }
}

for filepath, translations in markdown_files.items():
    if not os.path.exists(filepath):
        print(f"✗ Not found: {filepath}")
        continue
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Apply translations
        for turkish, english in translations.items():
            if turkish in content:
                content = content.replace(turkish, english)
        
        # Write back
        if original != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ {os.path.basename(filepath)}")
        else:
            print(f"- {os.path.basename(filepath)} (no changes)")
            
    except Exception as e:
        print(f"✗ {os.path.basename(filepath)}: {e}")

print("\nMarkdown translation complete!")
