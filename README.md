# Hırsızlık Davalarında Yapay Zeka Destekli Ceza Tahmini

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

Bu proje, hırsızlık davalarına ait verileri analiz ederek, dava parametreleri ve savunma metinleri üzerinden olası ceza durumunu (Hapis, Beraat vb.) öngören bir karar destek yazılımıdır.

Atatürk Üniversitesi Yapay Zeka ve Veri Mühendisliği Bölümü öğrencileri tarafından (Grup 7) geliştirilen bu sistem, hukuksal karar süreçlerinde yapay zekanın yardımcı bir araç olarak kullanımını simüle etmektedir.

## Proje Mimarisi ve İşleyiş

Sistem iki temel bileşenden oluşmaktadır:

1.  **Ana Uygulama (`app.py`):** Son kullanıcının etkileşime geçtiği nihai yazılımdır. Gerekli modelleri ve algoritmaları kullanarak, girilen veriler ışığında doğrudan sonuç üretir.
2.  **Geliştirme Ortamı (`notebooks/`):** Projenin Ar-Ge sürecinde kullanılan, veri üretimi, model eğitimi ve test çalışmalarını içeren referans kodları barındırır. Standart kullanım için bu dosyaların çalıştırılmasına gerek yoktur.

### Temel Özellikler

* **Sonuç Odaklı Analiz:** Kullanıcıdan alınan yapılandırılmış veriler (yaş, mal değeri vb.) ve metin verileri (ifadeler) işlenerek doğrudan tahmin çıktısı sunulur.
* **Hibrit Değerlendirme:** Sistem, sayısal veriler ile metin madenciliği (NLP) yöntemlerini bir arada kullanarak bütüncül bir değerlendirme yapar.
* **Hızlı Entegrasyon:** Önceden hazırlanmış mimari sayesinde, karmaşık eğitim süreçlerine ihtiyaç duymadan tahminleme gerçekleştirir.

## Kurulum ve Kullanım

Sistemi çalıştırmak için aşağıdaki adımlar izlenmelidir:

1.  **Repoyu Klonlayın:**
    ```bash
    git clone [https://github.com/kullaniciadi/Hirsizlik-Ceza-Tahmini.git](https://github.com/kullaniciadi/Hirsizlik-Ceza-Tahmini.git)
    cd Hirsizlik-Ceza-Tahmini
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets joblib openpyxl
    ```

3.  **Uygulamayı Başlatın:**
    Tahmin sistemini çalıştırmak için ana dizindeki `app.py` dosyasını çalıştırmanız yeterlidir:
    ```bash
    python app.py
    ```

## Kullanılan Teknolojiler

Proje altyapısında aşağıdaki teknolojilerden yararlanılmıştır:

* **Programlama Dili:** Python
* **Veri Analizi ve İşleme:** Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-Learn (Random Forest, Decision Tree)
* **Doğal Dil İşleme:** TF-IDF Vektörleştirme
* **Model Yönetimi:** Joblib

## Proje Ekibi (Grup 7)

Bu çalışma, Atatürk Üniversitesi Yapay Zeka ve Veri Mühendisliği Bölümü **Grup 7** ekibi tarafından hazırlanmıştır.

* **Mustafa Can Akkuş**
* **Mustafa Erfidan**
* **Mustafa Mesut Kaya**
* **Ömer Sunguralp Bektaş**
