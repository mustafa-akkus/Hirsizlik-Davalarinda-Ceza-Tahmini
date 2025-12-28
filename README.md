# Hırsızlık Davalarında Yapay Zeka Destekli Ceza Tahmini

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

Bu proje, hırsızlık davalarına ait verileri analiz ederek, dava parametreleri ve savunma metinleri üzerinden olası ceza durumunu (Hapis, Beraat vb.) öngören bir karar destek yazılımıdır.

Atatürk Üniversitesi Yapay Zeka ve Veri Mühendisliği Bölümü öğrencileri tarafından (Grup 7) geliştirilen bu sistem, hukuksal karar süreçlerinde yapay zekanın yardımcı bir araç olarak kullanımını simüle etmektedir.

## Proje Mimarisi ve İşleyiş

Sistem, kullanıcı etkileşimini en üst düzeye çıkarmak amacıyla Jupyter Notebook tabanlı bir arayüz üzerine inşa edilmiştir:

1.  **Ana Simülasyon (`app.ipynb`):** Son kullanıcının etkileşime geçtiği ana dosyadır. İçerisinde barındırdığı görsel arayüz (GUI) bileşenleri (butonlar, kaydırma çubukları vb.) sayesinde kod yazmaya gerek kalmadan tahminleme yapılmasını sağlar.
2.  **Geliştirme Ortamı (`notebooks/`):** Projenin Ar-Ge sürecinde kullanılan, veri analizi, model eğitimi ve test çalışmalarını içeren referans kodları barındırır.

### Temel Özellikler

* **İnteraktif Arayüz:** `ipywidgets` teknolojisi kullanılarak hazırlanan kullanıcı dostu panel üzerinden kolay veri girişi.
* **Hibrit Değerlendirme:** Sayısal veriler (yaş, mal değeri) ile metin madenciliği (NLP - ifade analizi) yöntemlerini birleştirerek bütüncül sonuç üretimi.
* **Anlık Tahmin:** Eğitilmiş modeller üzerinden saniyeler içerisinde karar analizi.

## Kurulum ve Kullanım

Sistemi ve arayüzü çalıştırmak için aşağıdaki adımlar izlenmelidir:

1.  **Repoyu Klonlayın:**
    ```bash
    git clone https://github.com/mustafa-akkus/Hirsizlik-Davalarinda-Ceza-Tahmini.git
    cd Hirsizlik-Davalarinda-Ceza-Tahmini
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets joblib openpyxl notebook
    ```

3.  **Uygulamayı Başlatın:**
    Terminal ekranına aşağıdaki kodu yazarak Jupyter ortamını başlatın:
    ```bash
    jupyter notebook
    ```
    
    * Açılan tarayıcı penceresinde **`app.ipynb`** dosyasına tıklayın.
    * Üst menüden **"Run"** (veya "Cell > Run All") seçeneğini kullanarak sistemi aktif hale getirin.
    * Sayfanın en altında belirecek olan **Suç Analiz ve Tahmin Sistemi** paneli üzerinden veri girişi yapabilirsiniz.

## Kullanılan Teknolojiler

Proje altyapısında aşağıdaki teknolojilerden yararlanılmıştır:

* **Programlama Dili:** Python
* **Arayüz:** Jupyter Notebook & IPyWidgets
* **Veri Analizi:** Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-Learn (Random Forest)
* **Doğal Dil İşleme:** TF-IDF Vektörleştirme

## Proje Ekibi (Grup 7)

Bu çalışma, Atatürk Üniversitesi Yapay Zeka ve Veri Mühendisliği Bölümü **Grup 7** ekibi tarafından hazırlanmıştır.

* **Mustafa Can Akkuş**
* **Mustafa Erfidan**
* **Mustafa Mesut Kaya**
* **Ömer Sunguralp Bektaş**
