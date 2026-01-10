# NLP Tabanlı Akıllı Kitap Öneri Sistemi

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge&logo=huggingface)
![LangChain](https://img.shields.io/badge/LangChain-Vector%20Search-green?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge&logo=gradio)

Bu proje, Doğal Dil İşleme (NLP) tekniklerini kullanarak uçtan uca geliştirilmiş bir **Kitap Öneri Sistemi**dir. Standart anahtar kelime aramalarının ötesine geçerek, kullanıcıların sorgularını hem **anlamsal (semantic)** hem de **duygusal (sentiment)** bağlamda analiz eder ve en uygun kitaplarla eşleştirir.

## Proje Demosu
*Projenin çalışma mantığını ve arayüzünü aşağıdaki videodan izleyebilirsiniz.*

https://github.com/user-attachments/assets/9d7a0409-879c-4d7f-b19a-9907227c1ddf



## Proje Akışı ve Özellikler

Proje, veri hazırlığından son kullanıcı arayüzüne kadar 5 ana aşamadan oluşmaktadır:

### 1. Veri Hazırlığı ve ETL
* **Veri Seti:** Kaggle üzerinden alınan "7k-books-with-metadata" seti kullanıldı.
* **İşlem:** Eksik veriler temizlendi, gereksiz sütunlar çıkarıldı ve korelasyon analizleri (heatmap) yapıldı. Kitap açıklamaları NLP modelleri için optimize edildi.

### 2. Zero-Shot Metin Sınıflandırma
Kitapların karmaşık kategorilerini daha anlaşılır üst kümelere ("Çocuk Kurgu", "Yetişkin Kurgu", "Kurgu Dışı" vb.) indirgemek için **Zero-Shot Classification** tekniği uygulandı.
* **Model:** `facebook/bart-large-mnli`

### 3. Duygu Analizi (Sentiment Analysis)
Kullanıcıların ruh haline göre (örn: *"bana neşeli bir kitap öner"*) arama yapabilmesi için kitap açıklamalarının duygu durumları analiz edildi.
* **Etiketler:** Anger, Fear, Joy, Sadness, Surprise, Neutral.
* **Model:** `j-hartmann/emotion-english-distilroberta-base`

### 4. Vektör Arama (Vector Search & RAG)
Metinlerin anlamsal olarak aranabilmesi için vektör veritabanı mimarisi kuruldu.
* **Embedding:** `Google Generative AI Embeddings (models/text-embedding-004)`
* **Veritabanı:** LangChain entegrasyonu ile **ChromaDB**.

### 5. Kullanıcı Arayüzü (Gradio)
Tüm arka plan süreçleri, **Gradio** ile geliştirilen modern ve interaktif bir web arayüzünde birleştirildi.

## Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Repoyu klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/repo-adi.git](https://github.com/KULLANICI_ADIN/repo-adi.git)
    cd repo-adi
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Çevre Değişkenlerini Ayarlayın:**
    Google Generative AI modellerini kullanmak için API anahtarına ihtiyacınız var. `.env` dosyası oluşturun:
    ```env
    GOOGLE_API_KEY=google_api_anahtari
    ```
4. **Eğer pip veya .venv vs. uğraşmak yerine uv kullanmak isterseniz neden_uv.md dosyasından uv kurulumuna detaylı bakabilirsiniz.**


