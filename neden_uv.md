# bookRecommender - Modern Python Geliştirme Rehberi (uv)

Bu proje, Python paket yönetimi için standart `pip` yerine çok daha hızlı ve güvenilir olan `uv` aracını kullanır. Bu rehber, projenin taşınması veya yeniden kurulması durumunda izlenecek adımları içerir.

---

## Neden `uv` Kullanıyoruz?

1. **Hız:** Paket kurulumu ve bağımlılık çözümü `pip`'ten 10-100 kat daha hızlıdır.
2. **Güvenilirlik:** `uv.lock` dosyası sayesinde her kurulumda aynı kütüphane sürümleri yüklenir.
3. **Temizlik:** Proje klasörü taşınsa bile `uv sync` ile ortam saniyeler içinde onarılır.

---

## Kurulum ve Yapılandırma

Aşağıdaki komutları **PowerShell** üzerinde projenin ana dizininde çalıştırın.

### 1. uv Aracını Kurun
Eğer sisteminizde `uv` yüklü değilse (bir kez yapılması yeterlidir):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Proje Ortamını Hazırlayın
Eski veya bozuk olan `.venv` klasörünü sildikten sonra:
```powershell
# Proje yapısını ilklendirir (pyproject.toml oluşturur)
uv init

# Sanal ortamı oluşturur
uv venv

# Sanal ortamı aktif eder
.\.venv\Scripts\activate
```

### 3. Kütüphaneleri Yükle
Projenin (`dashboard.py`) çalışması için gereken temel paketleri ekleyin:
```powershell
uv add pandas streamlit scikit-learn
```

---

## Uygulamayı Çalıştırma

Sanal ortam aktifken (başında `(.venv)` yazarken) şu komutları kullanın:

```powershell
# Eğer bu bir Streamlit uygulamasıysa:
uv run streamlit run dashboard.py

# Eğer standart bir script ise:
uv run python dashboard.py
```

---

##  Proje Taşınırsa (D'den C'ye vb.) Ne Yapılmalı?

Sanal ortamlar dosya yollarına duyarlıdır. Projeyi taşıdığınızda `No Python at...` hatası alırsanız:

1. Eski `.venv` klasörünü silin.
2. Yeni konumda terminali açıp şu komutu çalıştırın:
```powershell
uv sync
```
*`uv`, tüm bağımlılıkları saniyeler içinde yeni konuma göre yeniden yapılandıracaktır.*

---
