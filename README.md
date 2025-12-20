# 3 Top Bilardo Video Analizi

Bu proje, 3 top bilardo (carom bilardo) videolarını analiz ederek ralli başarısını otomatik olarak tespit eder.

## Projenin Amacı

Proje, bilardo videolarından aşağıdaki tespitleri yapar:

- **Top Tespiti:** Beyaz, kırmızı ve sarı topların her frame'deki konumlarını tespit eder
- **Band Vuruşu Tespiti:** Beyaz topun masa kenarlarına (sol, sağ, üst, alt) kaç kez vurduğunu sayar
- **Çarpışma Tespiti:** Beyaz topun kırmızı ve sarı toplara çarpıp çarpmadığını tespit eder
- **Ralli Değerlendirmesi:** Bir rallinin başarılı sayılması için:
  - Beyaz top en az 3 banda vurmalı
  - Beyaz top hem kırmızı hem de sarı topa çarpmalı

## Proje Yapısı

```
cv_2/
├── main.py           # Ana program
├── config.py         # Sabitler ve konfigürasyon
├── detection.py      # Masa ve top tespiti
├── detectors.py      # Band ve çarpışma dedektörleri
├── visualization.py  # Görselleştirme
├── template.jpg      # Ralli ayırıcı template
├── requirements.txt  # Bağımlılıklar
└── README.md
```

## Bağımlılıkların Yüklenmesi

Python 3.8 veya üzeri gereklidir.

```bash
pip install -r requirements.txt
```

Veya bağımlılıkları manuel olarak yükleyin:

```bash
pip install opencv-python>=4.8.0 numpy>=1.24.0 scikit-image>=0.21.0
```

## Kullanım

### Temel Kullanım

```bash
python main.py <video_dosyası>
```

Örnek:

```bash
python main.py input.mp4
```

### Opsiyonel Parametreler

```bash
python main.py <video_dosyası> --template <template_dosyası> --match-threshold <eşik_değeri>
```

- `--template`: Ralli ayırıcı template dosyası (varsayılan: `template.jpg`)
- `--match-threshold`: Template eşleştirme eşiği (varsayılan: `0.8`)

### Kontroller

- **ESC**: Programdan çıkış
- **Herhangi bir tuş**: Ralli özeti gösterildikten sonra devam

## Çıktı

Program çalışırken:
- Her frame'de topların konumları ve tespit edilen olaylar ekranda gösterilir
- Ralli sonunda özet bilgisi ekrana yazdırılır
- Program sonunda genel sonuçlar terminale yazdırılır

Örnek terminal çıktısı:

```
Ralli 1: frame 60-180, Bands=4, Red=True, Yellow=True, BASARILI
Ralli 2: frame 240-350, Bands=2, Red=True, Yellow=False, BASARISIZ
...

=== GENEL SONUCLAR ===
Toplam ralli sayisi: 10
Basarili ralli sayisi: 7
```
