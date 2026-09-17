# Sistem Deteksi Bahasa Isyarat Angka

Proyek ini memperluas sistem hand tracking untuk mendeteksi dan menerjemahkan bahasa isyarat angka 1-10.

## Struktur Proyek

```
tracking-hand-and-finger/
├── hand_tracking.py          # Main application dengan deteksi angka
├── camera.py                 # Utility kamera dan flip
├── sign_number.py           # Classifier bahasa isyarat angka
├── dataset_collector.py     # Tool koleksi dataset
└── README.md               # Dokumentasi ini
```

## Fitur

1. **Tracking Tangan & Jari** (existing)
   - Deteksi 21 landmark tangan menggunakan MediaPipe
   - Status lipat/lurus per jari
   - Multi-hand tracking (hingga 10 tangan)
   - Gesture "thumbs up" dengan emoji overlay

2. **Bahasa Isyarat Angka 1-10** (new)
   - Rule-based classifier untuk angka 0-9
   - Deteksi real-time melalui kamera
   - Toggle deteksi dengan tombol 'a'
   - Confidence threshold 70%

3. **Dataset Collection** (new)
   - Tool untuk koleksi data training
   - Export ke CSV format
   - 10 sampel per angka (dapat disesuaikan)

## Instalasi

```bash
pip install opencv-python mediapipe pillow
```

Model MediaPipe akan didownload otomatis saat pertama kali dijalankan.

## Penggunaan

### 1. Deteksi Angka Real-time

```bash
python hand_tracking.py [--camera 0] [--flip normal] [--max-hands 1] [--gpu 0]
```

**Kontrol:**
- `q` = keluar
- `f` = toggle flip mode
- `a` = toggle deteksi angka (hanya jika sign_number.py tersedia)

### 2. Koleksi Dataset

```bash
python dataset_collector.py [--camera 0] [--output dataset_numbers] [--samples 10]
```

**Kontrol:**
- `1`-`0` = pilih angka (1-10, 0 untuk angka 0)
- `SPACE` = capture sampel
- `s` = simpan batch dan lanjut
- `q` = keluar dan simpan

## Aturan Angka Bahasa Isyarat

| Angka | Konfigurasi Jari |
|-------|------------------|
| 0 | Semua jari lipat (genggaman) |
| 1 | Telunjuk lurus, jari lain lipat |
| 2 | Telunjuk + tengah lurus (V) |
| 3 | Telunjuk + tengah + manis lurus |
| 4 | 4 jari (telunjuk-tengah-manis-kelingking) lurus |
| 5 | Semua 5 jari terbuka |
| 6 | Jempol + kelingking lurus (shaka) |
| 7 | Jempol + telunjuk + tengah lurus |
| 8 | Jempol + telunjuk lurus (L) |
| 9 | Jempol lurus, jari lain lipat |

## Algoritma Classifier

Classifier menggunakan pendekatan rule-based dengan hysteresis untuk stabil detection:

1. **Hysteresis Thresholds:**
   - Jari normal: lurus ≥160°, kembali lipat ≤155° (5° hysteresis gap)
   - Jempol: lurus ≥150°, kembali lipat ≤140° (10° hysteresis gap)

2. **Noise Reduction:**
   - Moving average dari 5 frames terakhir
   - Minimal 60% frames lurus untuk dianggap "extended"
   - Hysteresis mencegah "kedut-kedut" saat sudut di ambang batas

3. **Logika Perhitungan:**
   - Hitung total jari extended dari semua tangan
   - 1 tangan: angka 0-5
   - 2 tangan: jumlahkan extended fingers (6-10)
   - Minimum confidence: 70%

4. **Stabil Detection:**
   - State smoothing untuk transisi smooth
   - No false positives dari sudut ambigu
   - Jempol yang sedikit menekuk (<140°) terdeteksi sebagai lipat

## Extensi Masa Depan

1. **Dataset ML**: Gunakan dataset yang dikoleksi untuk train model ML
2. **Gesture Lain**: Tambahkan alphabet A-Z, kata umum
3. **Two-hand gestures**: Deteksi gesture menggunakan dua tangan
4. **Real-time Translation**: Streaming translation ke teks
5. **Export Formats**: JSON, SQLite, TensorFlow/PyTorch datasets

## Troubleshooting

1. **ImportError: No module named 'sign_number'**
   - Pastikan semua file ada di direktori yang sama
   - Run dari directory `tracking-hand-and-finger/`

2. **Kamera tidak terbuka**
   - Coba `--camera 1` atau `--camera -1` untuk pilih kamera lain
   - Pastikan kamera tidak digunakan aplikasi lain

3. **Deteksi tidak akurat**
   - Pastikan pencahayaan cukup
   - Tangan harus dalam frame dengan jelas
   - Gunakan background kontras