# 🕺 VMC Tracker (VSeeFace) — Face • Body • Hands • Fingers

Project Python untuk tracking **wajah, mata, mulut, pose tubuh, tangan, dan jari** memakai **MediaPipe**, lalu mengirim datanya ke **VSeeFace** lewat **VMC/OSC (Virtual Motion Capture)**.

> Cocok buat tugas/portofolio PCV dan setup VTuber sederhana berbasis webcam.

---

## 🧩 Isi Singkat
- [Fitur Utama](#-fitur-utama)
- [Kebutuhan](#-kebutuhan)
- [Instalasi](#-instalasi)
- [Setting VSeeFace](#-setting-vseeface)
- [Menjalankan Program](#-menjalankan-program)
- [Konfigurasi & Tuning](#-konfigurasi--tuning)
- [Troubleshooting](#-troubleshooting)
- [Rencana Pengembangan](#-rencana-pengembangan)
- [Credits](#-credits)

---

## ✨ Fitur Utama

- 🧠 **Head tracking** (pitch / yaw / roll) + pembagian gerak **Neck vs Head**
- 👁️ **Eye gaze** (iris tracking) + **blink** (EAR threshold)
- 👄 **Mulut terbuka** untuk blendshape (contoh: `"A"`)
- 🧍 **Pose tubuh**: spine + bahu + lengan (upper/lower)
- 🤚 **Finger tracking**: 10 jari dengan **curl detection** (Thumb, Index, Middle, Ring, Little)
- 🧊 **Stabilizer**: Kalman filter untuk gerakan lebih halus
- 🪶 **Ringan**: diset supaya masih usable di device low-end

---

## 🧰 Kebutuhan

- Python 3.x
- Webcam
- VSeeFace (VMC receiver aktif)
- Koneksi jaringan:
  - Jika 1 PC: pakai `127.0.0.1`
  - Jika beda device: pastikan satu jaringan & IP benar

---

## 🔧 Instalasi

### 1) Clone repo
```bash
git clone <repo-url-kamu>
cd <nama-folder-project>
```

### 2) (Opsional tapi disarankan) Virtual environment

Windows (PowerShell / Git Bash)
```bash
python -m venv .venv
```

Aktifkan Powershell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Gitbash:
``` gitbash
source .venv/Scripts/activate
```

### 3) Install dependencies 
install requirements.txt dari environment:
```bash
pip install -r requirements.txt
```

NOTES: kalau kamu pernah kena error protobuf(ini jarang terjadi sih), kamu bisa install protobuff dengan:
```bash
pip install protobuf==4.25.3
```

## 🎛️ Setting VSeeFace
### 1) Nyalakan VMC receiver
1.Buka VSeeFace
2.Masuk Settings
3.Temukan bagian VMC Protocol
4.Enable VMC protocol receiver
5. Port: 39539 (atau sesuaikan, yang penting sama dengan yang ada di VSeeFace)

### 2) Pastikan tracking eksternal terbaca
Di bagian tracking (nama menu bisa beda tergantung versi), pastikan opsi semacam:
- ✅ External tracking / VMC receiver
- ✅ Additional bone tracking (jika ada)

### 3) IP untuk kasus beda device
Kalau VSeeFace ada di PC lain:
Jalankan ipconfig, cari IPv4 Address
Update di main.py:

```python
OSC_IP = "10.x.x.x"     # IP PC yang menjalankan VSeeFace
OSC_PORT = 39539        # pastikan sama dengan port VMC receiver
```

## 🚀 Menjalankan Program

Pastikan VSeeFace sudah jalan dan VMC receiver aktif
Jalankan:
```python
python main.py
```

### Kontrol:
- Tekan Q untuk keluar.
- Saran posisi biar tracking enak:
- Jarak ± 1–2 meter dari kamera
- Lighting cukup
- Tangan masuk frame (kalau mau finger tracking)

## 🧪 Konfigurasi & Tuning
Parameter penting yang biasanya kamu utak-atik ada di main.py:
Koneksi:
```python
OSC_IP = "127.0.0.1"
OSC_PORT = 39539
```
Kamera / performa:
```python
WEBCAM_ID = 0
TARGET_FPS = 30
```

Gerakan lengan dan jari:
```python
ARM_GAIN_XY = 1.2
ARM_GAIN_Z  = 0.5
FINGER_SENSITIVITY = 1.3
```
## 🧯 Troubleshooting
### Avatar tidak bergerak sama sekali:
- Pastikan IP + Port sama persis antara VSeeFace dan script
-Kalau beda device: pastikan 1 jaringan
- Cek firewall Windows (Python/port UDP bisa ke-block)

### FPS drop / berat:
- Turunkan resolusi capture (misal 640x360)
- Coba model_complexity=0 pada MediaPipe Holistic
- Kalau perlu, matikan finger tracking sementara

### Protobuf error:
- Kalau muncul error terkait MessageFactory/GetPrototype, install versi ini:

```python
pip install protobuf==4.25.3
```

## 🧭 Rencana Pengembangan
- Full leg chain (hip/knee/ankle) lebih stabil
- Save/Load profile kalibrasi
- GUI tuning real-time
- Multi-camera / fallback tracker
- Recording & playback tracking
- Dibuat untuk bisa semua avatar(karena sekarang masih terbatas di avatar default VSeeCam)

## 🙏 Credits
- MediaPipe (Google) — tracking face/pose/hands
- VSeeFace — VTuber software
- VMC (Virtual Motion Capture) — standard OSC message set
- python-osc — OSC client library untuk Python

Terimakasih telah memakai 🤩🤩🤩


Sudah saatnya menjadi V-Tuber dan menggantikan Gawr-Gura.
Otsu.. yubi yubi 🦈


