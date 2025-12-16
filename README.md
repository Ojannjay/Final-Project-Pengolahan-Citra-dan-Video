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
- [License](#-license)

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

