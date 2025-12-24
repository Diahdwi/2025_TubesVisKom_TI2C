# Deteksi Kendaraan Menggunakan YOLOv8n  
*Mata Kuliah: Visi Komputer*

## 📌 Deskripsi Proyek
Proyek ini merupakan implementasi **object detection** untuk mendeteksi kendaraan menggunakan algoritma **YOLOv8n**. Sistem dilatih menggunakan dataset kendaraan yang diperoleh dari Roboflow dan diuji pada gambar maupun video. Model dilatih di Google Colab, kemudian hasil training berupa file `best.pt` digunakan pada aplikasi lokal berbasis Python dengan antarmuka sederhana (UI) di Visual Studio Code.

Proyek ini bertujuan untuk memahami alur lengkap pengembangan sistem visi komputer, mulai dari pengolahan dataset, training model deep learning, hingga implementasi inferensi dan visualisasi hasil.

---

## 🎯 Tujuan
- Mengimplementasikan algoritma YOLOv8n untuk deteksi kendaraan  
- Memahami proses training dan pengujian model object detection  
- Mengembangkan aplikasi sederhana dengan UI untuk menampilkan hasil deteksi  
- Menerapkan workflow pengembangan berbasis Git dan GitHub  

---

## 🧠 Teknologi yang Digunakan
- Python 3.10+
- YOLOv8 (Ultralytics)
- OpenCV
- Roboflow Dataset
- Google Colab (Training)
- Visual Studio Code (Deployment)
- Git & GitHub

---

## 📁 Struktur Folder
Deteksi_Kendaraan/
│
├── model/
│ └── best.pt # Model hasil training (tidak di-push ke GitHub)
│
├── src/
│ └── app.py # Program utama UI dan inferensi
│
├── media/
│ └── sample.jpg # Contoh gambar uji
│
├── requirements.txt # Daftar dependency Python
├── .gitignore
└── README.md


---

## ⚙️ Instalasi & Setup (Local / VS Code)

### 1. Clone Repository
```bash
git clone https://github.com/username/Deteksi_Kendaraan.git
cd Deteksi_Kendaraan

### 2. Buat virtual enviroment
python -m venv venv

Aktifkan:

Windows: venv\Scripts\activate
Linux / Mac: source venv/bin/activate

3. Install Dependency
pip install -r requirements.txt

🚀 Menjalankan Program

Pastikan file best.pt sudah berada di folder model/.

python src/app.py