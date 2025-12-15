# 🌊 Aplikasi Prediksi Banjir Jakarta

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prediksi-banjir-jakarta.streamlit.app/)

## 📌 Tentang Program
Repositori ini berisi kode sumber (source code) untuk **Aplikasi Prediksi Banjir Jakarta**. Program ini dikembangkan sebagai bagian dari **Skripsi/Tugas Akhir** untuk memprediksi potensi banjir di wilayah Jakarta berdasarkan parameter cuaca dan data historis.

Aplikasi ini dibangun menggunakan bahasa pemrograman **Python** dengan framework **Streamlit** untuk antarmuka web yang interaktif dan mudah digunakan.

**Akses Aplikasi:**
Aplikasi sudah di-deploy dan dapat diakses secara online melalui tautan berikut:
👉 **[https://prediksi-banjir-jakarta.streamlit.app/](https://prediksi-banjir-jakarta.streamlit.app/)**

---

## ⚙️ Persyaratan Sistem & Instalasi

Jika Anda ingin menjalankan program ini di komputer lokal (localhost), ikuti langkah-langkah berikut:

### Prasyarat
* **Python 3.8** atau versi lebih baru.
* **PIP** (Python Package Installer).
* **Git** (Opsional, untuk clone repository).

### Langkah Instalasi

1.  **Clone atau Download Repositori**
    Buka terminal (CMD/PowerShell) dan jalankan:
    ```bash
    git clone [https://github.com/535220088/Program.git](https://github.com/535220088/Program.git)
    cd Program
    ```
    *Atau download ZIP dari GitHub dan ekstrak foldernya.*

2.  **Buat Virtual Environment (Disarankan)**
    Agar library tidak mengganggu sistem utama komputer Anda:
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Library**
    Install semua dependensi yang diperlukan:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Program**
    Jalankan aplikasi menggunakan Streamlit:
    ```bash
    streamlit run app.py
    ```
    *(Jika nama file utama Anda berbeda, sesuaikan `app.py` dengan nama file Anda, misal `main.py`)*

---

## 📂 Listing Program (Struktur File)

Berikut adalah penjelasan struktur file dalam repositori ini:

* **`Programku.py`**: File utama aplikasi. Berisi kode untuk menjalankan antarmuka Streamlit dan logika prediksi.
* **`requirements.txt`**: Daftar pustaka Python yang wajib diinstall (seperti `streamlit`, `pandas`, `scikit-learn`, `numpy`).
* **`dataset/`**: Folder yang menyimpan data latih (CSV) curah hujan dan data banjir historis.
* **`model/`**: Folder atau file (misal `.pkl`) yang berisi model Machine Learning yang sudah dilatih.
* **`README.md`**: File dokumentasi ini.

---

## 📖 Manual Pemakaian Program

1.  **Akses Aplikasi:**
    Buka [link ini](https://prediksi-banjir-jakarta.streamlit.app/) di browser atau jalankan secara lokal.
2.  **Input Data:**
    Pada menu di sebelah kiri (sidebar) atau halaman utama, masukkan data yang diminta, seperti:
    * Curah Hujan (mm/hari)
    * Lokasi / Wilayah
    * Parameter cuaca lainnya
3.  **Proses Prediksi:**
    Klik tombol **"Hitung Prediksi"** atau **"Analisis"**.
4.  **Hasil:**
    Sistem akan menampilkan status (misal: "Aman", "Siaga", "Banjir") beserta tingkat probabilitasnya.

---

## 🎓 Keterangan Tambahan
Program ini dibuat untuk memenuhi syarat Skripsi pada program studi Teknik Informatika.
**Penulis:** Alek Piter Wardoyo (535220088)

---