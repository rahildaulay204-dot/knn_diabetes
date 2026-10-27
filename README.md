# KNN Diabetes — Streamlit

Aplikasi dashboard edukasi klasifikasi diabetes menggunakan algoritma K-Nearest Neighbors (KNN).

## Cara menjalankan di komputer
1. Ekstrak ZIP ini.
2. Install Python 3.10 atau lebih baru.
3. Buka Terminal / Command Prompt pada folder ini.
4. Jalankan:
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
5. Browser akan membuka aplikasi lokal Streamlit.

## Cara memasang di Streamlit Community Cloud
1. Upload isi folder ini ke repository GitHub.
2. Di Streamlit Community Cloud, pilih repository tersebut.
3. Pilih `streamlit_app.py` sebagai file utama lalu deploy.

## Dataset
Aplikasi menyediakan data simulasi untuk demonstrasi antarmuka. Data simulasi bukan data medis dan tidak cocok untuk menarik kesimpulan klinis.
Untuk eksperimen yang lebih bermakna, unggah dataset CSV dengan kolom:
`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

Kolom `Outcome` biasanya bernilai 0 atau 1. Nama kolom Indonesia tertentu dikenali juga.

## Catatan
Aplikasi ini hanya prototipe pembelajaran, bukan alat diagnosis medis.
