import streamlit as st
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="centered"
)

# =========================
# DESAIN APLIKASI
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff5f8, #f8d9e8);
}
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}
.kartu {
    background: rgba(255,255,255,0.88);
    padding: 28px;
    border-radius: 30px;
    margin-bottom: 25px;
    box-shadow: 0 4px 15px rgba(150, 50, 100, 0.06);
}
h1 {
    color: #823456;
}
h2, h3 {
    color: #71394f;
}
.stButton > button {
    width: 100%;
    border-radius: 16px;
    height: 58px;
    color: white;
    font-size: 18px;
    background: linear-gradient(90deg, #dc609e, #c63883);
    border: none;
}
.stButton > button:hover {
    color: white;
    border: 1px solid #a72d6b;
}
</style>
""", unsafe_allow_html=True)

# =========================
# JUDUL APLIKASI
# =========================
st.markdown("""
<div class="kartu">
    <h1>Prediksi Risiko Diabetes</h1>
    <p style="font-size:20px; color:#80566a;">
        Sistem Pendukung Keputusan menggunakan
        metode K-Nearest Neighbors (KNN)
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# DATA LATIH DEMO
# =========================
# Data di bawah ini dibuat secara sintetis untuk
# demonstrasi aplikasi, bukan data pasien sungguhan.

@st.cache_resource
def buat_model():
    rng = np.random.default_rng(42)
    jumlah_data = 600

    usia = rng.integers(20, 80, jumlah_data)
    bmi = np.clip(rng.normal(28, 6, jumlah_data), 16, 50)
    glukosa = np.clip(rng.normal(115, 30, jumlah_data), 60, 220)
    tekanan = np.clip(rng.normal(75, 12, jumlah_data), 40, 130)
    keluarga = rng.integers(0, 2, jumlah_data)

    # Label simulasi untuk keperluan demonstrasi.
    skor = (
        (usia >= 45).astype(int) * 0.8
        + (bmi >= 30).astype(int) * 0.8
        + (glukosa >= 126).astype(int) * 1.8
        + (tekanan >= 80).astype(int) * 0.4
        + keluarga * 0.8
        + rng.normal(0, 0.8, jumlah_data)
    )

    target = (skor >= 2.5).astype(int)

    X = pd.DataFrame({
        "Usia": usia,
        "BMI": bmi,
        "Glukosa": glukosa,
        "Tekanan_Darah": tekanan,
        "Riwayat_Keluarga": keluarga
    })

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

    model.fit(X, target)
    return model

model = buat_model()

# =========================
# FORM DATA PASIEN
# =========================
st.markdown('<div class="kartu">', unsafe_allow_html=True)
st.subheader("Masukkan data pasien:")

usia = st.number_input(
    "Usia (tahun)",
    min_value=1,
    max_value=120,
    value=42,
    step=1
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=80.0,
    value=28.5,
    step=0.1
)

glukosa = st.number_input(
    "Glukosa",
    min_value=20,
    max_value=500,
    value=130,
    step=1
)

tekanan = st.number_input(
    "Tekanan darah",
    min_value=20,
    max_value=250,
    value=128,
    step=1
)

keluarga_text = st.selectbox(
    "Riwayat diabetes dalam keluarga",
    ["Ada", "Tidak ada"]
)

keluarga = 1 if keluarga_text == "Ada" else 0

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PROSES PREDIKSI
# =========================
if st.button("Prediksi"):

    data_pasien = pd.DataFrame([{
        "Usia": usia,
        "BMI": bmi,
        "Glukosa": glukosa,
        "Tekanan_Darah": tekanan,
        "Riwayat_Keluarga": keluarga
    }])

    hasil = int(model.predict(data_pasien)[0])
    probabilitas = model.predict_proba(data_pasien)[0]
    persentase = float(probabilitas[1] * 100)

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    if hasil == 1:
        st.error(
            "Hasil simulasi model: Risiko lebih tinggi"
        )
    else:
        st.success(
            "Hasil simulasi model: Risiko lebih rendah"
        )

    st.metric(
        "Skor keluaran model untuk kelas risiko lebih tinggi",
        f"{persentase:.1f}%"
    )

    st.info(
        "Catatan: aplikasi ini menggunakan data sintetis "
        "untuk demonstrasi pemrograman KNN. Hasilnya bukan "
        "prediksi medis yang tervalidasi dan tidak boleh "
        "digunakan untuk diagnosis. Konsultasikan hasil "
        "pemeriksaan glukosa dengan tenaga kesehatan."
    )

st.caption(
    "Proyek akademik | Sistem Pendukung Keputusan | Metode KNN"
)