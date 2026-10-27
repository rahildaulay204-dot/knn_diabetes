import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Sistem Pendukung Keputusan",
    page_icon="📊",
    layout="wide"
)

st.title("Sistem Pendukung Keputusan (SPK)")
st.subheader("Metode SMART")
st.write(
    "Aplikasi ini digunakan untuk membantu menentukan "
    "alternatif terbaik berdasarkan kriteria dan bobot."
)

st.divider()

# DATA KRITERIA
st.header("1. Data Kriteria")

data_kriteria = pd.DataFrame({
    "Kriteria": [
        "Harga",
        "Kualitas",
        "Jarak",
        "Pelayanan"
    ],
    "Tipe": [
        "Cost",
        "Benefit",
        "Cost",
        "Benefit"
    ],
    "Bobot": [30, 30, 20, 20]
})

kriteria = st.data_editor(
    data_kriteria,
    num_rows="dynamic",
    use_container_width=True,
    key="editor_kriteria"
)

st.caption(
    "Benefit berarti semakin besar semakin baik. "
    "Cost berarti semakin kecil semakin baik."
)

st.divider()

# DATA ALTERNATIF
st.header("2. Data Alternatif")

data_alternatif = pd.DataFrame({
    "Alternatif": [
        "Alternatif A",
        "Alternatif B",
        "Alternatif C"
    ],
    "Harga": [80, 70, 90],
    "Kualitas": [75, 90, 80],
    "Jarak": [85, 70, 75],
    "Pelayanan": [80, 85, 70]
})

alternatif = st.data_editor(
    data_alternatif,
    num_rows="dynamic",
    use_container_width=True,
    key="editor_alternatif"
)

st.divider()

# PROSES PERHITUNGAN
if st.button("Hitung Hasil SPK", type="primary"):

    try:
        if kriteria.empty or alternatif.empty:
            st.error("Data kriteria dan alternatif harus diisi.")
            st.stop()

        nama_kriteria = kriteria["Kriteria"].astype(str).str.strip()

        if (nama_kriteria == "").any():
            st.error("Nama kriteria tidak boleh kosong.")
            st.stop()

        if nama_kriteria.duplicated().any():
            st.error("Nama kriteria tidak boleh sama.")
            st.stop()

        bobot = pd.to_numeric(
            kriteria["Bobot"], errors="coerce"
        )

        if bobot.isna().any() or (bobot < 0).any():
            st.error("Bobot harus berupa angka positif.")
            st.stop()

        if bobot.sum() <= 0:
            st.error("Total bobot harus lebih besar dari nol.")
            st.stop()

        if alternatif["Alternatif"].astype(str).str.strip().eq("").any():
            st.error("Nama alternatif tidak boleh kosong.")
            st.stop()

        if alternatif["Alternatif"].astype(str).duplicated().any():
            st.error("Nama alternatif tidak boleh sama.")
            st.stop()

        bobot_normal = bobot / bobot.sum()

        nilai_akhir = []

        for i in range(len(alternatif)):
            total = 0

            for j in range(len(kriteria)):
                nama = nama_kriteria.iloc[j]
                tipe = kriteria["Tipe"].iloc[j]

                nilai = pd.to_numeric(
                    alternatif[nama], errors="coerce"
                )

                if nilai.isna().any():
                    st.error(
                        f"Nilai kriteria {nama} harus berupa angka."
                    )
                    st.stop()

                nilai = nilai.astype(float)

                if (nilai < 0).any():
                    st.error("Nilai tidak boleh negatif.")
                    st.stop()

                nilai_min = nilai.min()
                nilai_max = nilai.max()
                nilai_alternatif = nilai.iloc[i]

                if nilai_max == nilai_min:
                    utility = 1
                elif tipe == "Benefit":
                    utility = (
                        (nilai_alternatif - nilai_min)
                        / (nilai_max - nilai_min)
                    )
                elif tipe == "Cost":
                    utility = (
                        (nilai_max - nilai_alternatif)
                        / (nilai_max - nilai_min)
                    )
                else:
                    st.error(
                        f"Tipe kriteria {nama} harus Cost atau Benefit."
                    )
                    st.stop()

                total += utility * bobot_normal.iloc[j]

            nilai_akhir.append(total)

        # MENAMPILKAN HASIL
        hasil = pd.DataFrame({
            "Alternatif": alternatif["Alternatif"],
            "Nilai Akhir": nilai_akhir
        })

        hasil = hasil.sort_values(
            by="Nilai Akhir",
            ascending=False
        ).reset_index(drop=True)

        hasil.insert(0, "Peringkat", range(1, len(hasil) + 1))

        st.divider()
        st.header("3. Hasil Perangkingan")

        st.dataframe(
            hasil.style.format({"Nilai Akhir": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

        pemenang = hasil.iloc[0]

        st.success(
            f"Alternatif terbaik adalah {pemenang['Alternatif']} "
            f"dengan nilai {pemenang['Nilai Akhir']:.4f}"
        )

        st.subheader("Grafik Nilai Akhir")

        st.bar_chart(
            hasil.set_index("Alternatif")["Nilai Akhir"]
        )

        file_csv = hasil.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Unduh Hasil ke CSV",
            data=file_csv,
            file_name="hasil_spk.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

st.divider()

st.caption(
    "Aplikasi SPK SMART | Sistem Pendukung Keputusan"
)