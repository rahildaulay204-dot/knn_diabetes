import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="KNN Diabetes | Health Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f8fc; }
section[data-testid="stSidebar"] { background: #102b46; }
section[data-testid="stSidebar"] * { color: #f4f8fc !important; }
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p { color: #c6d6e6 !important; }
.hero {
 background: linear-gradient(115deg,#123d5a 0%,#176b70 58%,#35a68b 100%);
 border-radius: 22px; padding: 30px 32px; color: white; margin-bottom: 20px;
 box-shadow: 0 12px 30px rgba(16,59,80,.12);
}
.hero h1 {font-family:'Manrope',sans-serif; font-size:32px; margin:0 0 8px 0; color:white;}
.hero p {font-size:15px; margin:0; color:#e0f3f2;}
.eyebrow {text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; color:#9ce6d4; margin-bottom:10px;}
.metric-card {background:white; border:1px solid #e4ebf2; border-radius:16px; padding:18px 20px; min-height:120px; box-shadow:0 4px 16px rgba(26,54,79,.04);}
.metric-label {color:#708297; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.8px;}
.metric-value {color:#17364e; font-family:'Manrope',sans-serif; font-size:29px; font-weight:800; margin:7px 0 3px;}
.metric-note {color:#8797a8; font-size:12px;}
.panel {background:white; border:1px solid #e4ebf2; border-radius:16px; padding:20px 22px; margin-bottom:16px;}
.panel-title {font-family:'Manrope',sans-serif; font-size:18px; font-weight:800; color:#17364e; margin-bottom:5px;}
.panel-sub {font-size:12px; color:#8392a3; margin-bottom:15px;}
div[data-testid="stMetric"] {background:white; border:1px solid #e4ebf2; border-radius:16px; padding:16px 18px;}
.stButton>button {border-radius:10px; font-weight:700; min-height:44px;}
div[data-testid="stFormSubmitButton"] button {background:#147d78; color:white; border:0;}
.small-muted {font-size:12px; color:#8191a3;}
.footer {color:#93a1b1; font-size:11px; text-align:center; padding:24px 0 8px;}
</style>
""", unsafe_allow_html=True)

FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
ALIASES = {
    "pregnancies":["pregnancies","kehamilan","hamil"],
    "glucose":["glucose","glukosa","gula darah"],
    "bloodpressure":["bloodpressure","blood pressure","tekanan darah"],
    "skinthickness":["skinthickness","skin thickness","ketebalan kulit"],
    "insulin":["insulin"],
    "bmi":["bmi","imt","body mass index","indeks massa tubuh"],
    "diabetespedigreefunction":["diabetespedigreefunction","diabetes pedigree function","pedigree"],
    "age":["age","umur","usia"],
    "outcome":["outcome","diabetes","target","label","class","hasil"]
}

def normalize_columns(df):
    ren={}
    for col in df.columns:
        c=str(col).strip().lower().replace("_"," ").replace("-"," ")
        c=" ".join(c.split())
        for key, vals in ALIASES.items():
            if c in vals or c.replace(" ","") in [v.replace(" ","") for v in vals]:
                ren[col] = "Outcome" if key=="outcome" else next((f for f in FEATURES if f.lower()==key.lower()), col)
                break
    return df.rename(columns=ren)

def demo_data(n=500, seed=17):
    # Simulated demonstration data only; not a clinical dataset.
    rng=np.random.default_rng(seed)
    pregnancies=rng.integers(0,12,n)
    glucose=np.clip(rng.normal(120,30,n),55,205)
    bp=np.clip(rng.normal(72,12,n),35,125)
    skin=np.clip(rng.normal(25,9,n),5,65)
    insulin=np.clip(rng.lognormal(4.4,.65,n),10,500)
    bmi=np.clip(rng.normal(31,7,n),16,55)
    dpf=np.clip(rng.gamma(2,.22,n),.05,2.2)
    age=rng.integers(21,80,n)
    score=(-7.3 + .025*glucose + .045*bmi + .018*age + .16*pregnancies + .3*dpf)
    prob=1/(1+np.exp(-score))
    outcome=(rng.random(n)<prob).astype(int)
    return pd.DataFrame({"Pregnancies":pregnancies,"Glucose":glucose.round(1),"BloodPressure":bp.round(1),
        "SkinThickness":skin.round(1),"Insulin":insulin.round(1),"BMI":bmi.round(1),
        "DiabetesPedigreeFunction":dpf.round(3),"Age":age,"Outcome":outcome})

def load_dataset(uploaded):
    if uploaded is None:
        return demo_data(), True
    try:
        df=pd.read_csv(uploaded)
        df=normalize_columns(df)
        missing=[c for c in FEATURES+["Outcome"] if c not in df.columns]
        if missing:
            st.error("Kolom dataset belum lengkap. Kolom yang belum ditemukan: "+", ".join(missing))
            return None, False
        df=df[FEATURES+["Outcome"]].copy()
        for c in FEATURES+["Outcome"]:
            df[c]=pd.to_numeric(df[c], errors="coerce")
        df=df.dropna()
        if len(df)<10:
            st.error("Dataset terlalu sedikit atau banyak nilai kosong. Gunakan minimal 10 baris data valid.")
            return None, False
        return df, False
    except Exception as e:
        st.error(f"File tidak dapat dibaca: {e}")
        return None, False

def train_model(df, k):
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    X=df[FEATURES].copy()
    y=df["Outcome"].astype(int)
    if y.nunique()<2:
        return None, None, None, None, "Kolom Outcome harus memiliki dua kelas, biasanya 0 dan 1."
    test_size=0.2 if len(df)>=30 else 0.3
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)
    kk=min(k, len(X_train))
    model=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),
                    ("knn",KNeighborsClassifier(n_neighbors=kk))])
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    return model, accuracy_score(y_test,pred), confusion_matrix(y_test,pred,labels=[0,1]), classification_report(y_test,pred,labels=[0,1],target_names=["Tidak diabetes","Diabetes"],output_dict=True,zero_division=0), None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🩺 KNN Diabetes")
    st.markdown("**HEALTH ANALYTICS**")
    st.markdown("---")
    page=st.radio("MENU UTAMA", ["Dashboard","Prediksi","Dataset","Tentang Aplikasi"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**Model aktif**")
    st.caption("K-Nearest Neighbors")
    st.caption("Klasifikasi berbasis data")
    st.markdown("---")
    st.caption("Prototype untuk pembelajaran. Bukan alat diagnosis medis.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">HEALTH ANALYTICS • MACHINE LEARNING</div>
  <h1>Kenali Data, Pahami Risiko.</h1>
  <p>Dashboard klasifikasi diabetes menggunakan algoritma K-Nearest Neighbors (KNN).</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.session_state.get("dataset_file")
if page == "Dataset":
    st.markdown('<div class="panel"><div class="panel-title">📂 Dataset</div><div class="panel-sub">Unggah CSV untuk menggunakan data milikmu.</div></div>', unsafe_allow_html=True)
    file=st.file_uploader("Pilih file dataset (.csv)", type=["csv"], key="dataset_uploader")
    if file is not None:
        st.session_state["dataset_file"]=file
        st.success(f"File diterima: {file.name}")
        uploaded=file
    else:
        st.info("Belum ada file diunggah. Aplikasi menggunakan data simulasi untuk demonstrasi.")
    st.markdown("**Format kolom yang diperlukan**")
    st.code("Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome", language="text")
    st.markdown("Kolom `Outcome` umumnya menggunakan `0` = tidak diabetes dan `1` = diabetes. Nama kolom bahasa Indonesia tertentu juga dikenali.")
else:
    file=st.file_uploader("Unggah dataset CSV (opsional)", type=["csv"], key="quick_uploader", help="Unggah dataset berisi 8 fitur dan kolom Outcome untuk memakai data asli.")
    if file is not None:
        st.session_state["dataset_file"]=file
        uploaded=file

df, is_demo=load_dataset(uploaded)
if df is None:
    st.stop()
source_label="DATA SIMULASI" if is_demo else "DATASET DIUNGGAH"
st.caption(f"SUMBER DATA: {source_label}  •  {len(df):,} baris valid")

k=st.sidebar.slider("Jumlah tetangga (K)", min_value=1, max_value=15, value=5, step=2)
model, accuracy, cm, report, error=train_model(df,k)
if error:
    st.error(error)
    st.stop()

if page=="Dashboard":
    st.markdown("### Ringkasan Dashboard")
    a,b,c,d=st.columns(4)
    total=len(df); pos=int((df["Outcome"]==1).sum()); neg=int((df["Outcome"]==0).sum())
    with a: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Data</div><div class="metric-value">{total:,}</div><div class="metric-note">Baris valid</div></div>',unsafe_allow_html=True)
    with b: st.markdown(f'<div class="metric-card"><div class="metric-label">Outcome 1</div><div class="metric-value">{pos:,}</div><div class="metric-note">Label positif dalam dataset</div></div>',unsafe_allow_html=True)
    with c: st.markdown(f'<div class="metric-card"><div class="metric-label">Outcome 0</div><div class="metric-value">{neg:,}</div><div class="metric-note">Label negatif dalam dataset</div></div>',unsafe_allow_html=True)
    with d: st.markdown(f'<div class="metric-card"><div class="metric-label">Akurasi Uji</div><div class="metric-value">{accuracy:.1%}</div><div class="metric-note">Pemisahan data uji 80:20 (umumnya)</div></div>',unsafe_allow_html=True)
    st.write("")
    left,right=st.columns([1.15,1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Distribusi Outcome</div><div class="panel-sub">Jumlah label pada dataset yang sedang digunakan</div>',unsafe_allow_html=True)
        chart=pd.DataFrame({"Kategori":["Tidak diabetes (0)","Diabetes (1)"],"Jumlah":[neg,pos]}).set_index("Kategori")
        st.bar_chart(chart, color="#168b82")
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Ringkasan Model</div><div class="panel-sub">Evaluasi sederhana model KNN</div>',unsafe_allow_html=True)
        st.metric("Akurasi data uji", f"{accuracy:.1%}")
        st.metric("Nilai K", str(k))
        st.write("Fitur yang dipakai:", ", ".join(FEATURES))
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">Cuplikan Dataset</div><div class="panel-sub">Lima baris pertama dari data aktif</div>',unsafe_allow_html=True)
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)
    if is_demo:
        st.warning("Data yang tampil adalah data simulasi, bukan rekam medis atau dataset penelitian. Hasil akurasi hanya contoh dan tidak mewakili kinerja klinis.")
elif page=="Prediksi":
    st.markdown("### Prediksi Individu")
    st.write("Masukkan nilai fitur untuk melihat keluaran model KNN yang dilatih dari dataset aktif.")
    if is_demo:
        st.warning("Model saat ini dilatih dari data simulasi. Gunakan dataset yang valid untuk eksperimen pembelajaran; hasil bukan diagnosis.")
    with st.form("prediction_form"):
        x1,x2,x3,x4=st.columns(4)
        with x1: pregnancies=st.number_input("Jumlah kehamilan",0,20,1)
        with x2: glucose=st.number_input("Glukosa",0.0,300.0,120.0,step=1.0)
        with x3: bp=st.number_input("Tekanan darah",0.0,200.0,70.0,step=1.0)
        with x4: skin=st.number_input("Ketebalan kulit",0.0,100.0,20.0,step=1.0)
        x5,x6,x7,x8=st.columns(4)
        with x5: insulin=st.number_input("Insulin",0.0,1000.0,80.0,step=1.0)
        with x6: bmi=st.number_input("BMI",0.0,100.0,28.0,step=0.5)
        with x7: dpf=st.number_input("Diabetes pedigree function",0.0,5.0,0.5,step=0.01)
        with x8: age=st.number_input("Usia",1,120,30)
        submitted=st.form_submit_button("Jalankan Prediksi", use_container_width=True)
    if submitted:
        sample=pd.DataFrame([[pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]],columns=FEATURES)
        pred=int(model.predict(sample)[0])
        proba=model.predict_proba(sample)[0]
        classes=list(model.named_steps["knn"].classes_)
        positive_prob=float(proba[classes.index(1)]) if 1 in classes else 0.0
        if pred==1:
            st.error("Keluaran model: kelas 1 — label diabetes")
        else:
            st.success("Keluaran model: kelas 0 — label tidak diabetes")
        st.metric("Probabilitas model untuk kelas 1", f"{positive_prob:.1%}")
        st.caption("Ini adalah keluaran algoritma dari dataset yang dipilih, bukan diagnosis atau kepastian kondisi kesehatan. Konsultasikan penilaian kesehatan kepada tenaga medis.")
        st.dataframe(sample, use_container_width=True, hide_index=True)
elif page=="Dataset":
    st.markdown("### Pratinjau & Statistik Dataset")
    st.metric("Jumlah baris valid", f"{len(df):,}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("#### Statistik deskriptif")
    st.dataframe(df[FEATURES].describe().T, use_container_width=True)
    st.download_button("Unduh data aktif sebagai CSV", df.to_csv(index=False).encode("utf-8"), "dataset_diabetes_aktif.csv", "text/csv")
elif page=="Tentang Aplikasi":
    st.markdown("### Tentang Aplikasi")
    st.markdown("""
<div class="panel">
<div class="panel-title">KNN Diabetes Classification</div>
<p>Aplikasi pembelajaran untuk memahami alur machine learning: memuat data, menyiapkan fitur, melatih model K-Nearest Neighbors, mengevaluasi hasil, dan mencoba prediksi.</p>
</div>
""",unsafe_allow_html=True)
    st.markdown("**Alur kerja**")
    st.write("1. Unggah dataset CSV yang memiliki 8 fitur dan kolom `Outcome`.")
    st.write("2. Aplikasi membersihkan nilai kosong, menstandarkan fitur, lalu melatih model KNN.")
    st.write("3. Hasil evaluasi menggunakan sebagian data yang dipisahkan untuk pengujian.")
    st.write("4. Buka menu Prediksi untuk mencoba input fitur.")
    st.info("Penting: aplikasi ini hanya prototipe edukasi, bukan perangkat medis. Jangan gunakan hasilnya untuk mendiagnosis atau menentukan pengobatan.")
st.markdown('<div class="footer">KNN DIABETES • HEALTH ANALYTICS • PROTOTIPE PEMBELAJARAN</div>',unsafe_allow_html=True)
