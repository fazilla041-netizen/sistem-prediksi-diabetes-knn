import math
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sistem Keputusan KNN",
    layout="wide",
)

# Dataset sesuai studi kasus pada PDF yang diberikan.
# Catatan: ini dataset contoh akademik, bukan data klinis untuk diagnosis.
DATA = [
    {"Pasien": "P01", "Usia": 25, "BMI": 21.5, "Glukosa": 90,  "Tekanan Darah": 110, "Riwayat Keluarga": 0, "Kelas": "Rendah"},
    {"Pasien": "P02", "Usia": 30, "BMI": 23.1, "Glukosa": 95,  "Tekanan Darah": 115, "Riwayat Keluarga": 0, "Kelas": "Rendah"},
    {"Pasien": "P03", "Usia": 35, "BMI": 25.0, "Glukosa": 105, "Tekanan Darah": 120, "Riwayat Keluarga": 0, "Kelas": "Rendah"},
    {"Pasien": "P04", "Usia": 40, "BMI": 27.5, "Glukosa": 120, "Tekanan Darah": 125, "Riwayat Keluarga": 1, "Kelas": "Tinggi"},
    {"Pasien": "P05", "Usia": 45, "BMI": 30.2, "Glukosa": 140, "Tekanan Darah": 135, "Riwayat Keluarga": 1, "Kelas": "Tinggi"},
    {"Pasien": "P06", "Usia": 50, "BMI": 31.5, "Glukosa": 155, "Tekanan Darah": 140, "Riwayat Keluarga": 1, "Kelas": "Tinggi"},
    {"Pasien": "P07", "Usia": 55, "BMI": 33.0, "Glukosa": 165, "Tekanan Darah": 145, "Riwayat Keluarga": 1, "Kelas": "Tinggi"},
    {"Pasien": "P08", "Usia": 28, "BMI": 22.5, "Glukosa": 92,  "Tekanan Darah": 112, "Riwayat Keluarga": 0, "Kelas": "Rendah"},
    {"Pasien": "P09", "Usia": 38, "BMI": 26.8, "Glukosa": 115, "Tekanan Darah": 122, "Riwayat Keluarga": 0, "Kelas": "Rendah"},
    {"Pasien": "P10", "Usia": 48, "BMI": 29.5, "Glukosa": 145, "Tekanan Darah": 130, "Riwayat Keluarga": 1, "Kelas": "Tinggi"},
]
FEATURES = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]

st.markdown("""
<style>
    .stApp {background: #f4f7fb;}
    .block-container {max-width: 1150px; padding-top: 1.7rem; padding-bottom: 2.5rem;}
    .hero {padding: 1.5rem 1.6rem; border-radius: 18px; color: white;
           background: linear-gradient(120deg,#172554,#2563eb); margin-bottom: 1rem;}
    .hero h1 {margin: 0 0 .45rem 0; font-size: clamp(1.65rem, 4vw, 2.35rem);}
    .hero p {margin: .2rem 0; color: #e0eaff;}
    .step {background: white; border: 1px solid #e1e8f0; border-radius: 14px;
           padding: 1rem 1.1rem; margin-bottom: .75rem;}
    div[data-testid="stMetric"] {background:white; border:1px solid #e1e8f0;
           padding: .8rem; border-radius: 12px;}
    @media (max-width: 640px) {
      .block-container {padding-left: 1rem; padding-right: 1rem;}
    }
</style>
<div class="hero">
  <h1>Sistem Keputusan Prediksi Risiko Diabetes</h1>
  <p>Sistem Pendukung Keputusan • Data Mining • Algoritma K-Nearest Neighbors (KNN)</p>
  <p>Masukkan data, jalankan proses perhitungan, dan lihat hasil klasifikasi.</p>
</div>
""", unsafe_allow_html=True)

st.warning("Untuk pembelajaran akademik dengan dataset contoh 10 pasien. Hasil bukan diagnosis medis dan tidak menggantikan pemeriksaan tenaga kesehatan.")


st.header("1. Input Data Pasien")
st.caption("Isi nilai pasien baru. Nilai awal di bawah mengikuti contoh pasien baru pada PDF.")
with st.form("knn_form"):
    c1, c2 = st.columns(2)
    with c1:
        usia = st.number_input("Usia (tahun)", min_value=1, max_value=100, value=42, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
        glukosa = st.number_input("Glukosa", min_value=1, max_value=500, value=130, step=1)
    with c2:
        tekanan = st.number_input("Tekanan darah", min_value=1, max_value=300, value=128, step=1)
        riwayat_label = st.selectbox("Riwayat diabetes dalam keluarga", ["Tidak ada", "Ada"], index=1)
        k = st.selectbox("Nilai K (jumlah tetangga)", [1, 3, 5], index=1)
    submit = st.form_submit_button("⚙️ Proses Prediksi KNN", use_container_width=True)

if submit:
    patient = [float(usia), float(bmi), float(glukosa), float(tekanan), 1.0 if riwayat_label == "Ada" else 0.0]
    frame = pd.DataFrame(DATA)
    # Standarisasi berdasarkan data contoh agar skala setiap atribut lebih sebanding.
    means = frame[FEATURES].mean()
    stds = frame[FEATURES].std(ddof=0).replace(0, 1)
    patient_scaled = [(patient[i] - means[FEATURES[i]]) / stds[FEATURES[i]] for i in range(len(FEATURES))]

    ranked = []
    for row in DATA:
        row_scaled = [(float(row[f]) - means[f]) / stds[f] for f in FEATURES]
        distance = math.sqrt(sum((patient_scaled[i] - row_scaled[i]) ** 2 for i in range(len(FEATURES))))
        ranked.append({**row, "Jarak Euclidean": distance})
    ranked.sort(key=lambda item: item["Jarak Euclidean"])
    nearest = ranked[:int(k)]
    high_votes = sum(1 for row in nearest if row["Kelas"] == "Tinggi")
    low_votes = len(nearest) - high_votes
    result = "Tinggi" if high_votes > low_votes else "Rendah"

    st.header("2. Output Perhitungan")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Tetangga Risiko Rendah", low_votes)
    metric2.metric("Tetangga Risiko Tinggi", high_votes)
    metric3.metric("Nilai K", k)

    st.subheader("3. Data input yang diproses")
    input_table = pd.DataFrame([{
        "Usia": usia, "BMI": bmi, "Glukosa": glukosa,
        "Tekanan Darah": tekanan,
        "Riwayat Keluarga": "Ada" if riwayat_label == "Ada" else "Tidak ada"
    }])
    st.dataframe(input_table, use_container_width=True, hide_index=True)

    st.subheader("4. Semua data berdasarkan jarak terdekat")
    result_df = pd.DataFrame(ranked)
    result_df = result_df.rename(columns={"Riwayat Keluarga": "Riwayat Keluarga"})
    result_df["Jarak Euclidean"] = result_df["Jarak Euclidean"].map(lambda x: round(x, 4))
    st.dataframe(result_df[["Pasien", *FEATURES, "Kelas", "Jarak Euclidean"]], use_container_width=True, hide_index=True)

    st.subheader(f"{k} Tetangga Terdekat")
    near_df = pd.DataFrame(nearest)
    near_df["Jarak Euclidean"] = near_df["Jarak Euclidean"].map(lambda x: round(x, 4))
    st.dataframe(near_df[["Pasien", *FEATURES, "Kelas", "Jarak Euclidean"]], use_container_width=True, hide_index=True)

    st.header("5. Hasil Keputusan")
    if result == "Tinggi":
        st.error("Hasil klasifikasi dataset: RISIKO TINGGI")
    else:
        st.success("Hasil klasifikasi dataset: RISIKO RENDAH")
    st.write(
        f"Dari {k} tetangga terdekat, kelas Rendah memperoleh **{low_votes} suara** dan "
        f"kelas Tinggi memperoleh **{high_votes} suara**. Berdasarkan suara mayoritas, "
        f"hasil klasifikasi KNN adalah **Risiko {result}**."
    )
    st.caption("Penting: hasil hanya keluaran algoritma pada dataset contoh akademik, bukan diagnosis atau penilaian medis.")

with st.expander("Lihat Dataset Acuan (10 Pasien)"):
    st.dataframe(pd.DataFrame(DATA), use_container_width=True, hide_index=True)

st.divider()
st.caption("Mini Project Data Mining • Sistem Pendukung Keputusan KNN • Dataset contoh untuk pembelajaran")