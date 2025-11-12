import streamlit as st
import pandas as pd
import numpy as np
import pickle
# from sklearn.preprocessing import MinMaxScaler # <-- DIHAPUS
import openpyxl  # Diperlukan oleh pandas untuk membaca file Excel

# Menggunakan cache untuk data loading & preprocessing
@st.cache_data
def load_and_process_data(file_jkt, file_bgr, file_tma):
    """
    Fungsi ini mengambil semua langkah preprocessing data
    dari notebook Anda dan mengembalikannya sebagai DataFrame yang bersih.
    """
    try:
        # 1. Load Data
        jkt = pd.read_excel(file_jkt)
        bgr = pd.read_excel(file_bgr)
        tma = pd.read_excel(file_tma)

        # 2. Preprocessing 'jkt' (Stasiun Kemayoran)
        jkt = jkt.iloc[5:].reset_index(drop=True)
        if len(jkt) > 1:
            jkt.columns = jkt.iloc[1]
            jkt = jkt[2:].reset_index(drop=True)
        else: return None
        numeric_cols_jkt = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR']
        for col in numeric_cols_jkt:
            if col in jkt.columns: jkt[col] = pd.to_numeric(jkt[col], errors='coerce')
        if 'TANGGAL' not in jkt.columns: return None
        jkt['TANGGAL'] = pd.to_datetime(jkt['TANGGAL'], format='%d-%m-%Y', errors='coerce')
        jkt = jkt.sort_values('TANGGAL').reset_index(drop=True)
        jkt = jkt.dropna(subset=['TANGGAL'])
        jkt = jkt.drop_duplicates(subset=['TANGGAL'])
        jkt.replace(8888, np.nan, inplace=True)
        
        # Interpolasi data (kecuali RR)
        for col in ['TN', 'TX', 'TAVG', 'RH_AVG']:
            if col in jkt.columns: jkt[col] = jkt[col].interpolate(method='linear')
        
        # Ganti NaN di RR dengan 0 (sesuai notebook, Anda bisa juga interpolasi jika mau)
        if 'RR' in jkt.columns: jkt[col] = jkt[col].fillna(0) # Anda menggunakan interpolate() di notebook, tapi fillna(0) lebih aman jika NaN ada di awal/akhir
        
        # Pengecekan ulang NaN setelah interpolasi
        if jkt.isnull().values.any():
            jkt = jkt.fillna(method='bfill').fillna(method='ffill') # Isi sisa NaN

        # 3. Preprocessing 'bgr' (Stasiun Citeko)
        bgr = bgr.iloc[5:].reset_index(drop=True)
        if len(bgr) > 1:
            bgr.columns = bgr.iloc[1]
            bgr = bgr[2:].reset_index(drop=True)
        else: return None
        numeric_cols_bgr = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR']
        for col in numeric_cols_bgr:
             if col in bgr.columns: bgr[col] = pd.to_numeric(bgr[col], errors='coerce')
        if 'TANGGAL' not in bgr.columns: return None
        bgr['TANGGAL'] = pd.to_datetime(bgr['TANGGAL'], format='%d-%m-%Y', errors='coerce')
        bgr = bgr.sort_values('TANGGAL').reset_index(drop=True)
        bgr = bgr.dropna(subset=['TANGGAL'])
        bgr = bgr.drop_duplicates(subset=['TANGGAL'])
        bgr.replace(8888, np.nan, inplace=True)
        
        for col in ['TN', 'TX', 'TAVG', 'RH_AVG']:
             if col in bgr.columns: bgr[col] = bgr[col].interpolate(method='linear')
        
        if 'RR' in bgr.columns: bgr[col] = bgr[col].fillna(0)
        
        if bgr.isnull().values.any():
            bgr = bgr.fillna(method='bfill').fillna(method='ffill')

        # 4. Preprocessing 'tma' (TMA Banjir)
        if 'Tanggal' not in tma.columns or 'Banjir' not in tma.columns: return None
        tma['Tanggal'] = pd.to_datetime(tma['Tanggal'], format='%d-%m-%Y', errors='coerce')
        tma = tma.dropna(subset=['Tanggal'])
        tma['Banjir'] = pd.to_numeric(tma['Banjir'], errors='coerce')
        tma = tma.dropna(subset=['Banjir'])
        tma['Banjir'] = tma['Banjir'].astype(int)

        # 5. Gabungkan DataFrames (Gunakan inner join)
        gabung = pd.merge(jkt, bgr, on='TANGGAL', suffixes=('_JKT', '_BGR'), how='inner')
        df = pd.merge(gabung, tma, left_on='TANGGAL', right_on='Tanggal', how='inner')
        df.drop(columns=['Tanggal'], inplace=True) # Hapus kolom Tanggal duplikat

        # 6. Hapus baris NaN final (sangat penting)
        df = df.dropna()

        return df

    except FileNotFoundError as e:
        st.error(f"Error: File data tidak ditemukan. Pastikan file '{e.filename}' ada di folder yang sama.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memproses data: {e}")
        return None

# Menggunakan cache untuk memuat model (TANPA SCALER)
@st.cache_resource
def load_models(xgb_model_path="best_xgboost_model.pkl",
                rf_model_path="best_random_forest_model.pkl"):
    """
    Memuat model XGBoost terbaik dan model Random Forest terbaik
    dari file pickle. (TANPA SCALER)
    """
    try:
        with open(xgb_model_path, 'rb') as f_xgb_model:
            xgb_model = pickle.load(f_xgb_model)
        with open(rf_model_path, 'rb') as f_rf_model:
            rf_model = pickle.load(f_rf_model)

        st.success("Model XGBoost dan Model Random Forest berhasil dimuat.")
        return xgb_model, rf_model

    except FileNotFoundError as e:
        st.error(f"Error: File '{e.filename}' tidak ditemukan. Pastikan file model .pkl ada di folder ini.")
        return None, None
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        return None, None

# --- Halaman Utama Aplikasi ---
st.set_page_config(page_title="Prediksi Banjir Jakarta", layout="wide")
st.title("💧 Aplikasi Prediksi Banjir Jakarta")
st.write("Aplikasi ini menggunakan model Machine Learning (XGBoost & Random Forest) yang sudah dilatih untuk memprediksi potensi banjir.")

# Definisikan nama file data
file_jkt = 'Stasiun Kemayoran.xlsx'
file_bgr = 'Stasiun Citeko.xlsx'
file_tma = 'TMA Banjir.xlsx'

# Muat model (TANPA SCALER)
xgb_model, rf_model = load_models(
    xgb_model_path="best_xgboost_model.pkl",
    rf_model_path="best_random_forest_model.pkl"
)

# Hanya lanjutkan jika model berhasil dimuat
if xgb_model is not None and rf_model is not None:

    # Gunakan nama fitur ASLI (_JKT, _BGR) seperti saat training
    feature_names = [
        'TN_JKT', 'TX_JKT', 'TAVG_JKT', 'RH_AVG_JKT', 'RR_JKT',
        'TN_BGR', 'TX_BGR', 'TAVG_BGR', 'RH_AVG_BGR', 'RR_BGR',
        'Bendung Katulampa', 'Pos Depok', 'Manggarai BKB', 'PA. Karet'
    ]

    # --- Sidebar untuk Input Pengguna ---
    st.sidebar.header("Input Data untuk Prediksi")

    input_data = {}

    # Buat input fields dengan KEY yang sesuai feature_names (_JKT, _BGR)
    st.sidebar.subheader("Data Tinggi Muka Air (TMA)")
    input_data['Bendung Katulampa'] = st.sidebar.number_input("Bendung Katulampa (cm)", min_value=0.0, value=40.0, format="%.1f", key="katulampa")
    input_data['Pos Depok'] = st.sidebar.number_input("Pos Depok (cm)", min_value=0.0, value=110.0, format="%.1f", key="depok")
    input_data['Manggarai BKB'] = st.sidebar.number_input("Manggarai BKB (cm)", min_value=500.0, value=650.0, format="%.1f", key="manggarai")
    input_data['PA. Karet'] = st.sidebar.number_input("PA. Karet (cm)", min_value=200.0, value=300.0, format="%.1f", key="karet")

    st.sidebar.subheader("Data Cuaca Bogor (BGR/Citeko)") 
    input_data['TN_BGR'] = st.sidebar.number_input("Suhu Min Bogor (°C)", min_value=10.0, value=19.5, format="%.1f", key="tn_bgr")
    input_data['TX_BGR'] = st.sidebar.number_input("Suhu Max Bogor (°C)", min_value=15.0, value=26.0, format="%.1f", key="tx_bgr")
    input_data['TAVG_BGR'] = st.sidebar.number_input("Suhu Rata-rata Bogor (°C)", min_value=12.0, value=22.0, format="%.1f", key="tavg_bgr")
    input_data['RH_AVG_BGR'] = st.sidebar.number_input("Kelembaban Rata-rata Bogor (%)", min_value=50.0, max_value=100.0, value=85.0, format="%.1f", key="rh_bgr")
    input_data['RR_BGR'] = st.sidebar.number_input("Curah Hujan Bogor (mm)", min_value=0.0, value=15.0, format="%.1f", key="rr_bgr")

    st.sidebar.subheader("Data Cuaca Jakarta (JKT/Kemayoran)") 
    input_data['TN_JKT'] = st.sidebar.number_input("Suhu Min Jakarta (°C)", min_value=20.0, value=26.0, format="%.1f", key="tn_jkt")
    input_data['TX_JKT'] = st.sidebar.number_input("Suhu Max Jakarta (°C)", min_value=25.0, value=32.0, format="%.1f", key="tx_jkt")
    input_data['TAVG_JKT'] = st.sidebar.number_input("Suhu Rata-rata Jakarta (°C)", min_value=22.0, value=29.0, format="%.1f", key="tavg_jkt")
    input_data['RH_AVG_JKT'] = st.sidebar.number_input("Kelembaban Rata-rata Jakarta (%)", min_value=50.0, max_value=100.0, value=78.0, format="%.1f", key="rh_jkt")
    input_data['RR_JKT'] = st.sidebar.number_input("Curah Hujan Jakarta (mm)", min_value=0.0, value=5.0, format="%.1f", key="rr_jkt")

    # Tombol Prediksi
    predict_button = st.sidebar.button("Prediksi Sekarang", type="primary")

    # --- Halaman Utama (Main Page) ---

    st.header("📈 Hasil Prediksi")

    if predict_button:
        try:
            # Ubah input dictionary ke dataframe
            input_df = pd.DataFrame([input_data])
            # Pastikan urutan kolom sesuai dengan feature_names (_JKT, _BGR)
            input_df = input_df[feature_names]

            # --- HAPUS SCALING ---
            # input_scaled_xgb = xgb_scaler.transform(input_df) # <-- DIHAPUS
            # input_scaled_rf = rf_scaler.transform(input_df) # <-- DIHAPUS

            # --- Lakukan prediksi dengan KEDUA model (menggunakan input_df mentah) ---
            # XGBoost
            prediction_xgb = xgb_model.predict(input_df) # <-- DIUBAH
            prediction_proba_xgb = xgb_model.predict_proba(input_df) # <-- DIUBAH

            # Random Forest
            prediction_rf = rf_model.predict(input_df) # <-- DIUBAH
            prediction_proba_rf = rf_model.predict_proba(input_df) # <-- DIUBAH

            # --- Tampilkan Perbandingan Prediksi ---
            st.subheader("Perbandingan Prediksi Model")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### XGBoost (GridSearch 80/20)")
                st.success("🏆 **Model Terbaik**") # <-- TAMBAHAN
                if prediction_xgb[0] == 1:
                    st.error(f"**Potensi BANJIR**")
                else:
                    st.success(f"**Aman / Tidak Banjir**")
                st.metric("Probabilitas Banjir", f"{prediction_proba_xgb[0][1]:.2%}")
                st.metric("Probabilitas Aman", f"{prediction_proba_xgb[0][0]:.2%}")
                st.caption("Akurasi Test: **82.61%** (dari notebook)") # <-- DIUBAH (Akurasi terbaik)

            with col2:
                st.markdown("#### Random Forest (GridSearch 80/20)")
                if prediction_rf[0] == 1:
                    st.error(f"**Potensi BANJIR**")
                else:
                    st.success(f"**Aman / Tidak Banjir**")
                st.metric("Probabilitas Banjir", f"{prediction_proba_rf[0][1]:.2%}")
                st.metric("Probabilitas Aman", f"{prediction_proba_rf[0][0]:.2%}")
                st.caption("Akurasi Test: **80.43%** (dari notebook)") # <-- DIUBAH (Akurasi RF terbaik)

            # Menampilkan data input yang digunakan
            st.subheader("Data Input yang Digunakan:")
            st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi error saat melakukan prediksi: {e}")
            st.warning("Pastikan semua input di sidebar terisi dengan benar dan file model .pkl valid.")

    else:
        st.info("Masukkan data pada panel di sebelah kiri dan klik tombol 'Prediksi Sekarang'.")

    # (Opsional) Tampilkan cuplikan data jika berhasil diproses
    df_display = load_and_process_data(file_jkt, file_bgr, file_tma) # Muat ulang untuk display
    if df_display is not None:
        st.divider()
        st.header("📂 Cuplikan Data Gabungan (Setelah Preprocessing)")
        st.dataframe(df_display.head(), use_container_width=True)
        st.caption(f"Data ini digunakan untuk melatih model. Total baris bersih: {df_display.shape[0]}")
else:
    st.warning("Model tidak dapat dimuat. Aplikasi tidak dapat melakukan prediksi.")
    st.info("Pastikan file `best_xgboost_model.pkl` dan `best_random_forest_model.pkl` ada di folder yang sama.")