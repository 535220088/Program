import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import openpyxl  # Diperlukan oleh pandas untuk membaca file Excel

# Menggunakan cache untuk data loading & preprocessing agar lebih cepat
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
        jkt.columns = jkt.iloc[1]
        jkt = jkt[2:].reset_index(drop=True)
        numeric_cols_jkt = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR']
        for col in numeric_cols_jkt:
            jkt[col] = pd.to_numeric(jkt[col], errors='coerce')
        jkt['TANGGAL'] = pd.to_datetime(jkt['TANGGAL'], format='%d-%m-%Y', errors='coerce')
        jkt = jkt.sort_values('TANGGAL').reset_index(drop=True)
        jkt = jkt.dropna(subset=['TANGGAL'])
        jkt = jkt.drop_duplicates(subset=['TANGGAL'])
        jkt.replace(8888, np.nan, inplace=True)
        for col in ['TN', 'TX', 'TAVG', 'RH_AVG']:
            jkt[col] = jkt[col].interpolate()
        jkt['RR'] = jkt['RR'].fillna(0)

        # 3. Preprocessing 'bgr' (Stasiun Citeko)
        bgr = bgr.iloc[5:].reset_index(drop=True)
        bgr.columns = bgr.iloc[1]
        bgr = bgr[2:].reset_index(drop=True)
        numeric_cols_bgr = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR']
        for col in numeric_cols_bgr:
            bgr[col] = pd.to_numeric(bgr[col], errors='coerce')
        bgr['TANGGAL'] = pd.to_datetime(bgr['TANGGAL'], format='%d-%m-%Y', errors='coerce')
        bgr = bgr.sort_values('TANGGAL').reset_index(drop=True)
        bgr = bgr.dropna(subset=['TANGGAL'])
        bgr = bgr.drop_duplicates(subset=['TANGGAL'])
        bgr.replace(8888, np.nan, inplace=True)
        for col in ['TN', 'TX', 'TAVG', 'RH_AVG']:
            bgr[col] = bgr[col].interpolate()
        bgr['RR'] = bgr['RR'].fillna(0)

        # 4. Preprocessing 'tma' (TMA Banjir)
        tma['Tanggal'] = pd.to_datetime(tma['Tanggal'], format='%d-%m-%Y', errors='coerce')

        # 5. Gabungkan DataFrames
        gabung = pd.merge(tma, bgr, left_on='Tanggal', right_on='TANGGAL', how='inner')
        gabung = gabung.drop(columns=['TANGGAL'])
        df = pd.merge(gabung, jkt, left_on='Tanggal', right_on='TANGGAL', how='inner')
        df = df.drop(columns=['TANGGAL'])

        # 6. Rename kolom agar lebih mudah dibaca
        df = df.rename(columns={
            'TN_x': 'TN_Bogor', 'TX_x': 'TX_Bogor', 'TAVG_x': 'TAVG_Bogor', 'RH_AVG_x': 'RH_AVG_Bogor', 'RR_x': 'RR_Bogor',
            'TN_y': 'TN_Jakarta', 'TX_y': 'TX_Jakarta', 'TAVG_y': 'TAVG_Jakarta', 'RH_AVG_y': 'RH_AVG_Jakarta', 'RR_y': 'RR_Jakarta'
        })
        
        return df
    
    except FileNotFoundError as e:
        st.error(f"Error: File data tidak ditemukan. Pastikan file '{e.filename}' ada di folder yang sama dengan `app.py`.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memproses data: {e}")
        return None

# Menggunakan cache untuk training model agar tidak perlu training ulang setiap kali user berinteraksi
@st.cache_resource
def train_model(df):
    """
    Fungsi ini melatih model XGBoost terbaik berdasarkan
    parameter yang ditemukan di notebook Anda.
    """
    # Pisahkan fitur (X) dan target (y)
    X = df.drop(columns=['Tanggal', 'Banjir'])
    y = df['Banjir']
    
    # Simpan nama fitur untuk digunakan di input
    feature_names = X.columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ambil parameter terbaik dari GridSearchCV di notebook Anda
    # {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0}
    best_params = {
        'colsample_bytree': 0.8,
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 200,
        'subsample': 1.0
    }

    # Latih model XGBoost Tuned
    model = XGBClassifier(
        **best_params,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return scaler, model, report, cm, feature_names

# --- Halaman Utama Aplikasi ---
st.set_page_config(page_title="Prediksi Banjir Jakarta", layout="wide")
st.title("💧 Aplikasi Prediksi Banjir Jakarta")
st.write("Aplikasi ini menggunakan model Machine Learning (XGBoost) untuk memprediksi potensi banjir berdasarkan data cuaca dan tinggi muka air.")

# Load dan proses data
# Ganti nama file ini jika nama file Anda berbeda
file_jkt = 'Stasiun Kemayoran.xlsx'
file_bgr = 'Stasiun Citeko.xlsx'
file_tma = 'TMA Banjir.xlsx'

df = load_and_process_data(file_jkt, file_bgr, file_tma)

if df is not None:
    # Latih model
    scaler, model, report, cm, feature_names = train_model(df)

    # --- Sidebar untuk Input Pengguna ---
    st.sidebar.header("Input Data untuk Prediksi")
    
    input_data = {}
    
    # Buat input fields secara dinamis
    st.sidebar.subheader("Data Tinggi Muka Air (TMA)")
    input_data['Bendung Katulampa'] = st.sidebar.number_input("Bendung Katulampa (cm)", min_value=0.0, value=20.0, format="%.1f")
    input_data['Pos Depok'] = st.sidebar.number_input("Pos Depok (cm)", min_value=0.0, value=95.0, format="%.1f")
    input_data['Manggarai BKB'] = st.sidebar.number_input("Manggarai BKB (cm)", min_value=0.0, value=620.0, format="%.1f")
    input_data['PA. Karet'] = st.sidebar.number_input("PA. Karet (cm)", min_value=0.0, value=260.0, format="%.1f")
    
    st.sidebar.subheader("Data Cuaca Bogor")
    input_data['TN_Bogor'] = st.sidebar.number_input("Suhu Min Bogor (°C)", min_value=0.0, value=19.6, format="%.1f")
    input_data['TX_Bogor'] = st.sidebar.number_input("Suhu Max Bogor (°C)", min_value=0.0, value=25.8, format="%.1f")
    input_data['TAVG_Bogor'] = st.sidebar.number_input("Suhu Rata-rata Bogor (°C)", min_value=0.0, value=22.2, format="%.1f")
    input_data['RH_AVG_Bogor'] = st.sidebar.number_input("Kelembaban Rata-rata Bogor (%)", min_value=0.0, value=92.0, format="%.1f")
    input_data['RR_Bogor'] = st.sidebar.number_input("Curah Hujan Bogor (mm)", min_value=0.0, value=23.5, format="%.1f")

    st.sidebar.subheader("Data Cuaca Jakarta")
    input_data['TN_Jakarta'] = st.sidebar.number_input("Suhu Min Jakarta (°C)", min_value=0.0, value=26.4, format="%.1f")
    input_data['TX_Jakarta'] = st.sidebar.number_input("Suhu Max Jakarta (°C)", min_value=0.0, value=32.2, format="%.1f")
    input_data['TAVG_Jakarta'] = st.sidebar.number_input("Suhu Rata-rata Jakarta (°C)", min_value=0.0, value=29.6, format="%.1f")
    input_data['RH_AVG_Jakarta'] = st.sidebar.number_input("Kelembaban Rata-rata Jakarta (%)", min_value=0.0, value=77.0, format="%.1f")
    input_data['RR_Jakarta'] = st.sidebar.number_input("Curah Hujan Jakarta (mm)", min_value=0.0, value=1.8, format="%.1f")

    # Tombol Prediksi
    predict_button = st.sidebar.button("Prediksi Sekarang", type="primary")

    # --- Halaman Utama (Main Page) ---

    # Kolom untuk Hasil Prediksi
    col1, col2 = st.columns([1.5, 2])

    with col1:
        st.header("📈 Hasil Prediksi")
        if predict_button:
            # Ubah input dictionary ke dataframe
            input_df = pd.DataFrame([input_data])
            # Pastikan urutan kolom sesuai dengan saat training
            input_df = input_df[feature_names] 
            
            # Scaling data input
            input_scaled = scaler.transform(input_df)
            
            # Lakukan prediksi
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            st.subheader("Hasil Prediksi:")
            if prediction[0] == 1:
                st.error(f"**Potensi BANJIR**")
                st.write(f"**Probabilitas Banjir:** `{prediction_proba[0][1]:.2%}`")
                st.write(f"**Probabilitas Aman:** `{prediction_proba[0][0]:.2%}`")
            else:
                st.success(f"**Aman / Tidak Banjir**")
                st.write(f"**Probabilitas Aman:** `{prediction_proba[0][0]:.2%}`")
                st.write(f"**Probabilitas Banjir:** `{prediction_proba[0][1]:.2%}`")

        else:
            st.info("Masukkan data pada panel di sebelah kiri dan klik tombol 'Prediksi Sekarang'.")

    # Kolom untuk Performa Model
    with col2:
        st.header("📊 Performa Model (XGBoost Tuned)")
        
        st.subheader("Classification Report (Data Uji)")
        # Tampilkan classification report
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        st.caption(f"Akurasi model pada data uji: **{report['accuracy']:.4f}**")

        st.subheader("Confusion Matrix (Data Uji)")
        # Tampilkan confusion matrix
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Aman (0)', 'Banjir (1)'])
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title("Confusion Matrix - XGBoost Tuned")
        st.pyplot(fig)


    st.divider()

    # Tampilkan cuplikan data
    st.header("📂 Cuplikan Data Gabungan")
    st.write("Data berikut adalah hasil penggabungan dan pembersihan dari 3 file Excel.")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"Total data yang digunakan untuk melatih model: {df.shape[0]} baris.")