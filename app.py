import sys, os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

from utils.data_utils import load_and_prepare_data, create_supervised_data
from utils.plot_utils import plot_actual_vs_pred, plot_forecast, plot_residual_acf
from utils.metrics_utils import evaluate_model
from utils.forecast_utils import forecast_next_days
from models.knn_model import train_knn_model


# --- Konfigurasi Tampilan ---
st.set_page_config(page_title="Prediksi NO₂ Sumenep", layout="wide")
st.title("🌫️ Prediksi Kadar NO₂ Sumenep Menggunakan KNN Time Series")

# --- Baca dataset lokal ---
DATA_PATH = r"data\NO2_Sumenep.csv"

try:
    df = load_and_prepare_data(DATA_PATH)
except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()

# --- Tampilkan data awal ---
st.subheader("📈 Data Historis NO₂")
st.line_chart(df['NO2'])

# --- Sidebar pengaturan model ---
st.sidebar.header("⚙️ Pengaturan Model")
n_lags = st.sidebar.slider("Jumlah Lag Hari Sebelumnya", 3, 14, 7)
n_neighbors = st.sidebar.slider("Jumlah Tetangga (K)", 2, 15, 5)
n_future = st.sidebar.slider("Hari Prediksi ke Depan", 1, 14, 7)

# --- Persiapan data supervised ---
X_train, X_test, y_train, y_test, scaler = create_supervised_data(df, n_lags)

# --- Training model KNN ---
model = train_knn_model(X_train, y_train, n_neighbors)

# --- Evaluasi model ---
y_pred, rmse, r2, mape = evaluate_model(model, X_test, y_test)

st.subheader("📊 Evaluasi Model")
st.write(f"**RMSE:** {rmse:.6f}")
st.write(f"**R²:** {r2:.4f}")
st.write(f"**MAPE:** {mape:.2f}%")

st.pyplot(plot_actual_vs_pred(y_test, y_pred, n_neighbors))

# --- Prediksi ke depan ---
st.subheader(f"🔮 Prediksi {n_future} Hari ke Depan")
forecast_df = forecast_next_days(model, X_test, df, scaler, n_future)
st.dataframe(forecast_df)
st.pyplot(plot_forecast(df, forecast_df))

# --- Analisis residual ---
st.subheader("🔍 Analisis Autokorelasi Residual")
st.pyplot(plot_residual_acf(y_test, y_pred))

# --- Tombol unduh hasil ---
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Unduh Hasil Prediksi (CSV)", csv, "forecast_no2.csv", "text/csv")
