import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def plot_actual_vs_pred(y_test, y_pred, k):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.index, y_test, label='Aktual', color='darkorange')
    ax.plot(y_test.index, y_pred, label='Prediksi', color='navy')
    ax.set_title(f"KNN Forecasting NO₂ (K={k})")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Kadar NO₂")
    ax.legend()
    return fig

def plot_forecast(df, forecast_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['NO2'], label="Data Historis", color='gray')
    ax.plot(forecast_df['Tanggal'], forecast_df['Prediksi_NO2'], 'ro-', label="Prediksi NO₂")
    ax.set_title("Prediksi NO₂ Beberapa Hari ke Depan")
    ax.legend()
    return fig

def plot_residual_acf(y_test, y_pred):
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_acf(residuals, lags=20, ax=ax)
    ax.set_title("Autokorelasi Residual")
    return fig
