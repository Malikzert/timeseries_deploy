# utils/forecast_utils.py
import numpy as np
import pandas as pd

def forecast_next_days(model, X_test, df, scaler, n_future):
    """
    Membuat prediksi untuk n hari ke depan menggunakan model time series.
    """
    last_input = X_test[-1].reshape(1, -1)
    preds = []

    for _ in range(n_future):
        pred = model.predict(last_input)
        pred_value = pred[0] if pred.ndim == 1 else pred[0, 0]
        preds.append(pred_value)

        # sliding window
        new_input = np.append(last_input[:, 1:], pred_value).reshape(1, -1)
        last_input = new_input

    # --- inverse transform hanya kolom target (NO2) ---
    min_val = scaler.data_min_[-1]  # ambil min dari kolom target
    max_val = scaler.data_max_[-1]  # ambil max dari kolom target

    preds_rescaled = [p * (max_val - min_val) + min_val for p in preds]

    # buat DataFrame hasil forecast (kolom disesuaikan dengan plot_utils)
    last_date = df.index[-1]
    forecast_dates = pd.date_range(last_date, periods=n_future + 1, freq='D')[1:]
    forecast_df = pd.DataFrame({
        'Tanggal': forecast_dates,
        'Prediksi_NO2': preds_rescaled
    })

    return forecast_df
