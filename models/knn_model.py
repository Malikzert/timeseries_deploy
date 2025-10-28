from sklearn import neighbors
import pandas as pd
import numpy as np

def train_knn_model(X_train, y_train, n_neighbors=5):
    model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    model.fit(X_train, y_train)
    return model

def forecast_next_days(model, X_test, df, scaler, n_future):
    last_input = X_test[-1, :].reshape(1, -1)
    future_preds = []

    for _ in range(n_future):
        next_pred = model.predict(last_input)[0]
        future_preds.append(next_pred)
        last_input = np.roll(last_input, -1)
        last_input[0, -1] = next_pred

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_future)
    forecast_df = pd.DataFrame({"Tanggal": future_dates, "Prediksi_NO2": np.array(future_preds)})
    return forecast_df
