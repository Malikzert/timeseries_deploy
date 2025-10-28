import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.set_index('time')
    df['NO2'] = df['NO2'].interpolate(method='time')
    return df

def create_supervised_data(df, n_lags):
    supervised = pd.DataFrame()
    for i in range(n_lags, 0, -1):
        supervised[f'NO2(t-{i})'] = df['NO2'].shift(i)
    supervised['NO2(t)'] = df['NO2']
    supervised = supervised.dropna()

    X = supervised.drop('NO2(t)', axis=1)
    y = supervised['NO2(t)']

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
