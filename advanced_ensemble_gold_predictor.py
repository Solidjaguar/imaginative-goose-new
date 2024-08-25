import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def fetch_data(start_date, end_date):
    # Fetch gold price data
    gold = yf.download("GC=F", start=start_date, end=end_date)
    
    # Fetch additional features
    usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)["Close"]
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
    oil = yf.download("CL=F", start=start_date, end=end_date)["Close"]
    
    # Fetch interest rates (10-year Treasury yield as a proxy)
    interest_rates = yf.download("^TNX", start=start_date, end=end_date)["Close"]
    
    # Fetch inflation data (CPI as a proxy, note this is monthly data)
    inflation = yf.download("CPI", start=start_date, end=end_date)["Close"]
    
    # Combine all features
    df = pd.DataFrame({
        "Gold_Price": gold["Close"],
        "USD_Index": usd_index,
        "SP500": sp500,
        "Oil_Price": oil,
        "Interest_Rate": interest_rates,
        "Inflation": inflation
    })
    
    # Forward fill inflation data to match daily frequency
    df['Inflation'].ffill(inplace=True)
    
    df.dropna(inplace=True)
    return df

def create_features(df):
    df['MA7'] = df['Gold_Price'].rolling(window=7).mean()
    df['MA30'] = df['Gold_Price'].rolling(window=30).mean()
    df['Volatility'] = df['Gold_Price'].rolling(window=30).std()
    df['USD_Index_Change'] = df['USD_Index'].pct_change()
    df['Oil_Price_Change'] = df['Oil_Price'].pct_change()
    df['SP500_Change'] = df['SP500'].pct_change()
    df['Interest_Rate_Change'] = df['Interest_Rate'].pct_change()
    df['Inflation_Change'] = df['Inflation'].pct_change()
    df['Gold_Returns'] = df['Gold_Price'].pct_change()
    df.dropna(inplace=True)
    return df

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_arima(data):
    model = ARIMA(data, order=(1,1,1))
    results = model.fit()
    return results

def train_prophet(data):
    df = data.reset_index()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    return model

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def prepare_lstm_data(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'R-squared Score: {r2:.2f}')
    
    return mse, rmse, mae, r2

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label='Actual')
    plt.plot(y_true.index, y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def ensemble_predict(rf_pred, arima_pred, prophet_pred, lstm_pred):
    return (rf_pred + arima_pred + prophet_pred + lstm_pred) / 4

if __name__ == "__main__":
    # Fetch and prepare data
    df = fetch_data("2010-01-01", "2023-05-31")
    df = create_features(df)
    
    # Prepare features
    features = ['USD_Index', 'SP500', 'Oil_Price', 'Interest_Rate', 'Inflation', 'MA7', 'MA30', 'Volatility', 
                'USD_Index_Change', 'Oil_Price_Change', 'SP500_Change', 'Interest_Rate_Change', 'Inflation_Change']
    X = df[features]
    y = df['Gold_Price']
    
    # Initialize arrays to store results
    rf_predictions = []
    arima_predictions = []
    prophet_predictions = []
    lstm_predictions = []
    actual_values = []
    
    # Perform rolling window backtesting
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Random Forest
        rf_model = train_random_forest(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_predictions.extend(rf_pred)
        
        # ARIMA
        arima_model = train_arima(y_train)
        arima_pred = arima_model.forecast(steps=len(y_test))
        arima_predictions.extend(arima_pred)
        
        # Prophet
        prophet_model = train_prophet(pd.DataFrame({'ds': y_train.index, 'y': y_train.values}))
        future_dates = prophet_model.make_future_dataframe(periods=len(y_test))
        prophet_forecast = prophet_model.predict(future_dates)
        prophet_pred = prophet_forecast['yhat'].iloc[-len(y_test):].values
        prophet_predictions.extend(prophet_pred)
        
        # LSTM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_lstm, y_train_lstm = prepare_lstm_data(X_train_scaled)
        X_test_lstm, y_test_lstm = prepare_lstm_data(X_test_scaled)
        
        lstm_model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
        lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
        
        lstm_pred = lstm_model.predict(X_test_lstm)
        lstm_predictions.extend(lstm_pred.flatten())
        
        actual_values.extend(y_test.values)
    
    # Convert predictions to numpy arrays
    rf_predictions = np.array(rf_predictions)
    arima_predictions = np.array(arima_predictions)
    prophet_predictions = np.array(prophet_predictions)
    lstm_predictions = np.array(lstm_predictions)
    actual_values = np.array(actual_values)
    
    # Create ensemble predictions
    ensemble_predictions = ensemble_predict(rf_predictions, arima_predictions, prophet_predictions, lstm_predictions)
    
    # Evaluate individual models
    print("Random Forest Performance:")
    evaluate_model(actual_values, rf_predictions)
    
    print("\nARIMA Performance:")
    evaluate_model(actual_values, arima_predictions)
    
    print("\nProphet Performance:")
    evaluate_model(actual_values, prophet_predictions)
    
    print("\nLSTM Performance:")
    evaluate_model(actual_values, lstm_predictions)
    
    print("\nEnsemble Model Performance:")
    ensemble_metrics = evaluate_model(actual_values, ensemble_predictions)
    
    # Plot predictions
    plot_predictions(pd.Series(actual_values, index=y.index[-len(actual_values):]), 
                     pd.Series(ensemble_predictions, index=y.index[-len(ensemble_predictions):]), 
                     "Ensemble Model Predictions")
    
    print("\nPrediction plot has been saved as a PNG file.")
    
    # Feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nRandom Forest Feature Importance:")
    print(feature_importance)

    print("\nNote: This advanced ensemble model uses real data and sophisticated techniques, but should still be used cautiously for actual trading decisions.")