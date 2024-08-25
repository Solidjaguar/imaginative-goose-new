import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import requests
from textblob import TextBlob
import talib

def fetch_data(start_date, end_date):
    # Fetch gold price data
    gold = yf.download("GC=F", start=start_date, end=end_date)
    
    # Fetch additional features
    usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)["Close"]
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
    oil = yf.download("CL=F", start=start_date, end=end_date)["Close"]
    vix = yf.download("^VIX", start=start_date, end=end_date)["Close"]
    interest_rates = yf.download("^TNX", start=start_date, end=end_date)["Close"]
    inflation = yf.download("CPI", start=start_date, end=end_date)["Close"]
    
    # Fetch cryptocurrency data (Bitcoin as a proxy)
    btc = yf.download("BTC-USD", start=start_date, end=end_date)["Close"]
    
    # Combine all features
    df = pd.DataFrame({
        "Gold_Price": gold["Close"],
        "USD_Index": usd_index,
        "SP500": sp500,
        "Oil_Price": oil,
        "VIX": vix,
        "Interest_Rate": interest_rates,
        "Inflation": inflation,
        "Bitcoin": btc
    })
    
    # Forward fill missing data
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def fetch_news_sentiment(start_date, end_date):
    # This is a placeholder function. In a real scenario, you would use a news API
    # to fetch relevant news articles and perform sentiment analysis.
    # For this example, we'll generate random sentiment scores.
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiments = np.random.normal(0, 1, size=len(date_range))
    return pd.Series(sentiments, index=date_range)

def create_features(df):
    # Basic features
    df['MA7'] = df['Gold_Price'].rolling(window=7).mean()
    df['MA30'] = df['Gold_Price'].rolling(window=30).mean()
    df['Volatility'] = df['Gold_Price'].rolling(window=30).std()
    
    # Percentage changes
    for col in df.columns:
        df[f'{col}_Change'] = df[col].pct_change()
    
    # Technical indicators
    df['RSI'] = talib.RSI(df['Gold_Price'])
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Gold_Price'])
    df['ATR'] = talib.ATR(df['Gold_Price'], df['Gold_Price'], df['Gold_Price'])
    
    # Fourier transforms for cyclical patterns
    for period in [30, 90, 365]:
        fourier = np.fft.fft(df['Gold_Price'])
        frequencies = np.fft.fftfreq(len(df['Gold_Price']))
        indices = np.argsort(frequencies)
        top_indices = indices[-period:]
        restored_sig = np.fft.ifft(fourier[top_indices])
        df[f'Fourier_{period}'] = np.real(restored_sig)
    
    # Lag features
    for lag in [1, 7, 30]:
        df[f'Gold_Price_Lag_{lag}'] = df['Gold_Price'].shift(lag)
    
    df.dropna(inplace=True)
    return df

def optimize_model(model_name, X, y):
    def objective(trial):
        if model_name == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            model = RandomForestRegressor(**params)
        elif model_name == 'gb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
                'max_depth': trial.suggest_int('max_depth', 3, 10)
            }
            model = GradientBoostingRegressor(**params)
        elif model_name == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0)
            }
            model = XGBRegressor(**params)
        elif model_name == 'lgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0)
            }
            model = LGBMRegressor(**params)
        
        scores = []
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            scores.append(mean_squared_error(y_val, pred))
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    return study.best_params

def train_model(model_name, X_train, y_train, params=None):
    if model_name == 'rf':
        model = RandomForestRegressor(**params) if params else RandomForestRegressor()
    elif model_name == 'gb':
        model = GradientBoostingRegressor(**params) if params else GradientBoostingRegressor()
    elif model_name == 'xgb':
        model = XGBRegressor(**params) if params else XGBRegressor()
    elif model_name == 'lgbm':
        model = LGBMRegressor(**params) if params else LGBMRegressor()
    elif model_name == 'sarimax':
        model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,12))
    elif model_name == 'prophet':
        model = Prophet()
    elif model_name == 'lstm':
        model = Sequential([
            Bidirectional(LSTM(50, activation='relu', return_sequences=True), input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            Bidirectional(LSTM(50, activation='relu')),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    if model_name in ['rf', 'gb', 'xgb', 'lgbm']:
        model.fit(X_train, y_train)
    elif model_name == 'sarimax':
        model = model.fit()
    elif model_name == 'prophet':
        model.fit(pd.DataFrame({'ds': X_train.index, 'y': y_train}))
    elif model_name == 'lstm':
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
        model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    return model

def predict(model, model_name, X):
    if model_name in ['rf', 'gb', 'xgb', 'lgbm']:
        return model.predict(X)
    elif model_name == 'sarimax':
        return model.forecast(steps=len(X))
    elif model_name == 'prophet':
        future = model.make_future_dataframe(periods=len(X))
        forecast = model.predict(future)
        return forecast['yhat'].iloc[-len(X):]
    elif model_name == 'lstm':
        X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
        return model.predict(X_reshaped).flatten()

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

def ensemble_predict(predictions):
    return np.mean(predictions, axis=0)

if __name__ == "__main__":
    # Fetch and prepare data
    df = fetch_data("2010-01-01", "2023-05-31")
    news_sentiment = fetch_news_sentiment("2010-01-01", "2023-05-31")
    df['News_Sentiment'] = news_sentiment
    df = create_features(df)
    
    # Prepare features and target
    target = 'Gold_Price'
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]
    
    # Initialize arrays to store results
    model_names = ['rf', 'gb', 'xgb', 'lgbm', 'sarimax', 'prophet', 'lstm']
    all_predictions = {model: [] for model in model_names}
    actual_values = []
    
    # Perform rolling window backtesting
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        for model_name in model_names:
            if model_name in ['rf', 'gb', 'xgb', 'lgbm']:
                params = optimize_model(model_name, X_train_scaled, y_train)
                model = train_model(model_name, X_train_scaled, y_train, params)
            else:
                model = train_model(model_name, X_train_scaled, y_train)
            
            pred = predict(model, model_name, X_test_scaled)
            all_predictions[model_name].extend(pred)
        
        actual_values.extend(y_test.values)
    
    # Convert predictions to numpy arrays
    for model in model_names:
        all_predictions[model] = np.array(all_predictions[model])
    actual_values = np.array(actual_values)
    
    # Create ensemble predictions
    ensemble_predictions = ensemble_predict([all_predictions[model] for model in model_names])
    
    # Evaluate individual models and ensemble
    for model in model_names:
        print(f"\n{model.upper()} Performance:")
        evaluate_model(actual_values, all_predictions[model])
    
    print("\nEnsemble Model Performance:")
    ensemble_metrics = evaluate_model(actual_values, ensemble_predictions)
    
    # Plot predictions
    plot_predictions(pd.Series(actual_values, index=y.index[-len(actual_values):]), 
                     pd.Series(ensemble_predictions, index=y.index[-len(ensemble_predictions):]), 
                     "Ensemble Model Predictions")
    
    # Feature importance for Random Forest
    rf_model = train_model('rf', X, y)
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nRandom Forest Feature Importance:")
    print(feature_importance)
    
    # Seasonal Decomposition
    result = seasonal_decompose(df['Gold_Price'], model='multiplicative')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    result.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    result.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    result.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    result.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    plt.savefig('seasonal_decomposition.png')
    plt.close()
    
    print("\nSeasonal decomposition plot has been saved as 'seasonal_decomposition.png'.")
    print("\nNote: This ultra-advanced model uses sophisticated techniques and multiple data sources, but should still be used cautiously for actual trading decisions.")