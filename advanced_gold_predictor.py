import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import yfinance as yf

def fetch_data(start_date, end_date):
    # Fetch gold price data
    gold = yf.download("GC=F", start=start_date, end=end_date)
    
    # Fetch additional features
    usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)["Close"]
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
    oil = yf.download("CL=F", start=start_date, end=end_date)["Close"]
    
    # Combine all features
    df = pd.DataFrame({
        "Gold_Price": gold["Close"],
        "USD_Index": usd_index,
        "SP500": sp500,
        "Oil_Price": oil
    })
    
    df.dropna(inplace=True)
    return df

def create_features(df):
    df['MA7'] = df['Gold_Price'].rolling(window=7).mean()
    df['MA30'] = df['Gold_Price'].rolling(window=30).mean()
    df['Volatility'] = df['Gold_Price'].rolling(window=30).std()
    df['USD_Index_Change'] = df['USD_Index'].pct_change()
    df['Oil_Price_Change'] = df['Oil_Price'].pct_change()
    df['SP500_Change'] = df['SP500'].pct_change()
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

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'R-squared Score: {r2:.2f}')

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

if __name__ == "__main__":
    # Fetch and prepare data
    df = fetch_data("2010-01-01", "2023-05-31")
    df = create_features(df)
    
    # Prepare features for Random Forest
    features = ['USD_Index', 'SP500', 'Oil_Price', 'MA7', 'MA30', 'Volatility', 
                'USD_Index_Change', 'Oil_Price_Change', 'SP500_Change']
    X = df[features]
    y = df['Gold_Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    print("Random Forest Performance:")
    evaluate_model(y_test, rf_predictions)
    plot_predictions(y_test, rf_predictions, "Random Forest Predictions")
    
    # ARIMA
    arima_model = train_arima(df['Gold_Price'])
    arima_predictions = arima_model.forecast(steps=len(y_test))
    print("\nARIMA Performance:")
    evaluate_model(y_test, arima_predictions)
    plot_predictions(y_test, arima_predictions, "ARIMA Predictions")
    
    # Prophet
    prophet_model = train_prophet(df['Gold_Price'])
    future_dates = prophet_model.make_future_dataframe(periods=len(y_test))
    prophet_forecast = prophet_model.predict(future_dates)
    prophet_predictions = prophet_forecast['yhat'][-len(y_test):]
    print("\nProphet Performance:")
    evaluate_model(y_test, prophet_predictions)
    plot_predictions(y_test, prophet_predictions, "Prophet Predictions")

    print("\nPrediction plots have been saved as PNG files.")
    
    # Feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nRandom Forest Feature Importance:")
    print(feature_importance)

    print("\nNote: This model uses real data and more advanced techniques, but should still be used cautiously for actual trading decisions.")