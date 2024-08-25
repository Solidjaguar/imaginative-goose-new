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
from datetime import datetime, timedelta
import time
import joblib
from collections import defaultdict

# API keys
CURRENTS_API_KEY = "FkEEwNLACnLEfCoJ09fFe3yrVaPGZ76u-PKi8-yHqGRJ6hd8"
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your actual Alpha Vantage API key

def fetch_data(start_date, end_date):
    # ... (previous fetch_data function remains unchanged)

def fetch_news_sentiment(start_date, end_date, max_requests=600):
    # ... (previous fetch_news_sentiment function remains unchanged)

def fetch_recent_news(days=7):
    # ... (previous fetch_recent_news function remains unchanged)

def fetch_economic_calendar(start_date, end_date):
    # ... (previous fetch_economic_calendar function remains unchanged)

def create_features(df, economic_calendar):
    # ... (previous feature creation code)

    # Add features based on economic calendar
    important_countries = ['United States', 'China', 'European Union', 'Japan']
    important_events = ['GDP', 'Inflation Rate', 'Interest Rate Decision', 'Non-Farm Payrolls']
    
    for country in important_countries:
        for event in important_events:
            event_count = sum(1 for e in economic_calendar if e['country'] == country and event.lower() in e['event'].lower())
            df[f'{country}_{event}_Count'] = event_count

    # Add a feature for the overall number of high-impact events
    high_impact_count = sum(1 for e in economic_calendar if e['impact'] == 'High')
    df['High_Impact_Events_Count'] = high_impact_count

    df.dropna(inplace=True)
    return df

def train_model(model_name, X_train, y_train, params=None):
    # ... (previous train_model function remains unchanged)

def predict(model, model_name, X):
    # ... (previous predict function remains unchanged)

def evaluate_model(y_true, y_pred):
    # ... (previous evaluate_model function remains unchanged)

def plot_predictions(y_true, y_pred, title):
    # ... (previous plot_predictions function remains unchanged)

def ensemble_predict(predictions):
    # ... (previous ensemble_predict function remains unchanged)

def backtest_model(df, economic_calendar, model_names, start_date, end_date, window_size=30):
    """Backtest the model using expanding window approach."""
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    economic_calendar = [e for e in economic_calendar if start_date <= datetime.strptime(e['date'], "%Y-%m-%d") <= datetime.strptime(end_date, "%Y-%m-%d")]
    
    results = []
    for i in range(window_size, len(df)):
        train_df = df.iloc[:i]
        test_df = df.iloc[i:i+1]
        
        train_calendar = [e for e in economic_calendar if datetime.strptime(e['date'], "%Y-%m-%d") <= train_df.index[-1]]
        test_calendar = [e for e in economic_calendar if test_df.index[0] <= datetime.strptime(e['date'], "%Y-%m-%d") < test_df.index[0] + timedelta(days=7)]
        
        train_df = create_features(train_df, train_calendar)
        test_df = create_features(test_df, test_calendar)
        
        X_train, y_train = train_df.drop('Gold_Price', axis=1), train_df['Gold_Price']
        X_test, y_test = test_df.drop('Gold_Price', axis=1), test_df['Gold_Price']
        
        model_predictions = {}
        for model_name in model_names:
            model = train_model(model_name, X_train, y_train)
            pred = predict(model, model_name, X_test)
            model_predictions[model_name] = pred[0]
        
        ensemble_pred = ensemble_predict(list(model_predictions.values()))
        
        results.append({
            'date': test_df.index[0],
            'actual': y_test.values[0],
            'ensemble_prediction': ensemble_pred,
            **model_predictions
        })
        
        print(f"Processed up to {test_df.index[0]}")
    
    return pd.DataFrame(results)

def update_model(model, model_name, X_new, y_new):
    """Update the model with new data."""
    if model_name in ['rf', 'gb', 'xgb', 'lgbm']:
        # For tree-based models, we can partially fit with new data
        model.fit(X_new, y_new)
    elif model_name == 'lstm':
        # For LSTM, we need to retrain on all data
        model.fit(X_new, y_new, epochs=10, batch_size=32, verbose=0)
    # Add more conditions for other model types as needed
    return model

def assess_event_importance(backtest_results, economic_calendar):
    """Assess the importance of economic events based on prediction errors."""
    event_errors = defaultdict(list)
    
    for _, row in backtest_results.iterrows():
        date = row['date']
        error = abs(row['actual'] - row['ensemble_prediction'])
        
        # Find events within 7 days before the prediction date
        relevant_events = [e for e in economic_calendar if date - timedelta(days=7) <= datetime.strptime(e['date'], "%Y-%m-%d") < date]
        
        for event in relevant_events:
            event_key = (event['country'], event['event'])
            event_errors[event_key].append(error)
    
    # Calculate average error for each event type
    event_importance = {k: np.mean(v) for k, v in event_errors.items()}
    
    # Normalize importance scores
    max_importance = max(event_importance.values())
    event_importance = {k: v / max_importance for k, v in event_importance.items()}
    
    return event_importance

if __name__ == "__main__":
    # Fetch and prepare data
    df = fetch_data("2010-01-01", "2023-05-31")
    news_sentiment = fetch_news_sentiment("2010-01-01", "2023-05-31")
    df['News_Sentiment'] = news_sentiment
    
    # Fetch economic calendar data for the entire period
    economic_calendar = fetch_economic_calendar("2010-01-01", "2023-05-31")
    
    # Backtest the model
    model_names = ['rf', 'gb', 'xgb', 'lgbm', 'lstm']
    backtest_results = backtest_model(df, economic_calendar, model_names, "2015-01-01", "2023-05-31")
    
    # Evaluate backtest results
    for model_name in model_names + ['ensemble']:
        print(f"\n{model_name.upper()} Backtest Performance:")
        evaluate_model(backtest_results['actual'], backtest_results[f'{model_name}_prediction' if model_name == 'ensemble' else model_name])
    
    # Plot backtest results
    plot_predictions(backtest_results['actual'], backtest_results['ensemble_prediction'], "Backtest Results")
    
    # Assess event importance
    event_importance = assess_event_importance(backtest_results, economic_calendar)
    print("\nEvent Importance:")
    for event, importance in sorted(event_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{event[0]} - {event[1]}: {importance:.4f}")
    
    # Continuous model updating (simulation)
    print("\nSimulating continuous model updating...")
    latest_data = df.iloc[-30:]  # Last 30 days of data
    latest_calendar = [e for e in economic_calendar if datetime.strptime(e['date'], "%Y-%m-%d") >= latest_data.index[0]]
    latest_data = create_features(latest_data, latest_calendar)
    
    X_latest = latest_data.drop('Gold_Price', axis=1)
    y_latest = latest_data['Gold_Price']
    
    for model_name in model_names:
        model = joblib.load(f"{model_name}_model.joblib")  # Load the saved model
        updated_model = update_model(model, model_name, X_latest, y_latest)
        joblib.dump(updated_model, f"{model_name}_model_updated.joblib")  # Save the updated model
    
    print("Models updated with latest data.")
    
    # Fetch recent news and upcoming economic events for future predictions
    recent_news = fetch_recent_news()
    upcoming_calendar = fetch_economic_calendar(datetime.now().strftime("%Y-%m-%d"), (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"))
    
    print("\nRecent News Analysis:")
    for article in recent_news[:5]:  # Print top 5 recent news articles
        print(f"Title: {article['title']}")
        print(f"Sentiment: {TextBlob(article['title'] + ' ' + article['description']).sentiment.polarity}")
        print("---")

    print("\nUpcoming Economic Events:")
    for event in upcoming_calendar[:10]:  # Print top 10 upcoming events
        print(f"Date: {event['date']}")
        print(f"Country: {event['country']}")
        print(f"Event: {event['event']}")
        print(f"Impact: {event['impact']}")
        print(f"Importance Score: {event_importance.get((event['country'], event['event']), 0):.4f}")
        print("---")

print("\nNote: This ultra-advanced model now incorporates backtesting, continuous updating, and dynamic event importance assessment. However, it should still be used cautiously for actual trading decisions.")