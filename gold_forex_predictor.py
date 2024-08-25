import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import io
import base64
import logging
from trading_strategy import calculate_indicators

logging.basicConfig(filename='gold_forex_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_all_data(start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*2)
    if end_date is None:
        end_date = datetime.now()

    gold = yf.Ticker("GC=F")
    gold_data = gold.history(start=start_date, end=end_date)

    currency_pairs = ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X']
    fx_data = {}

    for pair in currency_pairs:
        currency = yf.Ticker(pair)
        fx_data[pair] = currency.history(start=start_date, end=end_date)

    return {
        'Gold': gold_data['Close'],
        'EUR/USD': fx_data['EURUSD=X']['Close'],
        'GBP/USD': fx_data['GBPUSD=X']['Close'],
        'JPY/USD': 1 / fx_data['JPYUSD=X']['Close']
    }

def prepare_data(data):
    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    for column in df.columns:
        df[f'{column}_returns'] = df[column].pct_change()

    indicators = calculate_indicators(df)
    df = pd.concat([df, indicators], axis=1)

    X = df.drop(['Gold', 'EUR/USD', 'GBP/USD', 'JPY/USD'], axis=1).dropna()
    y = df[['EUR/USD_returns', 'GBP/USD_returns', 'JPY/USD_returns']].shift(-1).dropna()

    X = X[:-1]
    y = y[:-1]

    return X, y

def train_model():
    data = fetch_all_data()
    X, y = prepare_data(data)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Perform cross-validation
    cv_scores = {
        'mse': cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'),
        'mae': cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error'),
        'r2': cross_val_score(model, X, y, cv=5, scoring='r2')
    }

    # Convert MSE and MAE to positive values
    cv_scores['mse'] = -cv_scores['mse']
    cv_scores['mae'] = -cv_scores['mae']

    # Calculate mean and std for each metric
    for metric in cv_scores:
        cv_scores[metric] = {
            'mean': cv_scores[metric].mean(),
            'std': cv_scores[metric].std()
        }

    # Save cross-validation scores
    with open('cv_scores.json', 'w') as f:
        json.dump(cv_scores, f)

    # Generate feature importance plot
    feature_importance = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(10, 6))
    plt.bar(features, feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_str = base64.b64encode(img.getvalue()).decode()

    with open('feature_importance.txt', 'w') as f:
        f.write(img_str)

    return model

def make_predictions(model):
    data = fetch_all_data(start_date=datetime.now() - timedelta(days=30))
    X, _ = prepare_data(data)
    X_latest = X.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(X_latest)[0]

    timestamp = datetime.now().isoformat()
    prediction_data = {
        'timestamp': timestamp,
        'prediction': prediction.tolist(),
        'actual': None  # This will be updated later when actual data is available
    }

    # Load existing predictions
    try:
        with open('predictions.json', 'r') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        predictions = []

    # Add new prediction
    predictions.append(prediction_data)

    # Save updated predictions
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

    logging.info(f"Made prediction at {timestamp}: {prediction.tolist()}")

def update_actual_values():
    # Load existing predictions
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)

    # Fetch the latest data
    latest_data = fetch_all_data(start_date=datetime.now() - timedelta(days=30))
    _, y = prepare_data(latest_data)

    # Update actual values for previous predictions
    for i, pred in enumerate(predictions):
        if pred['actual'] is None:
            timestamp = datetime.fromisoformat(pred['timestamp'])
            actual_data = y.loc[y.index.date == timestamp.date()]
            if not actual_data.empty:
                predictions[i]['actual'] = actual_data.iloc[0].tolist()

    # Save updated predictions
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

def incorporate_paper_trading_results(model):
    try:
        with open('paper_trading_state.json', 'r') as f:
            paper_trading_state = json.load(f)
        
        portfolio_values = paper_trading_state['portfolio_values']
        
        # Calculate daily returns from portfolio values
        portfolio_returns = [
            (portfolio_values[i]['value'] - portfolio_values[i-1]['value']) / portfolio_values[i-1]['value']
            for i in range(1, len(portfolio_values))
        ]
        
        # Fetch the corresponding feature data
        data = fetch_all_data(start_date=datetime.fromisoformat(portfolio_values[0]['timestamp']))
        X, _ = prepare_data(data)
        
        # Align the features with the portfolio returns
        aligned_X = X.iloc[-len(portfolio_returns):]
        aligned_y = pd.DataFrame(portfolio_returns, columns=['portfolio_returns'], index=aligned_X.index)
        
        # Retrain the model with the new data
        model.fit(aligned_X, aligned_y)
        
        logging.info("Incorporated paper trading results into the model")
    except FileNotFoundError:
        logging.warning("No paper trading results found. Skipping incorporation.")
    
    return model

def run_predictor():
    model = train_model()
    model = incorporate_paper_trading_results(model)
    make_predictions(model)
    update_actual_values()

if __name__ == "__main__":
    run_predictor()