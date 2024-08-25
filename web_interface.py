from flask import Flask, render_template, jsonify
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from paper_trader import PaperTrader

app = Flask(__name__)

def load_predictions():
    with open('predictions.json', 'r') as f:
        return json.load(f)

def create_plot():
    predictions = load_predictions()
    df = pd.DataFrame(predictions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    plt.figure(figsize=(12, 6))
    for i, currency in enumerate(['EUR/USD', 'GBP/USD', 'JPY/USD']):
        plt.plot(df.index, [p[i] for p in df['prediction']], label=f'{currency} Predicted')
        if df['actual'].notna().any():
            plt.plot(df.index, [a[i] if a else None for a in df['actual']], label=f'{currency} Actual')
    
    plt.title('Forex Predictions vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()

def calculate_metrics(predictions):
    actual_values = [p['actual'] for p in predictions if p['actual'] is not None]
    predicted_values = [p['prediction'] for p in predictions if p['actual'] is not None]
    
    if len(actual_values) > 0:
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    else:
        return None

def load_cv_scores():
    try:
        with open('cv_scores.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_feature_importance():
    try:
        with open('feature_importance.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None

def load_backtest_results():
    try:
        with open('backtest_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_backtest_plots():
    try:
        with open('backtest_plots.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_trading_strategy_results():
    try:
        with open('trading_strategy_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_paper_trading_results():
    try:
        with open('paper_trading_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    predictions = load_predictions()
    plot_url = create_plot()
    metrics = calculate_metrics(predictions)
    cv_scores = load_cv_scores()
    feature_importance = load_feature_importance()
    backtest_results = load_backtest_results()
    backtest_plots = load_backtest_plots()
    trading_strategy_results = load_trading_strategy_results()
    paper_trading_results = load_paper_trading_results()
    return render_template('index.html', predictions=predictions, plot_url=plot_url, metrics=metrics, 
                           cv_scores=cv_scores, feature_importance=feature_importance, 
                           backtest_results=backtest_results, backtest_plots=backtest_plots,
                           trading_strategy_results=trading_strategy_results,
                           paper_trading_results=paper_trading_results)

@app.route('/run_paper_trading')
def run_paper_trading():
    paper_trader = PaperTrader()
    portfolio_values = paper_trader.run_paper_trading()
    
    with open('paper_trading_results.json', 'w') as f:
        json.dump({
            'portfolio_values': portfolio_values,
            'trade_history': paper_trader.trade_history
        }, f)
    
    return jsonify({'status': 'success', 'message': 'Paper trading completed'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)