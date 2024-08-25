from flask import Flask, render_template
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import mean_squared_error
import numpy as np

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
        mae = np.mean(np.abs(np.array(actual_values) - np.array(predicted_values)))
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }
    else:
        return None

@app.route('/')
def index():
    predictions = load_predictions()
    plot_url = create_plot()
    metrics = calculate_metrics(predictions)
    return render_template('index.html', predictions=predictions, plot_url=plot_url, metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)