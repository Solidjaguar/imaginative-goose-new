from flask import Flask, render_template
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

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
    for currency in ['EUR/USD', 'GBP/USD', 'JPY/USD']:
        plt.plot(df.index, [p[currency] for p in df['prediction']], label=f'{currency} Predicted')
        if df['actual'].notna().any():
            plt.plot(df.index, [a[currency] if a else None for a in df['actual']], label=f'{currency} Actual')
    
    plt.title('Forex Predictions vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    predictions = load_predictions()
    plot_url = create_plot()
    return render_template('index.html', predictions=predictions, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)