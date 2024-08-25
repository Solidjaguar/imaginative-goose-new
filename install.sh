#!/bin/bash

# Update package list and install system dependencies
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required Python packages
pip install yfinance pandas numpy scipy statsmodels

# Create Python scripts
cat > ultra_advanced_gold_predictor.py << EOL
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
from datetime import datetime, timedelta

def fetch_gold_data(interval='15m', period='1d'):
    gold = yf.Ticker("GC=F")
    data = gold.history(interval=interval, period=period)
    return data[['Close']].reset_index()

def prepare_data(data):
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
    return data['Close']

def train_model(data):
    model = ARIMA(data, order=(1,1,1))
    return model.fit()

def predict_price(model, data, steps=4, confidence_threshold=0.7):
    last_datetime = data.index[-1]
    future_datetimes = pd.date_range(start=last_datetime + pd.Timedelta(minutes=15), periods=steps, freq='15T')
    forecast = model.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int(alpha=0.32)  # ~1 standard deviation

    predictions = []
    for datetime, mean, lower, upper in zip(future_datetimes, mean_forecast, confidence_intervals['lower Close'], confidence_intervals['upper Close']):
        std_dev = (upper - lower) / 2
        z_score = abs(mean - data.iloc[-1]) / std_dev
        confidence = 1 - 2 * (1 - norm.cdf(z_score))
        
        if confidence >= confidence_threshold:
            predictions.append({
                'Datetime': datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'Predicted_Price': mean,
                'Confidence': confidence
            })

    return predictions

if __name__ == "__main__":
    data = fetch_gold_data()
    prepared_data = prepare_data(data)
    model = train_model(prepared_data)
    predictions = predict_price(model, prepared_data)
    print(predictions)
EOL

cat > paper_trader.py << EOL
import json
from datetime import datetime, timedelta

class PaperTrader:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.gold_holdings = 0
        self.trades = []
        self.load_state()

    def load_state(self):
        try:
            with open('paper_trader_state.json', 'r') as f:
                state = json.load(f)
                self.balance = state['balance']
                self.gold_holdings = state['gold_holdings']
                self.trades = state['trades']
        except FileNotFoundError:
            self.save_state()

    def save_state(self):
        state = {
            'balance': self.balance,
            'gold_holdings': self.gold_holdings,
            'trades': self.trades
        }
        with open('paper_trader_state.json', 'w') as f:
            json.dump(state, f)

    def buy(self, price, amount):
        cost = price * amount
        if cost <= self.balance:
            self.balance -= cost
            self.gold_holdings += amount
            self.trades.append({
                'type': 'buy',
                'price': price,
                'amount': amount,
                'date': datetime.now().isoformat()
            })
            self.save_state()
            return True
        return False

    def sell(self, price, amount):
        if amount <= self.gold_holdings:
            self.balance += price * amount
            self.gold_holdings -= amount
            self.trades.append({
                'type': 'sell',
                'price': price,
                'amount': amount,
                'date': datetime.now().isoformat()
            })
            self.save_state()
            return True
        return False

    def get_portfolio_value(self, current_price):
        return self.balance + (self.gold_holdings * current_price)

    def get_recent_trades(self, hours=24):
        cutoff_date = datetime.now() - timedelta(hours=hours)
        return [trade for trade in self.trades if datetime.fromisoformat(trade['date']) > cutoff_date]

    def get_state(self):
        return {
            'balance': self.balance,
            'gold_holdings': self.gold_holdings,
            'trades': self.trades
        }

paper_trader = PaperTrader()
EOL

cat > background_predictor.py << EOL
import time
import json
from ultra_advanced_gold_predictor import fetch_gold_data, prepare_data, train_model, predict_price
from paper_trader import paper_trader
from datetime import datetime, timedelta

def calculate_accuracy(predictions, actual_prices):
    correct = sum(1 for pred, actual in zip(predictions, actual_prices) if (pred > 0 and actual > 0) or (pred < 0 and actual < 0))
    return correct / len(predictions) if predictions else 0

def update_predictions_and_trade():
    learning_progress = {
        'prediction_accuracy': [],
        'portfolio_value': [],
        'model_confidence': []
    }

    while True:
        try:
            gold_data = fetch_gold_data(interval='15m', period='1d')
            
            if gold_data.empty:
                print("Failed to fetch gold data")
                time.sleep(60)
                continue
            
            prepared_data = prepare_data(gold_data)
            model = train_model(prepared_data)
            prediction_data = predict_price(model, prepared_data, steps=4)
            
            latest_price = float(prepared_data.iloc[-1])
            
            # Trading strategy
            for prediction in prediction_data:
                predicted_price = prediction['Predicted_Price']
                confidence = prediction['Confidence']
                
                if confidence > 0.8:  # Only trade if confidence is high
                    if predicted_price > latest_price:
                        amount_to_buy = min(0.1, paper_trader.balance / latest_price)
                        if amount_to_buy > 0:
                            paper_trader.buy(latest_price, amount_to_buy)
                    elif predicted_price < latest_price:
                        amount_to_sell = min(0.1, paper_trader.gold_holdings)
                        if amount_to_sell > 0:
                            paper_trader.sell(latest_price, amount_to_sell)
            
            # Update learning progress
            if len(learning_progress['prediction_accuracy']) >= 96:  # Keep last 24 hours of data (96 15-minute intervals)
                learning_progress['prediction_accuracy'].pop(0)
                learning_progress['portfolio_value'].pop(0)
                learning_progress['model_confidence'].pop(0)

            # Calculate prediction accuracy
            previous_predictions = [pred['Predicted_Price'] for pred in json.loads(open('latest_predictions.json').read())['prediction_data']]
            actual_prices = prepared_data[-4:].tolist()
            accuracy = calculate_accuracy([p - latest_price for p in previous_predictions], [a - latest_price for a in actual_prices])
            
            learning_progress['prediction_accuracy'].append(accuracy)
            learning_progress['portfolio_value'].append(paper_trader.get_portfolio_value(latest_price))
            learning_progress['model_confidence'].append(sum(pred['Confidence'] for pred in prediction_data) / len(prediction_data))

            # Save the latest predictions and learning progress
            with open('latest_predictions.json', 'w') as f:
                json.dump({
                    'latest_price': latest_price,
                    'latest_date': prepared_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction_data': prediction_data,
                    'learning_progress': learning_progress
                }, f)
            
            print(f"Updated predictions at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Wait for 15 minutes before the next update
            time.sleep(900)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    update_predictions_and_trade()
EOL

cat > main.py << EOL
import json
from paper_trader import paper_trader

def main():
    try:
        with open('latest_predictions.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return json.dumps({"error": "Predictions not available yet"})

    latest_price = data['latest_price']
    latest_date = data['latest_date']
    prediction_data = data['prediction_data']
    learning_progress = data.get('learning_progress', {})
    
    result = {
        'latest_price': latest_price,
        'latest_date': latest_date,
        'prediction_data': prediction_data,
        'portfolio_value': paper_trader.get_portfolio_value(latest_price),
        'recent_trades': paper_trader.get_recent_trades(hours=24),
        'balance': paper_trader.balance,
        'gold_holdings': paper_trader.gold_holdings,
        'learning_progress': learning_progress
    }
    
    return json.dumps(result, default=str)

if __name__ == "__main__":
    print(main())
EOL

cat > server.py << EOL
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from main import main
import os
import time

class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            data = json.loads(main())
            latest_price = data.get('latest_price', 'N/A')
            latest_date = data.get('latest_date', 'N/A')
            predictions = data.get('prediction_data', [])
            portfolio_value = data.get('portfolio_value', 'N/A')
            recent_trades = data.get('recent_trades', [])
            balance = data.get('balance', 'N/A')
            gold_holdings = data.get('gold_holdings', 'N/A')
            learning_progress = data.get('learning_progress', {})
            
            # Check if background process is running
            try:
                with open('latest_predictions.json', 'r') as f:
                    last_update = os.path.getmtime('latest_predictions.json')
                    last_update_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_update))
                    bg_status = f"Running (Last update: {last_update_str})"
            except FileNotFoundError:
                bg_status = "Not running or hasn't made predictions yet"
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Gold Price Predictor and Paper Trader</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .price, .value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    .chart-container {{ width: 100%; height: 300px; margin-top: 20px; }}
                </style>
                <script>
                    function refreshPage() {{
                        location.reload();
                    }}
                    setInterval(refreshPage, 60000); // Refresh every 60 seconds
                </script>
            </head>
            <body>
                <h1>Ultra Advanced Gold Price Predictor and Paper Trader</h1>
                <p>Background AI Status: {bg_status}</p>
                <p>Latest gold price: <span class="price">${latest_price}</span> (as of {latest_date})</p>
                <h2>Paper Trading Portfolio:</h2>
                <p>Portfolio Value: <span class="value">${portfolio_value}</span></p>
                <p>Cash Balance: <span class="value">${balance}</span></p>
                <p>Gold Holdings: <span class="value">{gold_holdings} oz</span></p>
                <h2>Recent Trades (Last 24 Hours):</h2>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Price</th>
                        <th>Amount</th>
                    </tr>
                    {''.join(f"<tr><td>{trade['date']}</td><td>{trade['type']}</td><td>${trade['price']:.2f}</td><td>{trade['amount']:.4f} oz</td></tr>" for trade in recent_trades)}
                </table>
                <h2>Short-term Price Predictions (Next Hour):</h2>
                <table>
                    <tr>
                        <th>Datetime</th>
                        <th>Predicted Price</th>
                        <th>Confidence</th>
                    </tr>
                    {''.join(f"<tr><td>{pred['Datetime']}</td><td>${pred['Predicted_Price']:.2f}</td><td>{pred['Confidence']:.2%}</td></tr>" for pred in predictions)}
                </table>
                <p>Note: Only predictions with confidence level of 70% or higher are shown.</p>
                
                <h2>AI Learning Progress:</h2>
                <div class="chart-container">
                    <canvas id="learningProgressChart"></canvas>
                </div>
                
                <script>
                    var ctx = document.getElementById('learningProgressChart').getContext('2d');
                    var chart = new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: Array.from({{length: {len(learning_progress.get('prediction_accuracy', []))}}}, (_, i) => i + 1),
                            datasets: [
                                {{
                                    label: 'Prediction Accuracy',
                                    data: {json.dumps(learning_progress.get('prediction_accuracy', []))},
                                    borderColor: 'rgb(75, 192, 192)',
                                    tension: 0.1
                                }},
                                {{
                                    label: 'Portfolio Value',
                                    data: {json.dumps(learning_progress.get('portfolio_value', []))},
                                    borderColor: 'rgb(255, 99, 132)',
                                    tension: 0.1
                                }},
                                {{
                                    label: 'Model Confidence',
                                    data: {json.dumps(learning_progress.get('model_confidence', []))},
                                    borderColor: 'rgb(54, 162, 235)',
                                    tension: 0.1
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{
                                    beginAtZero: true
                                }}
                            }}
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            self.wfile.write(html_content.encode())
        else:
            super().do_GET()

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
EOL

# Create a startup script
cat > start.sh << EOL
#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Start the background predictor
python3 background_predictor.py &

# Start the web server
python3 server.py
EOL

# Make the startup script executable
chmod +x start.sh

echo "Installation complete. Run './start.sh' to start the application."