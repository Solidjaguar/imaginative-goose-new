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
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .price, .value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
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