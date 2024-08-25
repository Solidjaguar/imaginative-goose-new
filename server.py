from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from main import main

class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            data = json.loads(main())
            latest_price = data.get('latest_price', 'N/A')
            predictions = data.get('predictions', {})
            portfolio_value = data.get('portfolio_value', 'N/A')
            recent_trades = data.get('recent_trades', [])
            balance = data.get('balance', 'N/A')
            gold_holdings = data.get('gold_holdings', 'N/A')
            
            html_content = f"""
            <html>
            <head>
                <title>Gold Price Predictor and Paper Trader</title>
                <meta http-equiv="refresh" content="60">
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .price {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                </style>
            </head>
            <body>
                <h1>Gold Price Predictor and Paper Trader</h1>
                <p>Current gold price: <span class="price">${latest_price}</span></p>
                <h2>Predictions:</h2>
                <table>
                    <tr><th>Date</th><th>Predicted Price</th></tr>
                    {''.join(f"<tr><td>{date}</td><td>${price:.2f}</td></tr>" for date, price in predictions.items())}
                </table>
                <h2>Paper Trading Portfolio:</h2>
                <p>Portfolio Value: ${portfolio_value:.2f}</p>
                <p>Cash Balance: ${balance:.2f}</p>
                <p>Gold Holdings: {gold_holdings:.4f} oz</p>
                <h2>Recent Trades:</h2>
                <table>
                    <tr><th>Date</th><th>Type</th><th>Price</th><th>Amount</th></tr>
                    {''.join(f"<tr><td>{trade['date']}</td><td>{trade['type']}</td><td>${trade['price']:.2f}</td><td>{trade['amount']:.4f} oz</td></tr>" for trade in recent_trades)}
                </table>
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