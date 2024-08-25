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
            latest_prices = data.get('latest_prices', {})
            predictions = data.get('predictions', {})
            portfolio_values = data.get('portfolio_values', {})
            recent_trades = data.get('recent_trades', {})
            balances = data.get('balances', {})
            asset_holdings = data.get('asset_holdings', {})
            performance_metrics = data.get('performance_metrics', {})
            
            with open('templates/index.html', 'r') as f:
                html_template = f.read()
            
            markets_html = ''
            for market in latest_prices.keys():
                market_predictions_html = ''.join(f"<tr><td>{date}</td><td>${price:.2f}</td></tr>" for date, price in predictions[market].items())
                market_trades_html = ''.join(f"<tr><td>{trade['date']}</td><td>{trade['type']}</td><td>${trade['price']:.2f}</td><td>{trade['amount']:.4f}</td></tr>" for trade in recent_trades[market])
                market_metrics_html = ''.join(f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>" for metric, value in performance_metrics[market].items())
                
                market_html = f"""
                <h2>{market}</h2>
                <p>Latest Price: ${latest_prices[market]:.2f}</p>
                <p>Portfolio Value: ${portfolio_values[market]:.2f}</p>
                <p>Cash Balance: ${balances[market]:.2f}</p>
                <p>Asset Holdings: {asset_holdings[market]:.4f}</p>
                
                <h3>Predictions</h3>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Predicted Price</th>
                    </tr>
                    {market_predictions_html}
                </table>
                
                <h3>Recent Trades</h3>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Price</th>
                        <th>Amount</th>
                    </tr>
                    {market_trades_html}
                </table>
                
                <h3>Performance Metrics</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {market_metrics_html}
                </table>
                
                <h3>Price Predictions Chart</h3>
                <img src="/{market.replace('/', '_')}_predictions.png" alt="{market} Price Predictions">
                
                <h3>Trading Performance Chart</h3>
                <img src="/{market.replace('/', '_')}_trading_performance.png" alt="{market} Trading Performance">
                """
                markets_html += market_html
            
            html_content = html_template.format(markets=markets_html)
            
            self.wfile.write(html_content.encode())
        elif self.path.endswith('.png'):
            self.path = f'static{self.path}'
            return super().do_GET()
        else:
            super().do_GET()

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()