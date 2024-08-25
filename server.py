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
            
            with open('templates/index.html', 'r') as f:
                html_template = f.read()
            
            predictions_html = ''.join(f"<tr><td>{date}</td><td>${price:.2f}</td></tr>" for date, price in predictions.items())
            trades_html = ''.join(f"<tr><td>{trade['date']}</td><td>{trade['type']}</td><td>${trade['price']:.2f}</td><td>{trade['amount']:.4f} oz</td></tr>" for trade in recent_trades)
            
            html_content = html_template.format(
                latest_price=latest_price,
                predictions=predictions_html,
                portfolio_value=portfolio_value,
                balance=balance,
                gold_holdings=gold_holdings,
                recent_trades=trades_html
            )
            
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