from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from main import main

class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/prediction':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            prediction = main()
            self.wfile.write(prediction.encode())
        else:
            super().do_GET()

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()