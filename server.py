from http.server import HTTPServer, SimpleHTTPRequestHandler
import subprocess
import json

class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/prediction':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            result = subprocess.run(['python3', 'main.py'], capture_output=True, text=True)
            self.wfile.write(result.stdout.encode())
        else:
            return SimpleHTTPRequestHandler.do_GET(self)

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()