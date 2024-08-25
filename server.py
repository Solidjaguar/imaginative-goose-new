from http.server import HTTPServer, SimpleHTTPRequestHandler
import subprocess
import json
import threading
import time

class DataManager:
    def __init__(self):
        self.data = None
        self.last_update = 0

    def update_data(self):
        result = subprocess.run(['python3', 'main.py'], capture_output=True, text=True)
        self.data = result.stdout
        self.last_update = time.time()

    def get_data(self):
        if self.data is None or time.time() - self.last_update > 3600:  # Update every hour
            self.update_data()
        return self.data

data_manager = DataManager()

class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/prediction':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(data_manager.get_data().encode())
        else:
            return SimpleHTTPRequestHandler.do_GET(self)

def update_data_periodically():
    while True:
        data_manager.update_data()
        time.sleep(3600)  # Sleep for 1 hour

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on port {port}")
    
    # Start the periodic update thread
    update_thread = threading.Thread(target=update_data_periodically)
    update_thread.daemon = True
    update_thread.start()
    
    httpd.serve_forever()

if __name__ == '__main__':
    run()