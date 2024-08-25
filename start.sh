#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Start the background predictor
python3 background_predictor.py &

# Start the web server
python3 server.py
