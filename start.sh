#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Start the model retrainer
python3 model_retrainer.py &

# Start the background predictor
python3 background_predictor.py &

# Start the risk management system
python3 risk_management.py &

# Start the web server
python3 server.py