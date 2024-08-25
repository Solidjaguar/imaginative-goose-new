#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run the forex predictor script
python3 gold_forex_predictor.py

# Start the web server
python3 web_interface.py &

# Deactivate virtual environment
deactivate