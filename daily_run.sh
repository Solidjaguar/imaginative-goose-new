#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Start the paper trading script in the background
python3 paper_trader.py &

# Run the main prediction script
python3 gold_forex_predictor.py

# Start the web server
python3 web_interface.py &

echo "Continuous operation started. Paper trading and web server are running in the background."

# Keep the script running
while true; do
    # Run the prediction script every hour
    sleep 3600
    python3 gold_forex_predictor.py
done