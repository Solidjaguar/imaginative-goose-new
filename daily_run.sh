#!/bin/bash

# Install required packages
pip install yfinance pandas numpy scikit-learn matplotlib flask tensorflow nltk ta-lib

# Run the prediction script hourly
while true; do
    python3 gold_forex_predictor.py
    python3 backtesting.py
    sleep 3600
done &

# Run the paper trading script continuously
python3 paper_trader.py &

# Run the web server
python3 web_interface.py