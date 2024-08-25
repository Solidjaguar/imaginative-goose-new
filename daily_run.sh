#!/bin/bash

# Install required packages
pip install yfinance pandas numpy scikit-learn matplotlib flask tensorflow nltk ta-lib requests alpha_vantage

# Run the prediction script hourly
while true; do
    python3 gold_forex_predictor.py
    sleep 3600
done &

# Run the paper trading script continuously
python3 paper_trader.py &

# Run the model retrainer script
python3 model_retrainer.py &

# Run the web server
python3 web_interface.py