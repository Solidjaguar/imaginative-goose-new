#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the main prediction script
python3 gold_forex_predictor.py

# Run the backtesting script
python3 backtester.py

# Run the trading strategy script
python3 trading_strategy.py

# Run the paper trading script
python3 paper_trader.py

# Start the web server
python3 web_interface.py &

echo "Daily run completed. Web server started in the background."