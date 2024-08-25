#!/bin/bash

# Set the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Activate virtual environment if you're using one
# source /path/to/your/venv/bin/activate

# Set environment variables (replace with your actual API keys)
export NEWS_API_KEY='your_news_api_key'
export FRED_API_KEY='your_fred_api_key'

# Run the main prediction script
python3 ultra_advanced_gold_predictor.py

# Run the backtesting script
python3 backtest.py

# Log the completion of the daily run
echo "Daily run completed at $(date)" >> daily_run_log.txt

# Optionally, you can add commands to save or send the results
# For example, you could email the results or push them to a repository

# Deactivate virtual environment if you're using one
# deactivate