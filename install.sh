#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update package list and upgrade existing packages
echo "Updating and upgrading packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and pip if not already installed
echo "Installing Python and pip..."
sudo apt-get install -y python3 python3-pip

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y libssl-dev sqlite3 libsqlite3-dev

# Create a virtual environment
echo "Creating a virtual environment..."
python3 -m venv gold_predictor_env

# Activate the virtual environment
echo "Activating the virtual environment..."
source gold_predictor_env/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install pandas numpy yfinance scikit-learn statsmodels prophet tensorflow matplotlib seaborn pmdarima xgboost lightgbm optuna requests textblob TA-Lib river scipy scikit-optimize shap pyod

# Install additional dependencies that might require special handling
echo "Installing additional dependencies..."
pip install --no-binary :all: psutil

# Create necessary directories
echo "Creating project directories..."
mkdir -p data models results

# Download the latest version of the script
echo "Downloading the latest version of the gold predictor script..."
curl -o ultra_advanced_gold_predictor.py https://raw.githubusercontent.com/yourusername/gold-predictor/main/ultra_advanced_gold_predictor.py

echo "Installation complete!"
echo "To activate the virtual environment, run: source gold_predictor_env/bin/activate"
echo "To run the script, use: python ultra_advanced_gold_predictor.py"