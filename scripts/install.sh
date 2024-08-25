#!/bin/bash

# Update package list and install system dependencies
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required Python packages
pip install yfinance pandas numpy scipy statsmodels scikit-learn matplotlib

# Create necessary directories
mkdir -p static

# Make the start script executable
chmod +x start.sh

echo "Installation complete. Run './start.sh' to start the application."