#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update package list and upgrade existing packages
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    build-essential

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Setup MLflow
mkdir -p mlruns

# Create necessary directories
mkdir -p data models logs

# Set up environment variables (replace with actual values)
echo "ALPHA_VANTAGE_API_KEY=your_api_key_here" > .env
echo "OANDA_API_KEY=your_api_key_here" >> .env
echo "OANDA_ACCOUNT_ID=your_account_id_here" >> .env

# Clone the project repository (uncomment and modify if needed)
# git clone https://github.com/yourusername/forex-gold-prediction.git
# cd forex-gold-prediction

# Run tests
python -m pytest tests/

# Print success message
echo "Installation completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To start the application, run: python src/main.py"