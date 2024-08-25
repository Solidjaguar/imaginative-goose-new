#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update package list and upgrade existing packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip if not already installed
sudo apt install -y python3 python3-pip python3-venv

# Install system dependencies
sudo apt install -y libsqlite3-dev

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install CUDA and cuDNN (this part is tricky and might need manual intervention)
# Note: These commands are placeholders and might not work directly
# It's recommended to follow NVIDIA's official guide for installing CUDA and cuDNN
echo "Please install CUDA 11.2 and cuDNN v8.1.0 manually from NVIDIA's website."
echo "After installation, run the following command:"
echo "pip install tensorflow==2.9.0"

# Create necessary directories
mkdir -p data models logs

echo "Installation completed. Please install CUDA and cuDNN manually, then install TensorFlow with GPU support."
echo "After that, you can start the application by running: python src/main.py"