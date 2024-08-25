#!/bin/bash

# Update package list and upgrade existing packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3 and pip if they're not already installed
sudo apt-get install -y python3 python3-pip

# Install system dependencies
sudo apt-get install -y libssl-dev libffi-dev build-essential

# Install TA-Lib dependencies
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib*

# Upgrade pip
pip3 install --upgrade pip

# Install Python packages from requirements.txt
pip3 install -r requirements.txt

# Install TA-Lib Python wrapper
pip3 install TA-Lib

# Create cache directory
mkdir -p cache

echo "Installation complete!"