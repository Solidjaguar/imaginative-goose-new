import asyncio
import pandas as pd
import numpy as np
import sys
import os
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger to output to console
logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_fetcher import DataFetcher

async def fetch_forex_data(symbol):
    data_fetcher = DataFetcher()
    df = await data_fetcher.fetch_forex_data(symbol)
    
    if df is None or df.empty:
        logger.error(f"No data received for {symbol}")
        return None
    
    return df

async def main():
    # Fetch latest exchange rate for GBP/USD
    gbpusd_data = await fetch_forex_data('GBPUSD')
    
    if gbpusd_data is not None:
        logger.info(f"Latest GBP/USD exchange rate: {gbpusd_data['exchange_rate'].iloc[0]}")
    
    # Mock Gold price (as of August 2024)
    mock_gold_price = 1960.50
    logger.info(f"Latest Gold (XAU/USD) price (mock data): ${mock_gold_price}")

if __name__ == "__main__":
    asyncio.run(main())