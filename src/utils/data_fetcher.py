import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

load_dotenv()

class DataFetcher:
    def __init__(self):
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.oanda_api_key = os.getenv('OANDA_API_KEY')
        self.oanda_account_id = os.getenv('OANDA_ACCOUNT_ID')

    async def fetch_forex_data(self, symbol, interval):
        logger.info(f"Generating mock data for {symbol}")
        return self.generate_mock_data(symbol)

    def generate_mock_data(self, symbol, num_days=30):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        if symbol == 'GBPUSD':
            base_price = 1.3
        elif symbol == 'XAUUSD':
            base_price = 1800
        else:
            base_price = 100

        data = {
            'open': np.random.normal(base_price, base_price * 0.01, len(date_range)),
            'high': np.random.normal(base_price * 1.01, base_price * 0.01, len(date_range)),
            'low': np.random.normal(base_price * 0.99, base_price * 0.01, len(date_range)),
            'close': np.random.normal(base_price, base_price * 0.01, len(date_range)),
            'volume': np.random.randint(1000, 10000, len(date_range))  # Added volume
        }
        
        df = pd.DataFrame(data, index=date_range)
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df.to_dict('index')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_economic_indicators(self):
        # This method remains unchanged
        async with aiohttp.ClientSession() as session:
            try:
                url = f"https://api-fxpractice.oanda.com/v3/accounts/{self.oanda_account_id}/instruments"
                headers = {"Authorization": f"Bearer {self.oanda_api_key}"}
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data['instruments']
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching economic indicators: {str(e)}")
                raise

# Add more methods for other data sources as needed