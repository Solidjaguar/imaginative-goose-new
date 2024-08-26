import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class DataFetcher:
    def __init__(self):
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.alpha_vantage_api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_forex_data(self, symbol):
        async with aiohttp.ClientSession() as session:
            try:
                url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={symbol[:3]}&to_currency={symbol[3:]}&apikey={self.alpha_vantage_api_key}"
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    logger.info(f"Full API Response: {data}")
                    
                    if "Realtime Currency Exchange Rate" not in data:
                        logger.error(f"Expected key 'Realtime Currency Exchange Rate' not found in API response")
                        logger.error(f"Available keys: {list(data.keys())}")
                        return None
                    
                    exchange_rate = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
                    timestamp = data["Realtime Currency Exchange Rate"]["6. Last Refreshed"]
                    
                    df = pd.DataFrame({
                        'timestamp': [timestamp],
                        'exchange_rate': [exchange_rate]
                    })
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                    return df
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching forex data: {str(e)}")
                raise

# Usage:
# data_fetcher = DataFetcher()
# df = await data_fetcher.fetch_forex_data('GBPUSD')