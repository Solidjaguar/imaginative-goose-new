import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

class DataFetcher:
    def __init__(self):
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.oanda_api_key = os.getenv('OANDA_API_KEY')
        self.oanda_account_id = os.getenv('OANDA_ACCOUNT_ID')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_forex_data(self, symbol, interval):
        async with aiohttp.ClientSession() as session:
            try:
                url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={symbol[:3]}&to_symbol={symbol[3:]}&interval={interval}&apikey={self.alpha_vantage_api_key}"
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data['Time Series FX (60min)']
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching forex data: {str(e)}")
                raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_economic_indicators(self):
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