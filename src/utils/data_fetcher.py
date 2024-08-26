import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

class DataFetcher:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_forex_data(self, symbol, interval):
        try:
            url = f"https://api.example.com/forex?symbol={symbol}&interval={interval}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching forex data: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_economic_indicators(self):
        try:
            url = "https://api.example.com/economic_indicators"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching economic indicators: {str(e)}")
            raise

# Add more methods for other data sources as needed