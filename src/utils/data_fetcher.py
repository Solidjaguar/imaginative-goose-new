import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from fredapi import Fred
from src.utils.logger import app_logger
from config import ALPHA_VANTAGE_API_KEY, FRED_API_KEY

class DataFetcher:
    def __init__(self):
        self.av = TimeSeries(key=ALPHA_VANTAGE_API_KEY)
        self.fred = Fred(api_key=FRED_API_KEY)

    def fetch_forex_data(self, symbol='EUR/USD', interval='daily'):
        try:
            data, _ = self.av.get_forex_daily(symbol, outputsize='full')
            df = pd.DataFrame(data).T
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close']
            df = df.astype(float)
            app_logger.info(f"Successfully fetched forex data for {symbol}")
            return df
        except Exception as e:
            app_logger.error(f"Error fetching forex data: {str(e)}")
            raise

    def fetch_economic_indicator(self, indicator):
        try:
            data = self.fred.get_series(indicator)
            app_logger.info(f"Successfully fetched economic indicator: {indicator}")
            return data
        except Exception as e:
            app_logger.error(f"Error fetching economic indicator {indicator}: {str(e)}")
            raise

# Usage example:
# fetcher = DataFetcher()
# forex_data = fetcher.fetch_forex_data()
# gdp_data = fetcher.fetch_economic_indicator('GDP')