import unittest
from src.data_fetcher import fetch_all_data
import pandas as pd

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.config = {
            'api_keys': {
                'alpha_vantage': 'YOUR_ALPHA_VANTAGE_KEY',
                'currents': 'YOUR_CURRENTS_KEY'
            },
            'data': {
                'lookback_years': 5
            }
        }

    def test_fetch_all_data(self):
        data = fetch_all_data(self.config)
        
        self.assertIsInstance(data, dict)
        self.assertIn('Gold', data)
        self.assertIn('Forex', data)
        self.assertIn('Economic', data)
        self.assertIn('Crypto', data)
        self.assertIn('Sentiment', data)
        
        self.assertIsInstance(data['Gold'], pd.Series)
        self.assertIsInstance(data['Forex'], pd.DataFrame)
        self.assertIsInstance(data['Economic'], pd.DataFrame)
        self.assertIsInstance(data['Crypto'], pd.DataFrame)
        self.assertIsInstance(data['Sentiment'], pd.DataFrame)

if __name__ == '__main__':
    unittest.main()