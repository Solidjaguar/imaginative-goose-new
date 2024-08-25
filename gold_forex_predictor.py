import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import io
import base64
import logging
from ta.trend import MACD, SMAIndicator, EMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange

nltk.download('vader_lexicon', quiet=True)

logging.basicConfig(filename='gold_forex_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# API keys
ALPHA_VANTAGE_API_KEY = "PIFHGHQNBWL37L0T"
CURRENTS_API_KEY = "IEfpA5hCrH6Xh4E9f7R0jEOHcEjxSI8k6s71NwcYXRPqtohR"

# The rest of the code remains the same as in the previous version

def fetch_all_data(start_date=None, end_date=None):
    # ... (unchanged)

def fetch_economic_indicators(start_date, end_date):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    fx = ForeignExchange(key=ALPHA_VANTAGE_API_KEY)

    # Fetch GDP growth rate (quarterly)
    gdp_data, _ = ts.get_global_quote('GDP')
    
    # Fetch inflation rate (monthly)
    cpi_data, _ = ts.get_global_quote('CPI')
    
    # Fetch interest rates (daily)
    interest_rate_data, _ = fx.get_currency_exchange_daily('USD', 'EUR')  # Using EUR/USD as a proxy for interest rate differentials

    # Combine and resample data to daily frequency
    indicators = pd.concat([gdp_data, cpi_data, interest_rate_data], axis=1)
    indicators = indicators.resample('D').ffill()
    
    # Trim to the specified date range
    indicators = indicators.loc[start_date:end_date]

    return indicators

def fetch_news_sentiment(start_date, end_date):
    # ... (unchanged)

def add_advanced_features(df):
    # ... (unchanged)

def prepare_data(data, economic_indicators, news_sentiment):
    # ... (unchanged)

def train_model():
    # ... (unchanged)

def make_predictions(model):
    # ... (unchanged)

def update_actual_values():
    # ... (unchanged)

def run_predictor():
    model = train_model()
    make_predictions(model)
    update_actual_values()

if __name__ == "__main__":
    run_predictor()