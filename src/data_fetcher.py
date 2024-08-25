import yfinance as yf
import pandas as pd

def fetch_gold_data(interval='1d', period='1mo'):
    gold = yf.Ticker("GC=F")
    data = gold.history(interval=interval, period=period)
    return data[['Close']].reset_index()