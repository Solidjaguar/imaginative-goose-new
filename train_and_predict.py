import asyncio
import pandas as pd
import numpy as np
import sys
import os
from loguru import logger

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_fetcher import DataFetcher
from src.utils.feature_engineering import FeatureEngineer
from src.utils.hyperparameter_tuning import HyperparameterTuner
from src.models.model_trainer import ModelTrainer
from src.utils.model_versioner import ModelVersioner

async def fetch_and_prepare_data(symbol, interval='60min'):
    data_fetcher = DataFetcher()
    raw_data = await data_fetcher.fetch_forex_data(symbol, interval)
    
    if raw_data is None:
        logger.error(f"No data received for {symbol}")
        return None
    
    df = pd.DataFrame(raw_data).T
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = ['open', 'high', 'low', 'close']
    
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.engineer_features(df)
    
    return df_features

async def train_and_save_model(symbol):
    df = await fetch_and_prepare_data(symbol)
    
    if df is None or df.empty:
        logger.error(f"Unable to train model for {symbol} due to lack of data")
        return None
    
    # Prepare data for training
    X = df.drop('close', axis=1)
    y = df['close']
    
    # Tune hyperparameters
    tuner = HyperparameterTuner(n_trials=50)
    best_params = await tuner.tune_random_forest(X, y)
    
    # Train model
    trainer = ModelTrainer()
    model = await trainer.train_random_forest(X, y, **best_params)
    
    # Save model
    versioner = ModelVersioner()
    versioner.save_model(model, f"{symbol}_model")
    
    return model

async def main():
    # Train models for GBP/USD and Gold
    gbpusd_model = await train_and_save_model('GBPUSD')
    xauusd_model = await train_and_save_model('XAUUSD')  # XAU is the symbol for Gold
    
    if gbpusd_model:
        logger.info("GBP/USD model trained successfully")
    if xauusd_model:
        logger.info("Gold (XAU/USD) model trained successfully")

if __name__ == "__main__":
    asyncio.run(main())