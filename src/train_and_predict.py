import asyncio
import pandas as pd
import numpy as np
from src.utils.data_fetcher import DataFetcher
from src.utils.feature_engineering import FeatureEngineer
from src.utils.hyperparameter_tuning import HyperparameterTuner
from src.models.model_trainer import ModelTrainer
from src.utils.model_versioner import ModelVersioner

async def fetch_and_prepare_data(symbol, interval='60min'):
    data_fetcher = DataFetcher()
    raw_data = await data_fetcher.fetch_forex_data(symbol, interval)
    
    df = pd.DataFrame(raw_data).T
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = ['open', 'high', 'low', 'close']
    
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.engineer_features(df)
    
    return df_features

async def train_and_save_model(symbol):
    df = await fetch_and_prepare_data(symbol)
    
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
    await train_and_save_model('GBPUSD')
    await train_and_save_model('XAUUSD')  # XAU is the symbol for Gold

if __name__ == "__main__":
    asyncio.run(main())