import asyncio
from fastapi import FastAPI
from hypercorn.config import Config
from hypercorn.asyncio import serve
from src.utils.data_fetcher import DataFetcher
from src.utils.feature_engineering import FeatureEngineer
from src.utils.hyperparameter_tuning import HyperparameterTuner
from src.models.model_trainer import ModelTrainer
from src.utils.model_versioner import ModelVersioner
from src.api.main import app as api_app
from loguru import logger
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

class MainApplication:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.hyperparameter_tuner = HyperparameterTuner()
        self.model_trainer = ModelTrainer()
        self.model_versioner = ModelVersioner()

    async def fetch_and_process_data(self):
        try:
            forex_data = await self.data_fetcher.fetch_forex_data("EUR/USD", "1h")
            economic_data = await self.data_fetcher.fetch_economic_indicators()
            
            combined_data = self.combine_data(forex_data, economic_data)
            processed_data = self.feature_engineer.engineer_features(combined_data)
            
            return processed_data
        except Exception as e:
            logger.error(f"Error in data fetching and processing: {str(e)}")
            raise

    def combine_data(self, forex_data, economic_data):
        # Implement proper data combination logic
        combined = pd.merge(forex_data, economic_data, on='timestamp', how='left')
        return combined.ffill()  # Forward fill missing values

    async def train_and_version_models(self, data):
        try:
            X, y = self.prepare_data_for_training(data)
            
            rf_params = await self.hyperparameter_tuner.tune_random_forest(X, y)
            lstm_params = await self.hyperparameter_tuner.tune_lstm(X, y)
            
            rf_model = await self.model_trainer.train_random_forest(X, y, **rf_params)
            lstm_model = await self.model_trainer.train_lstm(X, y, **lstm_params)
            
            self.model_versioner.save_model(rf_model, "random_forest")
            self.model_versioner.save_model(lstm_model, "lstm")
            
            logger.info("Models trained and versioned successfully")
        except Exception as e:
            logger.error(f"Error in model training and versioning: {str(e)}")
            raise

    def prepare_data_for_training(self, data):
        # Implement proper data preparation logic
        target_column = 'close'  # or whatever your target is
        features = data.drop(columns=[target_column])
        target = data[target_column]
        return features, target

    async def run(self):
        while True:
            try:
                data = await self.fetch_and_process_data()
                await self.train_and_version_models(data)
                await asyncio.sleep(3600)  # Sleep for 1 hour
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(300)  # Sleep for 5 minutes before retrying

async def start_api():
    config = Config()
    config.bind = ["0.0.0.0:8080"]
    await serve(api_app, config)

async def main():
    main_app = MainApplication()
    await asyncio.gather(
        main_app.run(),
        start_api()
    )

if __name__ == "__main__":
    asyncio.run(main())