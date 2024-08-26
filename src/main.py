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

class MainApplication:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.hyperparameter_tuner = HyperparameterTuner()
        self.model_trainer = ModelTrainer()
        self.model_versioner = ModelVersioner()

    async def fetch_and_process_data(self):
        try:
            forex_data = self.data_fetcher.fetch_forex_data("EUR/USD", "1h")
            economic_data = self.data_fetcher.fetch_economic_indicators()
            
            # Combine forex and economic data
            combined_data = self.combine_data(forex_data, economic_data)
            
            # Engineer features
            processed_data = self.feature_engineer.engineer_features(combined_data)
            
            return processed_data
        except Exception as e:
            logger.error(f"Error in data fetching and processing: {str(e)}")
            raise

    def combine_data(self, forex_data, economic_data):
        # Implement logic to combine forex and economic data
        # This is a placeholder and should be implemented based on your specific data structures
        return forex_data  # For now, just return forex_data

    async def train_and_version_models(self, data):
        try:
            X, y = self.prepare_data_for_training(data)
            
            # Tune hyperparameters
            rf_params = self.hyperparameter_tuner.tune_random_forest(X, y)
            lstm_params = self.hyperparameter_tuner.tune_lstm(X, y)
            
            # Train models with tuned hyperparameters
            rf_model = self.model_trainer.train_random_forest(X, y, **rf_params)
            lstm_model = self.model_trainer.train_lstm(X, y, **lstm_params)
            
            # Version models
            self.model_versioner.save_model(rf_model, "random_forest")
            self.model_versioner.save_model(lstm_model, "lstm")
            
            logger.info("Models trained and versioned successfully")
        except Exception as e:
            logger.error(f"Error in model training and versioning: {str(e)}")
            raise

    def prepare_data_for_training(self, data):
        # Implement logic to prepare data for training
        # This is a placeholder and should be implemented based on your specific requirements
        X = data.drop('target', axis=1)
        y = data['target']
        return X, y

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