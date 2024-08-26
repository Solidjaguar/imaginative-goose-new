from src.utils.logger import app_logger
from src.utils.data_fetcher import DataFetcher
from src.models.model_trainer import ModelTrainer
from src.utils.model_versioner import ModelVersioner
from src.web.server import start_server
from src.api.main import app as fastapi_app
import uvicorn
import threading

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8080)

def main():
    app_logger.info("Starting the Forex and Gold Prediction System")

    # Initialize components
    data_fetcher = DataFetcher()
    model_trainer = ModelTrainer()
    model_versioner = ModelVersioner('models')

    try:
        # Fetch data
        app_logger.info("Fetching forex data")
        forex_data = data_fetcher.fetch_forex_data()
        
        app_logger.info("Fetching economic indicators")
        gdp_data = data_fetcher.fetch_economic_indicator('GDP')
        inflation_data = data_fetcher.fetch_economic_indicator('CPIAUCSL')

        # Train models
        app_logger.info("Training models")
        random_forest_model = model_trainer.train_random_forest(forex_data, gdp_data, inflation_data)
        lstm_model = model_trainer.train_lstm(forex_data, gdp_data, inflation_data)

        # Save models with versioning
        app_logger.info("Saving models")
        rf_info = model_versioner.save_model(random_forest_model, 'RandomForest', {'accuracy': 0.85})
        lstm_info = model_versioner.save_model(lstm_model, 'LSTM', {'accuracy': 0.87})

        # Start the FastAPI server in a separate thread
        app_logger.info("Starting FastAPI server")
        fastapi_thread = threading.Thread(target=run_fastapi)
        fastapi_thread.start()

        # Start the web server
        app_logger.info("Starting web server")
        start_server()

    except Exception as e:
        app_logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()