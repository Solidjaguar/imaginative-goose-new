import yaml
import logging
from src.data_fetcher import fetch_all_data
from src.data_processor import prepare_data
from src.model_trainer import train_model
from src.predictor import make_predictions
from src.visualizer import plot_predictions

def load_config():
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def setup_logging(config):
    logging.basicConfig(filename=config['paths']['logs'], level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = load_config()
    setup_logging(config)

    logging.info("Starting gold and forex prediction process")

    # Fetch data
    data = fetch_all_data(config)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data, config)

    # Train model
    model, scaler = train_model(X_train, y_train, config)

    # Make predictions
    predictions = make_predictions(model, scaler, X_test)

    # Visualize results
    plot_predictions(y_test, predictions, config)

    logging.info("Gold and forex prediction process completed")

if __name__ == "__main__":
    main()