import yaml
import logging
import argparse
from typing import Dict, Any
from src.data_fetcher import fetch_all_data
from src.data_processor import prepare_data
from src.model_trainer import train_model, load_model, save_model
from src.predictor import make_predictions, evaluate_predictions, save_predictions
from src.visualizer import plot_predictions, plot_feature_importance, plot_correlation_matrix

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.__dict__.update(config_dict)

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        return Config(yaml.safe_load(file))

def setup_logging(config: Config) -> None:
    logging.basicConfig(filename=config.paths['logs'], level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def main(config_path: str) -> None:
    config = load_config(config_path)
    setup_logging(config)

    logging.info("Starting gold and forex prediction process")

    try:
        # Fetch data
        data = fetch_all_data(config)
        logging.info("Data fetched successfully")

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(data, config)
        logging.info("Data prepared successfully")

        # Train model
        model = train_model(X_train, y_train, config)
        logging.info("Model trained successfully")

        # Save model
        save_model(model, config.paths['model'])
        logging.info("Model saved successfully")

        # Make predictions
        predictions = make_predictions(model, X_test)
        logging.info("Predictions made successfully")

        # Evaluate predictions
        evaluation = evaluate_predictions(y_test, predictions)
        logging.info(f"Model evaluation: MSE: {evaluation['mse']:.4f}, MAE: {evaluation['mae']:.4f}, R2: {evaluation['r2']:.4f}")

        # Save predictions
        save_predictions(y_test, predictions, config)
        logging.info("Predictions saved successfully")

        # Visualize results
        plot_predictions(y_test, predictions, config)
        plot_feature_importance(model, X_train.columns)
        plot_correlation_matrix(data['Gold'].to_frame().join(data['Forex']))
        logging.info("Visualizations created successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
    
    logging.info("Gold and forex prediction process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gold and forex prediction process")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)