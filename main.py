import yaml
import logging
from src.data_fetcher import fetch_all_data
from src.data_processor import prepare_data
from src.model_trainer import train_model, load_model
from src.predictor import make_predictions, evaluate_predictions, save_predictions
from src.visualizer import plot_predictions, plot_feature_importance, plot_correlation_matrix

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

        # Make predictions
        predictions = make_predictions(model, X_test)
        logging.info("Predictions made successfully")

        # Evaluate predictions
        evaluation = evaluate_predictions(y_test, predictions)
        logging.info(f"Model evaluation: {evaluation}")

        # Save predictions
        save_predictions(y_test, predictions, config)
        logging.info("Predictions saved successfully")

        # Visualize results
        plot_predictions(y_test, predictions, config)
        plot_feature_importance(model, X_train.columns)
        plot_correlation_matrix(data['Gold'].to_frame().join(data['Forex']))
        logging.info("Visualizations created successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    
    logging.info("Gold and forex prediction process completed")

if __name__ == "__main__":
    main()