import yaml
import logging
import argparse
from typing import Dict, Any
from ultra_advanced_gold_predictor import (
    fetch_all_data,
    prepare_data,
    train_model,
    make_predictions,
    evaluate_predictions,
    save_predictions,
    plot_predictions,
    plot_feature_importance,
    plot_correlation_matrix
)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(config: Dict[str, Any]) -> None:
    logging.basicConfig(filename=config['paths']['logs'], level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def main(config_path: str) -> None:
    config = load_config(config_path)
    setup_logging(config)

    logging.info("Starting gold prediction process")

    try:
        # Fetch data
        data = fetch_all_data(config)
        logging.info("Data fetched successfully")

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(data, config)
        logging.info("Data prepared successfully")

        # Train model
        models = train_model(X_train, y_train, config)
        logging.info("Models trained successfully")

        # Make predictions
        predictions = make_predictions(models, X_test)
        logging.info("Predictions made successfully")

        # Evaluate predictions
        evaluation = evaluate_predictions(y_test, predictions)
        logging.info(f"Model evaluation: MSE: {evaluation['mse']:.4f}, MAE: {evaluation['mae']:.4f}, R2: {evaluation['r2']:.4f}")

        # Save predictions
        save_predictions(y_test, predictions, config)
        logging.info("Predictions saved successfully")

        # Visualize results
        plot_predictions(y_test, predictions, config)
        plot_feature_importance(models['stacking_model'], X_train.columns)
        plot_correlation_matrix(data['gold'].to_frame().join(data['economic']))
        logging.info("Visualizations created successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
    
    logging.info("Gold prediction process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gold prediction process")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)