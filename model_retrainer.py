import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import time
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gold_forex_predictor import fetch_all_data, prepare_data, train_model, make_predictions, update_actual_values
from ensemble_model import StackingEnsembleModel, train_stacking_ensemble_model

logging.basicConfig(filename='model_retrainer.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_predictions():
    with open('predictions.json', 'r') as f:
        return json.load(f)

def calculate_performance_metrics(predictions):
    actual_values = []
    predicted_values = []
    
    for pred in predictions:
        if pred['actual'] is not None:
            actual_values.append(pred['actual'])
            predicted_values.append(pred['prediction'])
    
    if not actual_values:
        return None
    
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    mse = mean_squared_error(actual_values, predicted_values, multioutput='raw_values')
    mae = mean_absolute_error(actual_values, predicted_values, multioutput='raw_values')
    r2 = r2_score(actual_values, predicted_values, multioutput='raw_values')
    
    return {
        'MSE': mse.tolist(),
        'MAE': mae.tolist(),
        'R2': r2.tolist()
    }

def should_retrain(current_metrics, previous_metrics, threshold=0.05):
    if previous_metrics is None:
        return True
    
    mse_improvement = (np.array(previous_metrics['MSE']) - np.array(current_metrics['MSE'])) / np.array(previous_metrics['MSE'])
    mae_improvement = (np.array(previous_metrics['MAE']) - np.array(current_metrics['MAE'])) / np.array(previous_metrics['MAE'])
    r2_improvement = (np.array(current_metrics['R2']) - np.array(previous_metrics['R2'])) / np.array(previous_metrics['R2'])
    
    return np.any(mse_improvement > threshold) or np.any(mae_improvement > threshold) or np.any(r2_improvement > threshold)

def save_best_model(model, metrics):
    joblib.dump(model, 'best_model.joblib')
    with open('best_model_metrics.json', 'w') as f:
        json.dump(metrics, f)
    logging.info("Saved new best model")

def retrain_model():
    logging.info("Starting model retraining process")
    
    # Load existing predictions and calculate current performance metrics
    predictions = load_predictions()
    current_metrics = calculate_performance_metrics(predictions)
    
    # Load previous performance metrics
    try:
        with open('performance_metrics.json', 'r') as f:
            previous_metrics = json.load(f)
    except FileNotFoundError:
        previous_metrics = None
    
    if should_retrain(current_metrics, previous_metrics):
        logging.info("Performance threshold met. Retraining model...")
        
        # Retrain the model
        new_model = train_model()
        
        # Make new predictions with the retrained model
        make_predictions(new_model)
        
        # Update actual values
        update_actual_values()
        
        # Calculate new performance metrics
        new_predictions = load_predictions()
        new_metrics = calculate_performance_metrics(new_predictions)
        
        # Save new performance metrics
        with open('performance_metrics.json', 'w') as f:
            json.dump(new_metrics, f)
        
        # Save the model if it's the best so far
        if previous_metrics is None or np.mean(new_metrics['MSE']) < np.mean(previous_metrics['MSE']):
            save_best_model(new_model, new_metrics)
        
        logging.info("Model retrained and new predictions made")
    else:
        logging.info("Performance threshold not met. Skipping retraining")

def run_retrainer(interval_hours=24):
    while True:
        retrain_model()
        logging.info(f"Sleeping for {interval_hours} hours before next retraining check")
        time.sleep(interval_hours * 3600)

if __name__ == "__main__":
    run_retrainer()