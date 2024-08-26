import mlflow
import mlflow.sklearn
import mlflow.keras
from loguru import logger
import os
from datetime import datetime

class ModelVersioner:
    def __init__(self):
        mlflow.set_tracking_uri("file:" + os.path.join(os.getcwd(), "mlruns"))

    def save_model(self, model, model_name, metrics=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"
        
        with mlflow.start_run(run_name=run_name):
            if isinstance(model, dict):  # For LSTM model
                mlflow.keras.log_model(model['model'], model_name)
                mlflow.log_artifact(model['history'], "training_history")
            else:  # For sklearn models
                mlflow.sklearn.log_model(model, model_name)
            
            if metrics:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
            
            logger.info(f"Model {model_name} saved successfully with run ID: {mlflow.active_run().info.run_id}")

    def load_model(self, model_name, run_id=None):
        if run_id:
            model_uri = f"runs:/{run_id}/{model_name}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Model {model_name} loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

# Usage:
# versioner = ModelVersioner()
# versioner.save_model(rf_model, "random_forest", metrics={'mse': mse, 'r2': r2})
# versioner.save_model({'model': lstm_model, 'history': history}, "lstm", metrics={'mse': mse, 'r2': r2})
# loaded_model = versioner.load_model("random_forest")