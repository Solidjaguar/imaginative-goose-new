import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def make_predictions(model, X_test):
    return model.predict(X_test)

def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def save_predictions(y_true, y_pred, config):
    predictions_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    predictions_df.to_csv(config['paths']['predictions'], index=True)
    return predictions_df