import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data():
    # In a real scenario, you would load your prepared data here
    # For this example, we'll create some dummy data
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', end='2023-05-31', freq='D')
    X = pd.DataFrame({
        'feature1': np.random.randn(len(dates)),
        'feature2': np.random.randn(len(dates)),
        'feature3': np.random.randn(len(dates)),
    }, index=dates)
    y = pd.Series(np.random.randn(len(dates)), index=dates)
    return X, y

def advanced_model_selection(X, y):
    # Define models to test
    models = {
        'RandomForest': RandomForestRegressor(),
        'Lasso': Lasso(),
        'SVR': SVR()
    }

    # Define parameter grids for each model
    param_grids = {
        'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]},
        'Lasso': {'alpha': [0.1, 1.0, 10.0]},
        'SVR': {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
    }

    # Perform time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_score = float('-inf')
    best_params = None

    for model_name, model in models.items():
        print(f"Tuning {model_name}...")
        grid_search = GridSearchCV(model, param_grids[model_name], cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        
        if -grid_search.best_score_ > best_score:
            best_score = -grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        
        print(f"{model_name} best score: {-grid_search.best_score_}")
        print(f"{model_name} best params: {grid_search.best_params_}")
        print()

    print(f"Best model: {type(best_model).__name__}")
    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")

    return best_model

def make_predictions(model, X):
    return model.predict(X)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'R2': r2}

def save_model(model, filename='best_model.joblib'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Perform advanced model selection
    best_model = advanced_model_selection(X, y)
    
    # Make predictions
    predictions = make_predictions(best_model, X)
    
    # Evaluate the model
    evaluation = evaluate_model(y, predictions)
    print("Model Evaluation:")
    print(f"MSE: {evaluation['MSE']}")
    print(f"R2: {evaluation['R2']}")
    
    # Save the model
    save_model(best_model)