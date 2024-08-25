from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

def train_models(data):
    X = data.index.astype(int).values.reshape(-1, 1)
    y = data.values

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

    for model_name, model in models.items():
        print(f"Tuning {model_name}...")
        grid_search = GridSearchCV(model, param_grids[model_name], cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        
        if -grid_search.best_score_ > best_score:
            best_score = -grid_search.best_score_
            best_model = grid_search.best_estimator_
        
        print(f"{model_name} best score: {-grid_search.best_score_}")
        print(f"{model_name} best params: {grid_search.best_params_}")
        print()

    print(f"Best model: {type(best_model).__name__}")
    print(f"Best score: {best_score}")

    # Save the best model
    joblib.dump(best_model, 'best_model.joblib')
    print("Best model saved as best_model.joblib")

    return {'best_model': best_model}