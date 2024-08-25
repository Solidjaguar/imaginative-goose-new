from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(X_train, y_train, config):
    model_params = config['model']['params']
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, config['paths']['model'])
    
    return model

def load_model(config):
    return joblib.load(config['paths']['model'])