from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

def train_models(data):
    # ARIMA model
    arima_model = ARIMA(data, order=(1,1,1))
    arima_model = arima_model.fit()

    # Random Forest model
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return {'arima': arima_model, 'rf': rf_model}