import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def predict_price(models, data, steps=7):
    arima_model = models['arima']
    rf_model = models['rf']

    arima_forecast = arima_model.forecast(steps=steps)
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, steps+1)]
    rf_forecast = rf_model.predict(np.array(range(len(data), len(data)+steps)).reshape(-1, 1))

    ensemble_forecast = (arima_forecast + rf_forecast) / 2
    return pd.Series(ensemble_forecast, index=future_dates)