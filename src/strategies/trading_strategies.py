import pandas as pd
import numpy as np

def moving_average_crossover(data, short_window=10, long_window=50):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    signals['short_mavg'] = data.rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data.rolling(window=long_window, min_periods=1, center=False).mean()

    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                                > signals['long_mavg'][short_window:], 1.0, 0.0)   
    signals['positions'] = signals['signal'].diff()

    return signals

def rsi_strategy(data, rsi_window=14, overbought=70, oversold=30):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    rsi = calculate_rsi(data, rsi_window)
    signals['rsi'] = rsi

    signals['signal'] = np.where(rsi > overbought, -1.0, np.where(rsi < oversold, 1.0, 0.0))
    signals['positions'] = signals['signal'].diff()

    return signals

def bollinger_bands_strategy(data, window=20, num_std=2):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    signals['upper_band'] = rolling_mean + (rolling_std * num_std)
    signals['lower_band'] = rolling_mean - (rolling_std * num_std)

    signals['signal'] = np.where(data > signals['upper_band'], -1.0, 
                                 np.where(data < signals['lower_band'], 1.0, 0.0))
    signals['positions'] = signals['signal'].diff()

    return signals

def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)

    for i in range(window, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(window - 1) + upval)/window
        down = (down*(window - 1) + downval)/window
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi