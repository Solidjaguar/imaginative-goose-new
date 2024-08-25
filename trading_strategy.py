import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def calculate_indicators(data):
    indicators = pd.DataFrame(index=data.index)
    
    # Simple Moving Average (SMA)
    indicators['SMA_short'] = data.rolling(window=10).mean()
    indicators['SMA_long'] = data.rolling(window=30).mean()
    
    # Relative Strength Index (RSI)
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    indicators['MACD'] = exp1 - exp2
    indicators['Signal_Line'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    indicators['BB_middle'] = data.rolling(window=20).mean()
    indicators['BB_upper'] = indicators['BB_middle'] + 2 * data.rolling(window=20).std()
    indicators['BB_lower'] = indicators['BB_middle'] - 2 * data.rolling(window=20).std()
    
    return indicators

def sma_crossover_strategy(indicators, current_position):
    if indicators['SMA_short'] > indicators['SMA_long']:
        return 1  # Buy signal
    elif indicators['SMA_short'] < indicators['SMA_long']:
        return -1  # Sell signal
    else:
        return current_position  # Hold current position

def rsi_strategy(indicators, current_position):
    if indicators['RSI'] < 30:
        return 1  # Oversold, buy signal
    elif indicators['RSI'] > 70:
        return -1  # Overbought, sell signal
    else:
        return current_position  # Hold current position

def macd_strategy(indicators, current_position):
    if indicators['MACD'] > indicators['Signal_Line']:
        return 1  # Buy signal
    elif indicators['MACD'] < indicators['Signal_Line']:
        return -1  # Sell signal
    else:
        return current_position  # Hold current position

def bollinger_bands_strategy(indicators, current_position, current_price):
    if current_price < indicators['BB_lower']:
        return 1  # Price below lower band, buy signal
    elif current_price > indicators['BB_upper']:
        return -1  # Price above upper band, sell signal
    else:
        return current_position  # Hold current position

def combined_strategy(indicators, current_position, current_price):
    sma_signal = sma_crossover_strategy(indicators, current_position)
    rsi_signal = rsi_strategy(indicators, current_position)
    macd_signal = macd_strategy(indicators, current_position)
    bb_signal = bollinger_bands_strategy(indicators, current_position, current_price)
    
    # Simple majority voting
    signals = [sma_signal, rsi_signal, macd_signal, bb_signal]
    buy_votes = sum(1 for signal in signals if signal == 1)
    sell_votes = sum(1 for signal in signals if signal == -1)
    
    if buy_votes > sell_votes:
        return 1  # Buy signal
    elif sell_votes > buy_votes:
        return -1  # Sell signal
    else:
        return current_position  # Hold current position

def optimize_strategy(historical_data, paper_trading_results):
    X = historical_data.drop(['Returns'], axis=1)
    y = historical_data['Returns']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # Use the top 5 most important features
    top_features = feature_importance.head(5).index.tolist()
    
    def optimized_strategy(indicators, current_position, current_price):
        signals = []
        for feature in top_features:
            if feature in ['SMA_short', 'SMA_long']:
                signals.append(sma_crossover_strategy(indicators, current_position))
            elif feature == 'RSI':
                signals.append(rsi_strategy(indicators, current_position))
            elif feature in ['MACD', 'Signal_Line']:
                signals.append(macd_strategy(indicators, current_position))
            elif feature in ['BB_upper', 'BB_lower', 'BB_middle']:
                signals.append(bollinger_bands_strategy(indicators, current_position, current_price))
        
        buy_votes = sum(1 for signal in signals if signal == 1)
        sell_votes = sum(1 for signal in signals if signal == -1)
        
        if buy_votes > sell_votes:
            return 1  # Buy signal
        elif sell_votes > buy_votes:
            return -1  # Sell signal
        else:
            return current_position  # Hold current position
    
    return optimized_strategy

def apply_risk_management(suggested_position, entry_price, current_price, stop_loss=0.02, take_profit=0.04):
    if suggested_position == 1:  # Long position
        if current_price <= entry_price * (1 - stop_loss):
            return 0  # Close position (stop loss)
        elif current_price >= entry_price * (1 + take_profit):
            return 0  # Close position (take profit)
    elif suggested_position == -1:  # Short position
        if current_price >= entry_price * (1 + stop_loss):
            return 0  # Close position (stop loss)
        elif current_price <= entry_price * (1 - take_profit):
            return 0  # Close position (take profit)
    
    return suggested_position  # Maintain current position

def calculate_position_size(account_balance, risk_per_trade=0.02):
    return account_balance * risk_per_trade