import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_features(df):
    df['MA7'] = df['Gold_Price'].rolling(window=7).mean()
    df['MA30'] = df['Gold_Price'].rolling(window=30).mean()
    df['Volatility'] = df['Gold_Price'].rolling(window=30).std()
    df['USD_Index_Change'] = df['USD_Index'].pct_change()
    df['Oil_Price_Change'] = df['Oil_Price'].pct_change()
    df['SP500_Change'] = df['SP500'].pct_change()
    df.dropna(inplace=True)
    return df

def load_and_preprocess_data():
    # In a real scenario, you would load your data from a file or database
    # For this example, we'll create some dummy data
    dates = pd.date_range(start='2010-01-01', end='2023-05-31', freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Gold_Price': np.random.randn(len(dates)).cumsum() + 1000,
        'USD_Index': np.random.randn(len(dates)).cumsum() + 90,
        'Oil_Price': np.random.randn(len(dates)).cumsum() + 60,
        'SP500': np.random.randn(len(dates)).cumsum() + 2000,
    })
    df.set_index('Date', inplace=True)
    
    df = create_features(df)
    
    features = ['USD_Index', 'Oil_Price', 'SP500', 'MA7', 'MA30', 
                'Volatility', 'USD_Index_Change', 'Oil_Price_Change', 'SP500_Change']
    X = df[features]
    y = df['Gold_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'R-squared Score: {r2:.2f}')
    
    return y_pred

def print_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_importances = sorted(zip(importances, feature_names), reverse=True)
    print("\nFeature Importances:")
    for importance, feature in feature_importances:
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    
    feature_names = ['USD_Index', 'Oil_Price', 'SP500', 'MA7', 'MA30', 
                     'Volatility', 'USD_Index_Change', 'Oil_Price_Change', 'SP500_Change']
    print_feature_importance(model, feature_names)
    
    print("\nModel training and evaluation completed.")