import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

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
    
    # Feature engineering
    df['Gold_Price_MA7'] = df['Gold_Price'].rolling(window=7).mean()
    df['Gold_Price_MA30'] = df['Gold_Price'].rolling(window=30).mean()
    df['Gold_Price_Volatility'] = df['Gold_Price'].rolling(window=30).std()
    df['USD_Index_Change'] = df['USD_Index'].pct_change()
    df['Oil_Price_Change'] = df['Oil_Price'].pct_change()
    df['SP500_Change'] = df['SP500'].pct_change()
    
    df.dropna(inplace=True)
    
    # Prepare features and target
    features = ['USD_Index', 'Oil_Price', 'SP500', 'Gold_Price_MA7', 'Gold_Price_MA30', 
                'Gold_Price_Volatility', 'USD_Index_Change', 'Oil_Price_Change', 'SP500_Change']
    X = df[features]
    y = df['Gold_Price']
    
    # Train-test split (80-20)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Normalize the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    # Create sequences
    time_steps = 30
    X_train_seq, y_train_seq = create_dataset(pd.DataFrame(X_train_scaled), pd.Series(y_train_scaled.reshape(-1)), time_steps)
    X_test_seq, y_test_seq = create_dataset(pd.DataFrame(X_test_scaled), pd.Series(y_test_scaled.reshape(-1)), time_steps)
    
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler_y

def build_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X_train, y_train, X_test, y_test):
    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
                        callbacks=[early_stopping], verbose=1)
    return model, history

def evaluate_model(model, X_test, y_test, y_test_orig, scaler_y):
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_orig = y_test_orig.reshape(-1, 1)
    
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'R-squared Score: {r2:.2f}')

def plot_results(y_test_orig, y_pred, history):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_orig, label='Actual Gold Price')
    plt.plot(y_pred, label='Predicted Gold Price')
    plt.title('Gold Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.savefig('gold_price_prediction.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('model_loss.png')
    plt.close()

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler_y = load_and_preprocess_data()
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    evaluate_model(model, X_test, y_test, y_test_orig, scaler_y)
    
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    plot_results(y_test_orig, y_pred, history)
    
    print("Model training and evaluation completed. Check 'gold_price_prediction.png' and 'model_loss.png' for visualizations.")