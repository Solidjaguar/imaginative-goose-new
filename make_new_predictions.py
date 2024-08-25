import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import logging
from gold_forex_predictor import fetch_all_data, fetch_news_sentiment, prepare_data

logging.basicConfig(filename='new_predictions.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_scaler():
    model = joblib.load('trained_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

def make_new_predictions():
    # Load the trained model and scaler
    model, scaler = load_model_and_scaler()

    # Fetch new data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Get last 30 days of data
    
    data = fetch_all_data(start_date, end_date)
    news_sentiment = fetch_news_sentiment(start_date, end_date)
    
    df = prepare_data(data, news_sentiment)

    # Prepare features
    features = df.drop(['Close'], axis=1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Make predictions
    predictions = model.predict(features_scaled)

    # Create a DataFrame with dates and predictions
    prediction_df = pd.DataFrame({
        'Date': df.index,
        'Predicted_Price': predictions,
        'Actual_Price': df['Close']
    })

    # Save new predictions to CSV
    prediction_df.to_csv('new_predictions.csv', index=False)
    logging.info("New predictions saved to 'new_predictions.csv'")

    return prediction_df

if __name__ == "__main__":
    new_predictions = make_new_predictions()
    print(new_predictions.tail())  # Display the last few predictions