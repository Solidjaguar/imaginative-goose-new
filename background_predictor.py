import time
import json
from ultra_advanced_gold_predictor import fetch_gold_data, prepare_data, train_model, predict_price
from paper_trader import paper_trader
from datetime import datetime, timedelta

def calculate_accuracy(predictions, actual_prices):
    correct = sum(1 for pred, actual in zip(predictions, actual_prices) if (pred > 0 and actual > 0) or (pred < 0 and actual < 0))
    return correct / len(predictions) if predictions else 0

def update_predictions_and_trade():
    learning_progress = {
        'prediction_accuracy': [],
        'portfolio_value': [],
        'model_confidence': []
    }

    while True:
        try:
            gold_data = fetch_gold_data(interval='15m', period='1d')
            
            if gold_data.empty:
                print("Failed to fetch gold data")
                time.sleep(60)
                continue
            
            prepared_data = prepare_data(gold_data)
            model = train_model(prepared_data)
            prediction_data = predict_price(model, prepared_data, steps=4)
            
            latest_price = float(prepared_data.iloc[-1])
            
            # Trading strategy
            for prediction in prediction_data:
                predicted_price = prediction['Predicted_Price']
                confidence = prediction['Confidence']
                
                if confidence > 0.8:  # Only trade if confidence is high
                    if predicted_price > latest_price:
                        amount_to_buy = min(0.1, paper_trader.balance / latest_price)
                        if amount_to_buy > 0:
                            paper_trader.buy(latest_price, amount_to_buy)
                    elif predicted_price < latest_price:
                        amount_to_sell = min(0.1, paper_trader.gold_holdings)
                        if amount_to_sell > 0:
                            paper_trader.sell(latest_price, amount_to_sell)
            
            # Update learning progress
            if len(learning_progress['prediction_accuracy']) >= 96:  # Keep last 24 hours of data (96 15-minute intervals)
                learning_progress['prediction_accuracy'].pop(0)
                learning_progress['portfolio_value'].pop(0)
                learning_progress['model_confidence'].pop(0)

            # Calculate prediction accuracy
            previous_predictions = [pred['Predicted_Price'] for pred in json.loads(open('latest_predictions.json').read())['prediction_data']]
            actual_prices = prepared_data[-4:].tolist()
            accuracy = calculate_accuracy([p - latest_price for p in previous_predictions], [a - latest_price for a in actual_prices])
            
            learning_progress['prediction_accuracy'].append(accuracy)
            learning_progress['portfolio_value'].append(paper_trader.get_portfolio_value(latest_price))
            learning_progress['model_confidence'].append(sum(pred['Confidence'] for pred in prediction_data) / len(prediction_data))

            # Save the latest predictions and learning progress
            with open('latest_predictions.json', 'w') as f:
                json.dump({
                    'latest_price': latest_price,
                    'latest_date': prepared_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction_data': prediction_data,
                    'learning_progress': learning_progress
                }, f)
            
            print(f"Updated predictions at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Wait for 15 minutes before the next update
            time.sleep(900)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    update_predictions_and_trade()
