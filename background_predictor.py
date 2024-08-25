import time
import json
from ultra_advanced_gold_predictor import fetch_gold_data, prepare_data, train_model, predict_price
from paper_trader import paper_trader

def update_predictions_and_trade():
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
            
            # Save the latest predictions
            with open('latest_predictions.json', 'w') as f:
                json.dump({
                    'latest_price': latest_price,
                    'latest_date': prepared_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction_data': prediction_data
                }, f)
            
            print(f"Updated predictions at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Wait for 15 minutes before the next update
            time.sleep(900)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    update_predictions_and_trade()