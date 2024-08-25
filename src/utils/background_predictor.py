import time
import json
from ultra_advanced_gold_predictor import fetch_gold_data, prepare_data, predict_price
from paper_trader import paper_trader
from datetime import datetime, timedelta

def update_predictions_and_trade():
    while True:
        try:
            gold_data = fetch_gold_data(interval='1d', period='1mo')
            
            if gold_data.empty:
                print("Failed to fetch gold data")
                time.sleep(3600)  # Wait for an hour before trying again
                continue
            
            prepared_data = prepare_data(gold_data)
            predictions = predict_price(prepared_data)
            
            latest_price = float(prepared_data.iloc[-1])
            predicted_price = float(predictions.iloc[0])
            
            # Simple trading strategy
            if predicted_price > latest_price:
                amount_to_buy = min(1000 / latest_price, paper_trader.balance / latest_price)
                if amount_to_buy > 0:
                    paper_trader.buy(latest_price, amount_to_buy)
            elif predicted_price < latest_price:
                amount_to_sell = min(1, paper_trader.gold_holdings)
                if amount_to_sell > 0:
                    paper_trader.sell(latest_price, amount_to_sell)
            
            # Save the latest predictions
            with open('predictions.json', 'w') as f:
                json.dump({
                    'latest_price': latest_price,
                    'predictions': predictions.to_dict()
                }, f)
            
            print(f"Updated predictions at {datetime.now()}")
            
            # Wait for a day before the next update
            time.sleep(86400)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            time.sleep(3600)  # Wait for an hour before trying again

if __name__ == "__main__":
    update_predictions_and_trade()