import json
from ultra_advanced_gold_predictor import fetch_gold_data, prepare_data, train_model, predict_price
from paper_trader import paper_trader

def main():
    gold_data = fetch_gold_data(interval='15m', period='1d')
    
    if gold_data.empty:
        return json.dumps({"error": "Failed to fetch gold data"})
    
    prepared_data = prepare_data(gold_data)
    model = train_model(prepared_data)
    prediction_data = predict_price(model, prepared_data, steps=4)
    
    latest_price = float(prepared_data.iloc[-1])
    
    # More aggressive trading strategy
    for prediction in prediction_data:
        predicted_price = prediction['Predicted_Price']
        confidence = prediction['Confidence']
        
        if confidence > 0.8:  # Only trade if confidence is high
            if predicted_price > latest_price:
                amount_to_buy = min(0.1, paper_trader.balance / latest_price)  # Buy up to 0.1 oz or as much as we can afford
                if amount_to_buy > 0:
                    paper_trader.buy(latest_price, amount_to_buy)
            elif predicted_price < latest_price:
                amount_to_sell = min(0.1, paper_trader.gold_holdings)  # Sell up to 0.1 oz or as much as we have
                if amount_to_sell > 0:
                    paper_trader.sell(latest_price, amount_to_sell)
    
    result = {
        'latest_price': latest_price,
        'latest_date': prepared_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'prediction_data': prediction_data,
        'portfolio_value': paper_trader.get_portfolio_value(latest_price),
        'recent_trades': paper_trader.get_recent_trades(hours=24),
        'balance': paper_trader.balance,
        'gold_holdings': paper_trader.gold_holdings
    }
    
    return json.dumps(result, default=str)

if __name__ == "__main__":
    print(main())