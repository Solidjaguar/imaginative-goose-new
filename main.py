import json
from ultra_advanced_gold_predictor import fetch_gold_data, prepare_data, train_model, predict_price
from paper_trader import paper_trader

def main():
    gold_data = fetch_gold_data()
    
    if gold_data.empty:
        return json.dumps({"error": "Failed to fetch gold data"})
    
    prepared_data = prepare_data(gold_data)
    model = train_model(prepared_data)
    prediction_data = predict_price(model, prepared_data)
    
    latest_price = float(prepared_data.iloc[-1])
    
    # Simple trading strategy: buy if the next day's predicted price is higher, sell if it's lower
    if prediction_data and prediction_data[0]['Predicted_Price'] > latest_price:
        paper_trader.buy(latest_price, 1)
    elif prediction_data and prediction_data[0]['Predicted_Price'] < latest_price:
        paper_trader.sell(latest_price, 1)
    
    result = {
        'latest_price': latest_price,
        'latest_date': prepared_data.index[-1].strftime('%Y-%m-%d'),
        'prediction_data': prediction_data,
        'portfolio_value': paper_trader.get_portfolio_value(latest_price),
        'recent_trades': paper_trader.get_recent_trades(),
        'balance': paper_trader.balance,
        'gold_holdings': paper_trader.gold_holdings
    }
    
    return json.dumps(result, default=str)

if __name__ == "__main__":
    print(main())