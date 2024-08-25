import json
from paper_trader import paper_trader

def main():
    try:
        with open('latest_predictions.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return json.dumps({"error": "Predictions not available yet"})

    latest_price = data['latest_price']
    latest_date = data['latest_date']
    prediction_data = data['prediction_data']
    learning_progress = data.get('learning_progress', {})
    
    result = {
        'latest_price': latest_price,
        'latest_date': latest_date,
        'prediction_data': prediction_data,
        'portfolio_value': paper_trader.get_portfolio_value(latest_price),
        'recent_trades': paper_trader.get_recent_trades(hours=24),
        'balance': paper_trader.balance,
        'gold_holdings': paper_trader.gold_holdings,
        'learning_progress': learning_progress
    }
    
    return json.dumps(result, default=str)

if __name__ == "__main__":
    print(main())