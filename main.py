import json
from paper_trader import paper_trader

def main():
    try:
        with open('predictions.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return json.dumps({"error": "Predictions not available yet"})

    latest_price = data['latest_price']
    predictions = data['predictions']
    
    result = {
        'latest_price': latest_price,
        'predictions': predictions,
        'portfolio_value': paper_trader.get_portfolio_value(latest_price),
        'recent_trades': paper_trader.get_recent_trades(),
        'balance': paper_trader.balance,
        'gold_holdings': paper_trader.gold_holdings
    }
    
    return json.dumps(result, default=str)

if __name__ == "__main__":
    print(main())