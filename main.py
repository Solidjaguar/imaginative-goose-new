import json
from paper_trader import paper_trader
from advanced_gold_predictor import get_latest_predictions
from performance_metrics import calculate_metrics
from risk_management import get_risk_assessment

def main():
    try:
        with open('predictions.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return json.dumps({"error": "Predictions not available yet"})

    latest_price = data['latest_price']
    predictions = data['predictions']
    
    # Get advanced predictions
    advanced_predictions = get_latest_predictions()
    
    # Calculate performance metrics
    metrics = calculate_metrics(paper_trader.trades)
    
    # Get risk assessment
    risk_assessment = get_risk_assessment(latest_price, advanced_predictions)
    
    result = {
        'latest_price': latest_price,
        'predictions': predictions,
        'advanced_predictions': advanced_predictions,
        'portfolio_value': paper_trader.get_portfolio_value(latest_price),
        'recent_trades': paper_trader.get_recent_trades(),
        'balance': paper_trader.balance,
        'gold_holdings': paper_trader.gold_holdings,
        'performance_metrics': metrics,
        'risk_assessment': risk_assessment
    }
    
    return json.dumps(result, default=str)

if __name__ == "__main__":
    print(main())