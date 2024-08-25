import json
from src.data_fetcher import fetch_gold_data
from src.data_processor import prepare_data
from src.model_trainer import train_models
from src.predictor import predict_price
from src.visualizer import plot_predictions
from paper_trader import paper_trader

def main():
    try:
        # Fetch and prepare data
        gold_data = fetch_gold_data()
        prepared_data = prepare_data(gold_data)

        # Train models
        models = train_models(prepared_data)

        # Make predictions
        predictions = predict_price(models, prepared_data)

        # Plot predictions
        plot_predictions(prepared_data, predictions)

        # Get current price (last known price)
        latest_price = float(prepared_data.iloc[-1])

        # Paper trading logic
        if predictions.iloc[0] > latest_price:
            amount_to_buy = min(1000 / latest_price, paper_trader.balance / latest_price)
            if amount_to_buy > 0:
                paper_trader.buy(latest_price, amount_to_buy)
        elif predictions.iloc[0] < latest_price:
            amount_to_sell = min(1, paper_trader.gold_holdings)
            if amount_to_sell > 0:
                paper_trader.sell(latest_price, amount_to_sell)

        result = {
            'latest_price': latest_price,
            'predictions': predictions.to_dict(),
            'portfolio_value': paper_trader.get_portfolio_value(latest_price),
            'recent_trades': paper_trader.get_recent_trades(),
            'balance': paper_trader.balance,
            'gold_holdings': paper_trader.gold_holdings
        }

        return json.dumps(result, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    print(main())