import json
from src.data_fetcher import fetch_gold_data
from src.data_processor import prepare_data
from src.model_trainer import train_models
from src.predictor import predict_price, StackingEnsembleModel
from src.visualizer import plot_predictions, plot_performance, calculate_performance_metrics
from src.trading_strategies import moving_average_crossover, rsi_strategy, bollinger_bands_strategy
from paper_trader import paper_trader

def main():
    try:
        # Fetch and prepare data
        gold_data = fetch_gold_data()
        prepared_data = prepare_data(gold_data)

        # Train models
        models = train_models(prepared_data)

        # Create and fit the stacking ensemble model
        ensemble = StackingEnsembleModel()
        X = prepared_data.index.astype(int).values.reshape(-1, 1)
        y = prepared_data.values
        ensemble.fit(X, y)

        # Make predictions
        predictions = predict_price(ensemble, prepared_data)

        # Plot predictions
        plot_predictions(prepared_data, predictions)

        # Get current price (last known price)
        latest_price = float(prepared_data.iloc[-1])

        # Generate trading signals using different strategies
        mac_signals = moving_average_crossover(prepared_data)
        rsi_signals = rsi_strategy(prepared_data)
        bb_signals = bollinger_bands_strategy(prepared_data)

        # Combine signals (you can adjust the weights as needed)
        combined_signals = 0.4 * mac_signals['signal'] + 0.3 * rsi_signals['signal'] + 0.3 * bb_signals['signal']

        # Paper trading logic
        if combined_signals.iloc[-1] > 0:
            amount_to_buy = min(1000 / latest_price, paper_trader.balance / latest_price)
            if amount_to_buy > 0:
                paper_trader.buy(latest_price, amount_to_buy)
        elif combined_signals.iloc[-1] < 0:
            amount_to_sell = min(1, paper_trader.gold_holdings)
            if amount_to_sell > 0:
                paper_trader.sell(latest_price, amount_to_sell)

        # Calculate portfolio value over time
        portfolio = paper_trader.get_portfolio_history(prepared_data)

        # Plot trading performance
        plot_performance(prepared_data, combined_signals, portfolio)

        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(portfolio)

        result = {
            'latest_price': latest_price,
            'predictions': predictions.to_dict(),
            'portfolio_value': paper_trader.get_portfolio_value(latest_price),
            'recent_trades': paper_trader.get_recent_trades(),
            'balance': paper_trader.balance,
            'gold_holdings': paper_trader.gold_holdings,
            'performance_metrics': performance_metrics
        }

        return json.dumps(result, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    print(main())