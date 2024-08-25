import json
from src.data_fetcher import fetch_all_data
from src.data_processor import prepare_data
from src.model_trainer import train_models
from src.predictor import predict_prices
from src.visualizer import plot_predictions, plot_performance, calculate_performance_metrics
from src.trading_strategies import moving_average_crossover, rsi_strategy, bollinger_bands_strategy
from paper_trader import PaperTrader

def main():
    try:
        # Fetch and prepare data
        raw_data = fetch_all_data()
        prepared_data = prepare_data(raw_data)

        # Train models
        models = train_models(prepared_data)

        # Make predictions
        predictions = predict_prices(prepared_data)

        # Plot predictions
        plot_predictions(prepared_data, predictions)

        # Initialize paper traders for each market
        paper_traders = {market: PaperTrader() for market in prepared_data.keys()}

        # Generate trading signals and perform paper trading
        signals = {}
        portfolios = {}
        for market, market_data in prepared_data.items():
            # Generate trading signals using different strategies
            mac_signals = moving_average_crossover(market_data['price'])
            rsi_signals = rsi_strategy(market_data['price'])
            bb_signals = bollinger_bands_strategy(market_data['price'])

            # Combine signals (you can adjust the weights as needed)
            combined_signals = 0.4 * mac_signals['signal'] + 0.3 * rsi_signals['signal'] + 0.3 * bb_signals['signal']
            signals[market] = combined_signals

            # Paper trading logic
            trader = paper_traders[market]
            latest_price = float(market_data['price'].iloc[-1])

            if combined_signals.iloc[-1] > 0:
                amount_to_buy = min(1000 / latest_price, trader.balance / latest_price)
                if amount_to_buy > 0:
                    trader.buy(latest_price, amount_to_buy)
            elif combined_signals.iloc[-1] < 0:
                amount_to_sell = min(1, trader.asset_holdings)
                if amount_to_sell > 0:
                    trader.sell(latest_price, amount_to_sell)

            # Calculate portfolio value over time
            portfolios[market] = trader.get_portfolio_history(market_data['price'])

        # Plot trading performance
        plot_performance(prepared_data, signals, portfolios)

        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(portfolios)

        result = {
            'latest_prices': {market: float(data['price'].iloc[-1]) for market, data in prepared_data.items()},
            'predictions': {market: pred.to_dict() for market, pred in predictions.items()},
            'portfolio_values': {market: trader.get_portfolio_value(prepared_data[market]['price'].iloc[-1]) for market, trader in paper_traders.items()},
            'recent_trades': {market: trader.get_recent_trades() for market, trader in paper_traders.items()},
            'balances': {market: trader.balance for market, trader in paper_traders.items()},
            'asset_holdings': {market: trader.asset_holdings for market, trader in paper_traders.items()},
            'performance_metrics': performance_metrics
        }

        return json.dumps(result, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    print(main())