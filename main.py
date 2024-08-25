import json
from src.utils.data_fetcher import fetch_all_data
from src.utils.data_processor import prepare_data
from src.models.model_trainer import train_models
from src.models.predictor import predict_prices, AdaptivePredictor
from src.utils.visualizer import plot_predictions, plot_performance, calculate_performance_metrics
from src.strategies.trading_strategies import moving_average_crossover, rsi_strategy, bollinger_bands_strategy
from src.strategies.paper_trader import PaperTrader

# ... rest of the main.py content remains the same ...