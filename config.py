import os

# API Keys
FRED_API_KEY = os.getenv('FRED_API_KEY', 'YOUR_FRED_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'YOUR_NEWS_API_KEY')

# Data parameters
LOOKBACK_DAYS = 30
TRAIN_TEST_SPLIT = 0.8

# Model parameters
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_DENSE_UNITS = 32
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32

# Ensemble model names
MODEL_NAMES = ['rf', 'xgb', 'lgbm', 'elasticnet']

# Feature selection
N_FEATURES_TO_SELECT = 30

# Anomaly detection
ANOMALY_CONTAMINATION = 0.1

# Prediction interval confidence
PREDICTION_INTERVAL_CONFIDENCE = 0.95

# Retraining schedule
RETRAIN_TIME = "00:00"  # Midnight

# Caching
CACHE_EXPIRY = 86400  # 24 hours in seconds

# Database
DB_NAME = 'performance_metrics.db'

# Symbols
GOLD_SYMBOL = "GC=F"
RELATED_ASSETS = [('Silver', "SI=F"), ('Oil', "CL=F"), ('SP500', "^GSPC")]

# FRED Indicators
FRED_INDICATORS = {
    'CPIAUCSL': 'Inflation_Rate',
    'FEDFUNDS': 'Interest_Rate',
    'DTWEXBGS': 'USD_Index',
    'GDP': 'GDP',
    'UNRATE': 'Unemployment_Rate'
}

# File paths
CHECKPOINT_FILE = 'model_checkpoint.joblib'
CACHE_DIR = 'cache'

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'