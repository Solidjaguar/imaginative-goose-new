import os
import multiprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# API keys (your actual keys)
ALPHA_VANTAGE_API_KEY = 'XMTJLR6ZHLQ2JZJL'
FRED_API_KEY = '64c5fd46f9f13ca6cb24ae0c2ebfb08d'

# Model file paths
RANDOM_FOREST_MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
LASSO_MODEL_PATH = os.path.join(MODELS_DIR, 'lasso_model.joblib')
SVR_MODEL_PATH = os.path.join(MODELS_DIR, 'svr_model.joblib')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_model.joblib')
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, 'lgbm_model.joblib')
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.h5')

# Other configuration settings
LOOK_BACK_PERIOD = 30
PREDICTION_DAYS = 7

# Hardware settings
USE_GPU = True
NUM_CPU_CORES = multiprocessing.cpu_count()  # This will detect all available cores