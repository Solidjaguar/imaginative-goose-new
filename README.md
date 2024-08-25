# Advanced Forex and Gold Prediction System

This project is an advanced AI-driven system for predicting forex (EUR/USD) and gold prices using machine learning and deep learning techniques.

## Features

- Multiple prediction models including Random Forest, LSTM, XGBoost, and ensemble methods
- Real-time data fetching from Alpha Vantage and FRED APIs
- Advanced feature engineering and data processing
- Backtesting framework for strategy evaluation
- Paper trading simulation
- Web interface for easy interaction with the system
- Robust logging system for better debugging and monitoring
- Model versioning for tracking different iterations of models
- Comprehensive unit testing

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo-url/forex-gold-predictor.git
   cd forex-gold-predictor
   ```

2. Run the installation script:
   ```
   bash install.sh
   ```

   Note: For Windows users, please follow the manual installation steps in the script.

3. Install CUDA and cuDNN manually following NVIDIA's official guide.

4. After CUDA and cuDNN installation, install TensorFlow with GPU support:
   ```
   pip install tensorflow==2.9.0
   ```

## Usage

1. Start the prediction system:
   ```
   python src/main.py
   ```

2. Access the web interface at `http://localhost:8080`

## Configuration

- API keys and other configurations can be set in `config/config.yaml`
- Adjust model parameters in `src/models/model_trainer.py`

## Logging

The system uses a robust logging mechanism. Logs are stored in the `logs/` directory and are rotated when they reach 500 MB in size. You can adjust the logging configuration in `src/utils/logger.py`.

## Model Versioning

We use a custom model versioning system to keep track of different iterations of our models. Each model is saved with metadata including version number, timestamp, and performance metrics. You can find the implementation in `src/utils/model_versioner.py`.

## Testing

To run the unit tests:

```
python -m unittest discover tests
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Alpha Vantage for providing financial market data
- FRED (Federal Reserve Economic Data) for economic indicators