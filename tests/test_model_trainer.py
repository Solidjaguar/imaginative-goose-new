import unittest
from unittest.mock import patch, MagicMock
from src.models.model_trainer import ModelTrainer
import numpy as np
import pandas as pd

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = ModelTrainer()

    @patch('src.models.model_trainer.RandomForestRegressor')
    def test_train_random_forest(self, mock_rf):
        mock_model = MagicMock()
        mock_rf.return_value = mock_model

        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.rand(100))

        self.trainer.train_random_forest(X, y)

        mock_rf.assert_called_once()
        mock_model.fit.assert_called_once_with(X, y)

    @patch('src.models.model_trainer.Sequential')
    def test_train_lstm(self, mock_sequential):
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        X = np.random.rand(100, 10, 5)
        y = np.random.rand(100, 1)

        self.trainer.train_lstm(X, y)

        mock_sequential.assert_called_once()
        mock_model.fit.assert_called_once()

if __name__ == '__main__':
    unittest.main()