import unittest
import os
import shutil
from src.utils.model_versioner import ModelVersioner
from sklearn.ensemble import RandomForestRegressor

class TestModelVersioner(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_models'
        os.makedirs(self.test_dir, exist_ok=True)
        self.versioner = ModelVersioner(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_and_load_model(self):
        # Create a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        metrics = {'accuracy': 0.85}

        # Save the model
        model_info = self.versioner.save_model(model, 'TestModel', metrics)

        # Check if the model info is correct
        self.assertEqual(model_info['version'], 1)
        self.assertEqual(model_info['metrics'], metrics)

        # Load the model
        loaded_model, loaded_info = self.versioner.load_latest_model('TestModel')

        # Check if the loaded model info matches the saved info
        self.assertEqual(loaded_info['version'], 1)
        self.assertEqual(loaded_info['metrics'], metrics)

        # Check if the loaded model is of the correct type
        self.assertIsInstance(loaded_model, RandomForestRegressor)

    def test_multiple_versions(self):
        model1 = RandomForestRegressor(n_estimators=10, random_state=42)
        model2 = RandomForestRegressor(n_estimators=20, random_state=42)

        self.versioner.save_model(model1, 'TestModel', {'accuracy': 0.85})
        self.versioner.save_model(model2, 'TestModel', {'accuracy': 0.87})

        # Load the latest model
        _, loaded_info = self.versioner.load_latest_model('TestModel')

        # Check if the latest version is loaded
        self.assertEqual(loaded_info['version'], 2)
        self.assertEqual(loaded_info['metrics']['accuracy'], 0.87)

if __name__ == '__main__':
    unittest.main()