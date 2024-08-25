import os
import json
from datetime import datetime
from src.utils.logger import app_logger

class ModelVersioner:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.version_file = os.path.join(model_dir, 'versions.json')
        self.versions = self._load_versions()

    def _load_versions(self):
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_versions(self):
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

    def save_model(self, model, model_name, metrics):
        version = len(self.versions.get(model_name, [])) + 1
        timestamp = datetime.now().isoformat()
        filename = f"{model_name}_v{version}.joblib"
        
        model_info = {
            'version': version,
            'timestamp': timestamp,
            'filename': filename,
            'metrics': metrics
        }

        if model_name not in self.versions:
            self.versions[model_name] = []
        
        self.versions[model_name].append(model_info)
        self._save_versions()

        model_path = os.path.join(self.model_dir, filename)
        joblib.dump(model, model_path)
        
        app_logger.info(f"Saved {model_name} version {version}")
        return model_info

    def load_latest_model(self, model_name):
        if model_name not in self.versions or not self.versions[model_name]:
            app_logger.warning(f"No versions found for {model_name}")
            return None

        latest_version = self.versions[model_name][-1]
        model_path = os.path.join(self.model_dir, latest_version['filename'])
        
        if not os.path.exists(model_path):
            app_logger.error(f"Model file not found: {model_path}")
            return None

        model = joblib.load(model_path)
        app_logger.info(f"Loaded {model_name} version {latest_version['version']}")
        return model, latest_version

# Usage example:
# versioner = ModelVersioner('models')
# model_info = versioner.save_model(my_model, 'RandomForest', {'accuracy': 0.85})
# loaded_model, version_info = versioner.load_latest_model('RandomForest')