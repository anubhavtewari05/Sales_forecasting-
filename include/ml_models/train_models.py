import yaml

from include.utils.mlflow_utils import MLflowManager

class ModelTrainer:
    def __init__(self, config: str = '/usr/local/airflow/include/config/ml_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']
        self.training_config = self.config['training']
        self.mlflow_manager = MLflowManager(config_path)
