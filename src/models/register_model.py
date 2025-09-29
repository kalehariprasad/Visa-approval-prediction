import sys
import os
import mlflow
from src.logger import logging
from src.exception import CustomException
from src.utils import MLFlowInstance
from src.configuration.config import ModelConfig


class Mlflow:
    def __init__(self):
        self.config = ModelConfig()
        self.mlflow = MLFlowInstance()

    def Model_register(self):
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                raise CustomException("No MLFLOW_TRACKING_URI in environment", sys)
            mlflow.set_tracking_uri(tracking_uri)
            logging.info(f"MLflow tracking URI set to {tracking_uri}")

            model_info = self.mlflow.load_model_info(
                self.config.model_experiment_info
                )
            self.mlflow.register_model(model_info=model_info)
            logging.info("model Registered to Model register")
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        di = Mlflow()
        report = di.Model_register()

    except Exception as e:
        raise CustomException(e, sys)
