import sys
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
            mlflow.set_tracking_uri(
                "http://ec2-44-204-17-132.compute-1.amazonaws.com:5000"
                )
            model_info = self.mlflow.load_model_info(
                self.config.model_experiment_info
                )
            self.mlflow.model_stage_transfer(model_info=model_info)
            logging.info('model Transfered to staging')
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        di = Mlflow()
        report = di.Model_register()

    except Exception as e:
        raise CustomException(e, sys)
