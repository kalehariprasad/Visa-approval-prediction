import os
import sys
import mlflow
import mlflow.sklearn
import dagshub
from src.logger import logging
from src.exception import CustomException
from src.utils import DataHandler, Model
from src.configuration.config import DataPreprocessconfig
from sklearn.neighbors import KNeighborsClassifier


mlflow.set_tracking_uri("http://ec2-34-201-37-209.compute-1.amazonaws.com:5000//")


class Modeling:
    def __init__(self):
        self.data_handler = DataHandler()
        self.model = Model()
        self.config=DataPreprocessconfig()

    def train_model(self):
        try:
            model_class = KNeighborsClassifier
            train_x = self.data_handler.load_numpy_array(self.config.train_x_path)
            train_y = self.data_handler.load_numpy_array(self.config.train_y_path)
            params = {
                "algorithm": "auto",
                "weights": "distance",
                "n_neighbors": 4
            }

            trained_model = self.model.train_model(model_class=model_class, train_x=train_x, train_y=train_y, params=params)
            return trained_model
        except Exception as e:
            raise CustomException(e, sys)
        
    def evaluate_model(self):
        try:
            model, run_id = self.train_model()
            test_x =self.data_handler.load_numpy_array(self.config.test_x_path)
            test_y = self.data_handler.load_numpy_array(self.config.test_y_path)
            report = self.model.evaluate_model(model=model,test_x=test_x,test_y=test_y,training_run_id=run_id)
            return report
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    try:
        di = Modeling()
        report =di.evaluate_model()

    except Exception as e:
        raise CustomException(e,sys)