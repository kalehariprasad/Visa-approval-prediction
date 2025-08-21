import os
import pandas as pd
import sys
import mlflow
from src.logger import logging
from src.configuration.config import SCHEMA_FILE_PATH
from src.exception import CustomException
from src.utils import MLFlowInstance, Preprocessing, DataHandler
from src.configuration.config import ModelConfig, DataInjectionConfig, DataPreprocessconfig


class Mlflow:
    def __init__(self):
        self.config = ModelConfig()
        self.data_injection_config = DataInjectionConfig()
        self.preprocessing_config = DataPreprocessconfig()
        self.mlflow = MLFlowInstance()
        self.preprocessing = Preprocessing()
        self.data_handler = DataHandler()
        self.schema = self.data_handler.read_yaml_file(SCHEMA_FILE_PATH)  

    def Model_inference(self):
        try:
            mlflow.set_tracking_uri("http://ec2-54-145-130-138.compute-1.amazonaws.com:5000//")
            model_info = self.mlflow.load_model_info(self.config.model_experiment_info)
            model = self.mlflow.load_model_for_inference(model_info)
            df = pd.read_csv(self.data_injection_config.test_file_path)
            fe_data = self.preprocessing.age_caluculate(df)
            train_x, _ = self.preprocessing.split_data(fe_data, self.schema["target_column"])
            preprocessor = self.data_handler.load_object(self.preprocessing_config.preprocessor)
            train_x = preprocessor.transform(train_x)
            predictions = model.predict(train_x)
            print("✅ Sample Predictions:", predictions[:10])

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        di = Mlflow()
        _ = di.Model_inference()
    except Exception as e:
        raise CustomException(e, sys)
