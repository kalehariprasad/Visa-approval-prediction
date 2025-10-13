import os,sys
from app.logger import logging
from app.exception import CustomException
import joblib
import json
from typing import Dict
import mlflow



class streamlitutilies:
    def __init__(self):
        pass
    def load_object(self, file_path: str):
        """
        Load the object from the specified file path using joblib.
        Logs the status and raises a custom exception if it fails.
        """
        try:   
            obj = joblib.load(file_path)
            logging.info(f"Object successfully loaded from: {file_path}")
            return obj
        except Exception as e:
            logging.error(f"Failed to load object from {file_path}. Error: {e}")
            raise CustomException(e, sys)

    def load_model_info(self,file_path: str) -> Dict[str, str]:
        """
        Load model run ID and path from a JSON file.
        """
        try:
            with open(file_path, 'r') as file:
                model_info = json.load(file)
            logging.info('Model info loaded from %s', file_path)
            return model_info  
        except Exception as e:
            logging.error('Error occurred while loading the model info: %s', e)
            raise CustomException(e, sys)
        
    def load_model_for_inference(self,model_info):
        """"
        Load the Model for inferencing from model info
        """
        try:
            model_uri = model_info['model_uri']
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info ('model loded from mlflow model registry')
            return model
        except Exception as e:
            logging.error('Error occurred while loading the model for inference: %s', e)
            raise CustomException (e,sys)

    