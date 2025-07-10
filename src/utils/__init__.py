import os
import sys
import mlflow.sklearn
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import mlflow
import mlflow.sklearn
import dagshub
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve 



class DataHandler:
    def __init__(self):
        pass

    def save_data(
        self, data: pd.DataFrame, file_path: str
    ) -> None:
        """Save the train and test datasets."""
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data.to_csv(file_path, index=False)
            logging.info(f'data saved to {file_path}')
        except Exception as e:
            logging.info(
                'Unexpected error occurred while saving the data: %s', e)
            raise CustomException(e, sys)
        
    def read_yaml_file(self,file_path: str) -> dict:
        try:
            with open(file_path, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) 

    def write_yaml_file(self,file_path: str, content: object, replace: bool = False) -> None:
        try:
            if replace:
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                yaml.dump(content, file)
        except Exception as e:
            raise CustomException(e, sys) 


    def save_text(self, file_path: str, text: str, validation_status: bool):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Validation Successful ✅\n" if validation_status else "Validation Failed ❌\n")
                f.write(f"{text.strip()}\n")

        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def read_csv(file_path) ->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        

    def save_object(self,object,file_path:str):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                joblib.dump(object, f)
        except Exception as e:
            raise CustomException (e,sys)
    def save_numpy_array(self,file_path: str, array: np.ndarray):
        """
        Saves a NumPy array to the specified file path (.npy format).
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            np.save(file_path, array)
        except Exception as e:
            raise CustomException(e, sys)
        
    def load_numpy_array(self,file_path: str) -> np.ndarray:
        """
        Loads a NumPy array from the specified file path.
        """
        try:
            return np.load(file_path)
        except Exception as e:
            raise CustomException(e, sys)


class Preprocessing:
    def __init__(self):
        pass

    def drop_colums(self,data_frame:pd.DataFrame,drop_colums):
        try:
            df = data_frame.drop(drop_colums,axis=1)
            return df
        except Exception as e:
            raise CustomException (e,sys)
        
    def split_data(self,dataframe:pd.DataFrame,target_column):
        try:
            x = dataframe.drop(columns=[target_column],axis=1)
            y = dataframe[target_column]
            return x,y 
        except Exception as e:
            raise CustomException(e,sys)
        
    def target_encoding(self,column):
        try:
            encoded = np.where(column == 'Denied', 0, 1)
            return encoded 
        except Exception as e:
            raise CustomException(e,sys)
        
    def age_caluculate(self,data_frame:pd.DataFrame):
        try:

            todays_date = date.today()
            current_year= todays_date.year
            age = current_year-data_frame["yr_of_estab"]
            data_frame["company_age"]=age
            data_frame.drop("yr_of_estab",inplace=True,axis=1)
            return data_frame

        except Exception as e:
            raise CustomException(e,sys)
        
class Model:
    def __init__(self):
        pass

    def train_model(self, model_class, train_x, train_y, params: dict):
        try:
            mlflow.set_experiment("production")
            with mlflow.start_run(run_name='model_training'):
                logging.info(f"train_y array type: {type(train_y)} shape: {train_y.shape}")
                
                train_y = train_y.ravel()

                logging.info(f"train_y array type: {type(train_x)} shape: {train_x.shape}")
                logging.info(f"train_y array type: {type(train_y)} shape: {train_y.shape}")
                model = model_class(**params)
                model.fit(train_x, train_y)
                
                # Log hyperparameters
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # Log the trained model
                #mlflow.sklearn.log_model(model, "model")

            return model
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, model, test_x, test_y):
        try:
            with mlflow.start_run(run_name='model_evaluation'):
                predicted = model.predict(test_x)
                acc = accuracy_score(test_y, predicted)
                mlflow.log_metric('accuracy', acc)
                
                f1 = f1_score(test_y, predicted)
                mlflow.log_metric('f1_score', f1)
                
                precision = precision_score(test_y, predicted)
                mlflow.log_metric('precision', precision)
                
                recall = recall_score(test_y, predicted)
                mlflow.log_metric('recall', recall)
                
                roc_auc = roc_auc_score(test_y, predicted)
                mlflow.log_metric('roc_auc', roc_auc)
                
                report = classification_report(test_y, predicted)
                with open("report.txt", "w") as f:
                    f.write(report)
                mlflow.log_artifact("report.txt")

            return report
        except Exception as e:
            raise CustomException(e, sys)   