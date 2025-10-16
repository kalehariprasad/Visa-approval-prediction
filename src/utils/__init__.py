import os
import sys
import shutil
import mlflow.sklearn
import mlflow.pyfunc
import yaml
import joblib
import json
from typing import Dict
import pandas as pd
import numpy as np
from datetime import date
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from src.logger import logging
from src.exception import CustomException
from src.configuration.config import ModelConfig
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score,
                             f1_score, roc_auc_score)


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

    def read_yaml_file(self, file_path: str) -> dict:
        try:
            with open(file_path, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys)

    def write_yaml_file(self, file_path: str,
                        content: object, replace: bool = False
                        ) -> None:
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
                f.write("Validation Successful ✅\n"
                        if validation_status else "Validation Failed ❌\n")
                f.write(f"{text.strip()}\n")

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_csv(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def save_object(self, object, file_path: str):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                joblib.dump(object, f)
        except Exception as e:
            raise CustomException(e, sys)

    def load_object(self, file_path: str):
        try:
            with open(file_path, "rb") as f:
                return joblib.load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def save_numpy_array(self, file_path: str, array: np.ndarray):
        """
        Saves a NumPy array to the specified file path (.npy format).
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            np.save(file_path, array)
        except Exception as e:
            raise CustomException(e, sys)

    def load_numpy_array(self, file_path: str) -> np.ndarray:
        """
        Loads a NumPy array from the specified file path.
        """
        try:
            return np.load(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def save_json(self, data: dict, file_path: str):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            raise CustomException(e, sys)

    def save_txt(self, text: str, file_path: str):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            raise CustomException(e, sys)


class Preprocessing:
    def __init__(self):
        pass

    def drop_colums(self, data_frame: pd.DataFrame, drop_colums):
        try:
            df = data_frame.drop(drop_colums, axis=1)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self, dataframe: pd.DataFrame, target_column):
        try:
            x = dataframe.drop(columns=[target_column], axis=1)
            y = dataframe[target_column]
            return x, y
        except Exception as e:
            raise CustomException(e, sys)

    def target_encoding(self, column):
        try:
            encoded = np.where(column == 'Denied', 0, 1)
            return encoded
        except Exception as e:
            raise CustomException(e, sys)

    def age_caluculate(self, data_frame: pd.DataFrame):
        try:

            todays_date = date.today()
            current_year = todays_date.year
            age = current_year-data_frame["yr_of_estab"]
            data_frame["company_age"] = age
            data_frame.drop("yr_of_estab", inplace=True, axis=1)
            return data_frame
        except Exception as e:
            raise CustomException(e, sys)


class Model:
    def __init__(self):

        self.model_config = ModelConfig()
        self.handler = DataHandler()
        self.mlflow = MLFlowInstance()

    def train_model(self, model_class, train_x, train_y, params: dict):
        try:
            mlflow.set_experiment("production")
            with mlflow.start_run(run_name='model_training') as run:
                logging.info(f"train_y array type: {type(train_y)}\n"
                             " shape: {train_y.shape}")
                train_y = train_y.ravel()

                logging.info(f"train_y array type: {type(train_x)} \n"
                             "shape: {train_x.shape}")
                logging.info(f"train_y array type: {type(train_y)} \n"
                             " shape: {train_y.shape}")
                model = model_class(**params)
                model.fit(train_x, train_y)
                # Log hyperparameters
                for key, value in params.items():
                    mlflow.log_param(key, value)

                # Infer model input/output signature
                predictions = model.predict(train_x)
                signature = infer_signature(train_x, predictions)
                # Log the trained model
                input_example = (
                                    train_x.iloc[:5]
                                    if hasattr(train_x, "iloc")
                                    else train_x[:5]
                                )
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example
                )
                training_run_id = run.info.run_id

            return model, training_run_id
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, model, test_x, test_y, training_run_id):
        try:

            with mlflow.start_run(run_name='model_evaluation'):
                predicted = model.predict(test_x)
                acc = accuracy_score(test_y, predicted)
                f1 = f1_score(test_y, predicted)
                precision = precision_score(test_y, predicted)
                recall = recall_score(test_y, predicted)
                roc_auc = roc_auc_score(test_y, predicted)

                # Log metrics
                mlflow.log_metrics({
                    'accuracy': acc,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'roc_auc': roc_auc
                })

                # Classification report
                report_dict = classification_report(test_y,
                                                    predicted, output_dict=True
                                                    )
                report_txt = classification_report(test_y, predicted)

                # Save report
                self.handler.save_json(report_dict,
                                       self.model_config.report_json
                                       )
                self.handler.save_txt(report_txt,
                                      self.model_config.report_txt
                                      )

                # Log artifacts
                mlflow.log_artifact(self.model_config.report_json)
                mlflow.log_artifact(self.model_config.report_txt)

                # Save run info
                model_name = self.model_config.model_name
                model_path = self.model_config.model_artifact_path
                model_info_path = self.model_config.model_experiment_info
                model_info_app = self.model_config.app_model_experiment_info
                self.mlflow.save_model_info(run_id=training_run_id,
                                            model_name=model_name,
                                            model_path=model_path,
                                            file_path=model_info_path)
                os.makedirs(os.path.dirname(model_info_app), exist_ok=True)
                shutil.copy(model_info_path, model_info_app)
                logging.info(f"Copied model info to {model_info_app}")
            return report_dict
        except Exception as e:
            raise CustomException(e, sys)


class MLFlowInstance:
    def __init__(self):
        self.model_config = ModelConfig()

    def save_model_info(
        self, run_id: str, model_path: str, file_path: str, model_name: str
    ) -> None:
        """Save the model run ID, path, URI, and name to a JSON file."""
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            model_info = {
                'run_id': run_id,
                'model_path': model_path,
                'model_uri': model_uri,
                'model_name': model_name
            }
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                json.dump(model_info, file, indent=4)
            logging.info(f"Model info saved to {file_path}")

        except Exception as e:
            logging.error('Error occurred while saving the model info: %s',
                          e, exc_info=True)
            raise CustomException(e, sys)

    def load_model_info(self, file_path: str) -> Dict[str, str]:
        """Load model run ID and path from a JSON file."""
        try:
            with open(file_path, 'r') as file:
                model_info = json.load(file)
            logging.info('Model info loaded from %s', file_path)
            return model_info
        except Exception as e:
            logging.error('Error occurred while loading the model info: %s', e)
            raise CustomException(e, sys)

    def register_model(self, model_info: dict):
        """Register the model to the MLflow Model Registry."""
        try:

            model_uri = model_info['model_uri']
            model_name = model_info['model_name']
            model_version = mlflow.register_model(
                model_uri=model_uri, name=model_name
                )
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
               name=model_name,
               version=model_version.version,
               stage="Staging"
            )
            logging.info(f"Model {model_name} version {model_version.version}"
                         " registered and transitioned to Staging.")

        except Exception as e:
            logging.error('Error during model registration: %s', e)
            raise CustomException(e, sys)

    def model_stage_transfer(self, model_info: dict):
        """Transfer the latest model in 'Staging' to
        'Production' in MLflow Model Registry."""
        try:
            model_name = model_info['model_name']
            client = mlflow.tracking.MlflowClient()
            staging_versions = client.get_latest_versions(
                model_name, stages=["Staging"]
                )
            if not staging_versions:
                logging.warning(
                    f"No model in 'Staging' stage for model {model_name}"
                    )
                return
            latest_staging_version = staging_versions[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=latest_staging_version,
                stage="Production",
                # archive_existing_versions=True
            )
            logging.info(
                f"Model {model_name} version {latest_staging_version}\n"
                " transitioned to 'Production'."
                )

        except Exception as e:
            logging.error(
                'Error during model stage transfer to Production: %s', e
                )
            raise CustomException(e, sys)

    def load_model_for_inference(self, model_info):
        try:
            model_uri = model_info['model_uri']
            model = mlflow.pyfunc.load_model(model_uri)
            return model
        except Exception as e:
            raise CustomException(e, sys)
