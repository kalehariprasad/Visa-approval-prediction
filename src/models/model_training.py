import sys
import os
import mlflow
import mlflow.sklearn
from src.logger import logging
from src.exception import CustomException
from src.utils import DataHandler, Model
from src.configuration.config import DataPreprocessconfig, ModelConfig
from catboost import CatBoostClassifier

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
# tracking_uri = os.getenv("http://ec2-3-83-240-213.compute-1.amazonaws.com:5000/")

if not tracking_uri:
   raise CustomException("No MLFLOW_TRACKING_URI in environment", sys)
mlflow.set_tracking_uri(tracking_uri)
logging.info(f"MLflow tracking URI set to {tracking_uri}")


class Modeling:
    def __init__(self):
        self.data_handler = DataHandler()
        self.model = Model()
        self.config = DataPreprocessconfig()
        self.model_config = ModelConfig()

    def train_model(self):
        try:
            model_class = CatBoostClassifier
            train_x = self.data_handler.load_numpy_array(
                self.config.train_x_path
                )
            train_y = self.data_handler.load_numpy_array(
                self.config.train_y_path
                )

            params = {
                'learning_rate': 0.1,
                'l2_leaf_reg': 3,
                'iterations': 200,
                'depth': 6,
                'border_count': 64,
            }

            trained_model = self.model.train_model(
                model_class=model_class,
                train_x=train_x,
                train_y=train_y,
                params=params
            )
            return trained_model
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self):
        try:
            model, run_id = self.train_model()
            test_x = self.data_handler.load_numpy_array(
                self.config.test_x_path
                )
            test_y = self.data_handler.load_numpy_array(
                self.config.test_y_path
                )

            report = self.model.evaluate_model(
                model=model,
                test_x=test_x,
                test_y=test_y,
                training_run_id=run_id
            )

            self.data_handler.save_object(
                object=model,
                file_path=self.model_config.model_path_loacl
            )

            model_info_path = self.model_config.report_json_app
            self.model.mlflow.save_model_info(
                run_id=run_id,
                model_name=self.model_config.model_name,
                model_path=self.model_config.model_artifact_path,
                file_path=model_info_path
            )
            logging.info(f"✅ Model info saved to {model_info_path}")

            logging.info('Model saved locally.')
            return report

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        di = Modeling()
        report = di.evaluate_model()
    except Exception as e:
        raise CustomException(e, sys)
