import os
import sys
import yaml
import pandas as pd
from src.logger import logging
from src.exception import CustomException



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
            logging.info(f'Processed data saved to {file_path}')
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
