import os
import sys 
import pandas as pd
from src.configuration.config import DataInjectionConfig,DataValidationconfig
from src.exception import CustomException
from src.logger import logging
from src.utils import  DataHandler
from src.constants import SCHEMA_FILE_PATH



class Datavalidation:
    def __init__(self,data_injection_artifact:DataInjectionConfig,data_validation_artifact:DataValidationconfig,data_handler:DataHandler):
        """
        param data_injection_artifact: output reference for datainjection artfact
        param data_validation_artifact: output reference for datavalidation artfact
        param data_handler: utilities used for commonnfunctiond
        """
        try:
            self.data_ingestion_artifact = data_injection_artifact
            self.data_validation_config = data_validation_artifact
            self.utilities =data_handler 
            self._schema_config = data_handler.read_yaml_file(SCHEMA_FILE_PATH)      
        except Exception as e:
            raise CustomException(e,sys)
    def validate_number_of_columns(self, dataframe:pd.DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            actual_columns = dataframe.columns.tolist()
            expected_columns = list(self._schema_config["columns"])
            logging.info(f"Actual columns ({(expected_columns)}): {actual_columns}")
            logging.info(f"Is required column present: [{status}]")
      
            return status
        except Exception as e:
            raise CustomException(e, sys)
        
    def is_column_exist(self, df:pd.DataFrame) -> bool:
    
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_data_validation(self) :
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
         
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df =self.utilities.read_csv(file_path=self.data_ingestion_artifact.train_file_path)
            test_df = self.utilities.read_csv(file_path= self.data_ingestion_artifact.test_file_path)
            

            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."

            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.is_column_exist(df=test_df)

            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0
            self.utilities.save_text(
                file_path=self.data_validation_config.validation_file,
                text=validation_error_msg,
                validation_status=validation_status
            )

            logging.info(f"Validation Successful ✅\n" if {validation_status} else {validation_error_msg})

            return validation_status
        except Exception as e:
            raise CustomException(e, sys) 
        

if __name__ == "__main__":
    try:
        di = Datavalidation(DataInjectionConfig(), DataValidationconfig(), DataHandler())

        status = di.initiate_data_validation()
        if not status:
            logging.error("❌ Data validation failed. Exiting pipeline.")
            sys.exit(1)  # ← This line stops the pipeline if validation fails
        logging.info("✅ Data validation passed.")
    except Exception as e:
        logging.error(f"Failed to inject data: {e}")
        sys.exit(1)