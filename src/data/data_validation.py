import sys
import pandas as pd
from src.configuration.config import (
    DataInjectionConfig,
    DataValidationconfig
)
from src.exception import CustomException
from src.logger import logging
from src.utils import DataHandler
from src.constants import SCHEMA_FILE_PATH


class Datavalidation:
    def __init__(
        self,
        data_injection_artifact: DataInjectionConfig,
        data_validation_artifact: DataValidationconfig,
        data_handler: DataHandler
    ):
        """
        param data_injection_artifact:
        output reference for datainjection artifact
        param data_validation_artifact:
        output reference for datavalidation artifact
        param data_handler: utilities used for common functions
        """
        try:
            self.data_ingestion_artifact = data_injection_artifact
            self.data_validation_config = data_validation_artifact
            self.utilities = data_handler
            self._schema_config = data_handler.read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates number of columns in the dataframe.
        """
        try:
            actual_cols = len(dataframe.columns)
            expected_cols = len(self._schema_config["columns"])
            status = actual_cols == expected_cols
            actual_columns = dataframe.columns.tolist()
            expected_columns = list(self._schema_config["columns"])
            logging.info(
                f"Actual columns ({expected_columns}): {actual_columns}"
            )
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise CustomException(e, sys)

    def is_column_exist(self, df: pd.DataFrame) -> bool:
        """
        Checks if expected numerical and categorical columns exist.
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if missing_numerical_columns:
                logging.info(
                    f"Missing numerical column: {missing_numerical_columns}"
                )

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if missing_categorical_columns:
                logging.info(
                    f"Missing categorical column:{missing_categorical_columns}"
                )

            return not (
                missing_numerical_columns or missing_categorical_columns
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_validation(self):
        """
        Initiates data validation.
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            train_df = self.utilities.read_csv(
                file_path=self.data_ingestion_artifact.train_file_path
            )
            test_df = self.utilities.read_csv(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            # Validate train columns
            status = self.validate_number_of_columns(train_df)
            logging.info(
                f"All required columns present in training dataframe: {status}"
            )
            if not status:
                validation_error_msg += (
                    "Columns are missing in training dataframe.\n"
                )

            # Validate test columns
            status = self.validate_number_of_columns(test_df)
            logging.info(
                f"All required columns present in testing dataframe: {status}"
            )
            if not status:
                validation_error_msg += (
                    "Columns are missing in test dataframe.\n"
                )

            # Validate train column existence
            status = self.is_column_exist(train_df)
            if not status:
                validation_error_msg += (
                    "Expected columns missing in training dataframe.\n"
                )

            # Validate test column existence
            status = self.is_column_exist(test_df)
            if not status:
                validation_error_msg += (
                    "Expected columns missing in test dataframe.\n"
                )

            validation_status = len(validation_error_msg.strip()) == 0

            self.utilities.save_text(
                file_path=self.data_validation_config.validation_file,
                text=validation_error_msg.strip(),
                validation_status=validation_status
            )

            if validation_status:
                logging.info("Validation Successful ✅")
            else:
                logging.warning(f"Validation Errors:\n{validation_error_msg}")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        di = Datavalidation(
            DataInjectionConfig(),
            DataValidationconfig(),
            DataHandler()
        )
        status = di.initiate_data_validation()

        if not status:
            logging.error("❌ Data validation failed. Exiting pipeline.")
            sys.exit(1)

        logging.info("✅ Data validation passed.")

    except Exception as e:
        logging.error(f"Failed to inject data: {e}")
        sys.exit(1)
