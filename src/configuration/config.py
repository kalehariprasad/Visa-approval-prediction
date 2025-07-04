from dataclasses import dataclass
import os
from src.constants import *

@dataclass
class DataInjectionConfig:
    base_path: str = os.path.join(os.getcwd(), ARTIFACTS_DIRECTORY)

    def __post_init__(self):
        self.raw_data_directory = os.path.join(self.base_path, RAW_DATA_DIRECTORY)
        self.raw_file_path = os.path.join(self.raw_data_directory,RAW_DATA_FILE)
        self.train_file_path = os.path.join(self.raw_data_directory,RAW_TRAIN_FILE)
        self.test_file_path = os.path.join(self.raw_data_directory, RAW_TEST_FILE)


@dataclass
class DataValidationconfig:
    base_path: str = os.path.join(os.getcwd(), ARTIFACTS_DIRECTORY)
    def get_validation_file_name(self):
        return f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def __post_init__(self):
        self.validation_folder = os.path.join(self.base_path,DATA_VALIDATION_FOLDER)
        self.validation_file = os.path.join(self.validation_folder, self.get_validation_file_name())

@dataclass
class DataPreprocessconfig:
    base_path = os.path.join(os.getcwd(),ARTIFACTS_DIRECTORY)

    def __post_init__(self):
        self.preprocessed_folder = os.path.join(self.base_path,DATA_PREPROCESSING_FOLDER)
        self.preprocessed_train = os.path.join(self.preprocessed_folder,PREPROCESSED_TRAIN)
        self.preprocessed_test = os.path.join(self.preprocessed_folder,PREPROCESSED_TEST)
        self.preprocessor = os.path.join(self.preprocessed_folder,PREPROCESSOR_OBJECT)
        self.features_folder = os.path.join(self.base_path,FEATURE_FOLDER)
        self.train_x_path = os.path.join(self.features_folder,TRAIN_X)
        self.train_y_path = os.path.join(self.features_folder,TRAIN_Y)
        self.test_x_path = os.path.join(self.features_folder,TEST_X)
        self.test_y_path = os.path.join(self.features_folder,TEST_Y)