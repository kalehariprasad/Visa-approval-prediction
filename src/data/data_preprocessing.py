import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from src.constants import SCHEMA_FILE_PATH
from src.configuration.config import DataInjectionConfig,DataPreprocessconfig
from src.exception import CustomException 
from src.logger import logging
from src.utils import DataHandler,Preprocessing



class DataPreprocessing:
    def __init__(self):
        self.data_injection_config = DataInjectionConfig()
        self.data_prepprocessing_config = DataPreprocessconfig()
        self.data_utils = DataHandler()
        self.preproceesing_utils = Preprocessing()
        self.schema = self.data_utils.read_yaml_file(SCHEMA_FILE_PATH)


    def get_preperocessor(self):
        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()
            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
       
            oh_columns = self.schema["one_hot_encoding_columns"]
            or_columns = self.schema["Ordinal_encoding_columns"]
            transform_columns = self.schema["transform_features"]
            num_features = self.schema["num_features"]

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            return preprocessor                        
        except Exception as e:
            raise CustomException(s,sys)
        
    def Initiate_preprocessing(self):
        try:
            train_df= self.data_utils.read_csv(self.data_injection_config.train_file_path)
            test_df = self.data_utils.read_csv(self.data_injection_config.test_file_path)

            train_df = self.preproceesing_utils.drop_colums(train_df,self.schema["drop_columns"])
            test_df = self.preproceesing_utils.drop_colums(test_df, self.schema["drop_columns"])
            logging.info('Reading training and testing files for preprocessing completed.')
            train_x,train_y = self.preproceesing_utils.split_data(dataframe=train_df,target_column=self.schema["target_column"])
            test_x,test_y = self.preproceesing_utils.split_data(dataframe=test_df,target_column=self.schema["target_column"])
            logging.info('Splitting features and target columns for training and testing datasets completed.')
            feature_engineered_train = self.preproceesing_utils.age_caluculate(train_x)
            feature_engineered_test = self.preproceesing_utils.age_caluculate(test_x)
            self.data_utils.save_data(feature_engineered_train,file_path=self.data_prepprocessing_config.preprocessed_train)
            self.data_utils.save_data(feature_engineered_test,self.data_prepprocessing_config.preprocessed_test)
            logging.info(f'Feature engineering (e.g., age calculation) completed on training and testing datasets and saved.')
            preprocessor = self.get_preperocessor()
            train_array = preprocessor.fit_transform(feature_engineered_train)
            test_array = preprocessor.transform(feature_engineered_test)
            logging.info('Preprocessor (scaler/encoder) applied to training and testing features.')
            train_y_array = self.preproceesing_utils.target_encoding(train_y)
            test_y_array = self.preproceesing_utils.target_encoding(test_y)
            logging.info('Target encoding completed on training and testing labels.')
            self.data_utils.save_object(object=preprocessor,file_path= self.data_prepprocessing_config.preprocessor)
            logging.info(f'preprocessor object saved at {self.data_prepprocessing_config.preprocessor}')
            logging.info('Preprocessing completed. Returning train_array, train_y_array, test_array, and test_y_array.')
            self.data_utils.save_numpy_array(file_path=self.data_prepprocessing_config.train_x_path,array=train_array)
            self.data_utils.save_numpy_array(file_path=self.data_prepprocessing_config.train_y_path,array=train_y_array)
            self.data_utils.save_numpy_array(file_path=self.data_prepprocessing_config.test_x_path,array=test_array)
            self.data_utils.save_numpy_array(file_path=self.data_prepprocessing_config.test_y_path,array=test_y_array)
            logging.info(f'train_array saved at{self.data_prepprocessing_config.train_x_path}')
            logging.info(f'train_y_array saved at {self.data_prepprocessing_config.train_y_path}') 
            logging.info(f'test_array saved at {self.data_prepprocessing_config.test_x_path}')
            logging.info(f'test_y_array saved at {self.data_prepprocessing_config.test_y_path}')
            return train_array, train_y_array,test_array,test_y_array

            
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    try:
        di = DataPreprocessing()
        treain_array, train_y_array,test_array,test_y_array = di.Initiate_preprocessing()
    except Exception as e:
        raise CustomException(e,sys)
