import  os 
import sys 
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.mangdb_connection import MongoDBClient
from src.configuration.config import DataInjectionConfig
from sklearn.model_selection import train_test_split
from src.utils import DataHandler



class DataInjection:
    def __init__(self):
        self.mangodb = MongoDBClient()
        self.data_handler = DataHandler()
        self.artifact = DataInjectionConfig()


    def  initiate_data_injection(self):
        try:
            db = self.mangodb
            data= self.data_handler
            artifacts = self.artifact
          
            df = pd.DataFrame(list(db.collection.find()))
            
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            data.save_data(df, artifacts.raw_file_path)
            data.save_data(train_df, artifacts.train_file_path)
            data.save_data(test_df, artifacts.test_file_path)
            
            data.save_data(df,)
            logging.info('testing working')
            
        except Exception as e:
            logging.info(f"error occiured while data injection as{e}")
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    try:
        di = DataInjection()
        di.initiate_data_injection()
    except Exception as e:
        logging.error(f"Failed to inject data: {e}")




