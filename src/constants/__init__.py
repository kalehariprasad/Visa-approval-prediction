import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# data injection related constatnts

connection_string = os.getenv('connection_string')
DB_NAME = "VISA"
COLLECTION_NAME = "visa_data"
CONNECTION_URL = connection_string


ARTIFACTS_DIRECTORY = "data"
RAW_DATA_DIRECTORY = "raw"
RAW_DATA_FILE = "Easyvisa.csv"
RAW_TRAIN_FILE= "train.csv"
RAW_TEST_FILE= "test.csv"

# data validation related constant

DATA_VALIDATION_FOLDER="data_validation"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


# date transfromation realetd constants

DATA_PREPROCESSING_FOLDER = "preprocessed" 
PREPROCESSED_TRAIN = "train.csv"
PREPROCESSED_TEST = "test.csv"
PREPROCESSOR_OBJECT = "preprocessor.pkl"


#features 
FEATURE_FOLDER = 'festures'
TRAIN_X = 'train_x.npy'
TRAIN_Y = 'train_y.npy'
TEST_X ='test_x.npy'
TEST_Y = 'test_y.npy'
