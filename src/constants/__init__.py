import os
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

