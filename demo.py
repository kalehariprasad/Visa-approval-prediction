import os,sys
from src.logger import logging
from src.exception import CustomException
from src.constants import SCHEMA_FILE_PATH

try:
    print(SCHEMA_FILE_PATH)
 

except Exception as e:
    logging.info('Failed to complete the addition : %s', e)
    raise CustomException(e, sys)