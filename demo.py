import os,sys
from src.logger import logging
from src.exception import CustomException

try:
    a = 4
    b= "5"
    c = a+b
    logging.info('addition successfull')

except Exception as e:
    logging.info('Failed to complete the addition : %s', e)
    raise CustomException(e, sys)