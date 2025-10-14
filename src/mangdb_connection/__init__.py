from src.constants import DB_NAME,  COLLECTION_NAME
from src.logger import logging
from src.exception import CustomException
import pymongo
import certifi
import sys
import os


ca = certifi.where()


class MongoDBClient:
    """
    Class: MongoDBClient
    Description: MongoDB and provides access to the specified database.
    """
    client = None

    def __init__(self, database_name=DB_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                # CONNECTION_URL = os.getenv("MONGO_CONNECTION_URL")
                CONNECTION_URL = ("mongodb+srv://hariprasad9693:bSw2MFORkyusvdN8@cluster0.jnoviod.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
                if not CONNECTION_URL:
                    raise Exception("MongoDB connection string"
                                    "(MONGO_CONNECTION_URL) "
                                    "not set in environment.")
                MongoDBClient.client = pymongo.MongoClient(
                    CONNECTION_URL, tlsCAFile=ca
                )

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.collection = self.database[COLLECTION_NAME]

            logging.info(
                f"MongoDB connection successful to database: {database_name}"
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        mongo_client = MongoDBClient()
        collection = mongo_client.collection

        # Fetch all documents
        records = list(collection.find())

        print(f"Total records fetched: {len(records)}")
        if records:
            print("Sample document:", records[0])

    except CustomException as ce:
        logging.error(f"Custom exception occurred: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
