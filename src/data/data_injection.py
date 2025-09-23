import sys
import os
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

    def initiate_data_injection(self):
        try:
            # Debug: Print MONGO_URI to ensure it's loaded
            mongo_uri = os.getenv("MONGO_URI")
            print(f"[DEBUG] MONGO_URI: {mongo_uri[:8]}********")
            print("[DEBUG] Connecting to MongoDB...")

            db = self.mangodb
            data = self.data_handler
            artifacts = self.artifact

            # Fetch data from MongoDB
            df = pd.DataFrame(list(db.collection.find()))

            print(f"[DEBUG] Number of records fetched: {len(df)}")

            if df.empty:
                print("❌ No data fetched from MongoDB. Exiting.")
                sys.exit(1)

            # Drop MongoDB ID column if it exists
            if '_id' in df.columns:
                df.drop(columns='_id', inplace=True)

            # Split data
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )

            # Save data
            data.save_data(df, artifacts.raw_file_path)
            data.save_data(train_df, artifacts.train_file_path)
            data.save_data(test_df, artifacts.test_file_path)

            # Final check for files
            if not os.path.exists(artifacts.train_file_path):
                print("❌ train.csv was not created.")
                sys.exit(1)

            if not os.path.exists(artifacts.test_file_path):
                print("❌ test.csv was not created.")
                sys.exit(1)

            print("✅ Data injection completed successfully.")
            logging.info("✅ Data injection completed successfully.")

        except Exception as e:
            logging.error(f"❌ Error occurred during data injection: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        di = DataInjection()
        di.initiate_data_injection()
    except Exception as e:
        logging.error(f"❌ Failed to inject data: {e}")
        sys.exit(1)
