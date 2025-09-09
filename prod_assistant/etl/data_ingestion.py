import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore

from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config

class DataIngestion:
    def __init__(self):
        print("Initializing DataIngestion pipeline...")
        self.model_loader = ModelLoader()
        self._load_env_variable()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config = load_config()

    def _load_env_variable(self):
        load_dotenv()
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN","ASTRA_DB_KEYSPACE"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing enviroment variables: {missing_vars}")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
        
    def _get_csv_path(self):
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir,'data','product_reviews.csv')
        if not os.path.exists(csv_path):
            raise FileExistsError(f"CSV file not found at: {csv_path}")
        
    def _load_csv(self):
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_id','product_title','rating','total_reviews','price','top_reviews'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")


    def transform_data(self):
        pass
    def store_in_vector(self):
        pass
    def run_pipeline(self):
        pass