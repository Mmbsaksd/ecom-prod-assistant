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
        pass
    def load_env_variable(self):
        pass
    def get_csv_path(self):
        pass
    def load_csv(self):
        pass
    def transform_data(self):
        pass

    #
    