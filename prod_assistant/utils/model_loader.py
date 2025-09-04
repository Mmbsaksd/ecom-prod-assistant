import os
import json
import sys
from dotenv import load_dotenv
from prod_assistant.utils.config_loader import load_config

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from prod_assistant.logger import GLOBAL_LOGGER as log

