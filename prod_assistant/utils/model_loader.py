import os
import json
import sys
from dotenv import load_dotenv
from prod_assistant.utils.config_loader import load_config

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from prod_assistant.logger import GLOBAL_LOGGER as log
from prod_assistant.exception.custom_exception import ProductAssistantExeption

class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed,dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS screat")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        for key in self.REQUIRED_KEYS:
            if not self.api_keys.het(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_key=missing)
            raise ProductAssistantExeption("Missing API keys", sys)
        log.info("API keys loaded", keys = {k: v[:6]+"..." for k,v in self.api_keys.items()})

    def get(self, key:str)->str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val
    
    

