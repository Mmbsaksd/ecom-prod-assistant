import os
import json
import sys
import asyncio
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

    class ModelLoader:
        def __init__(self):
            if os.getenv("ENV","local").lower != "production":
                load_dotenv()
                log.info("Running in LOCAL mode: .env loaded")
            else:
                log.info("Running in PRODUCTION mode")

            self.api_key_mgr = ApiKeyManager()
            self.config = load_config()
            log.info("YAML config loaded", config_keys = list(self.config.keys()))


        def load_embeddings(self):
            try:
                model_name = self.config["embedding_model"]["model_name"]
                log.info("Loading embedding model", model=model_name)

                try:
                    asyncio.get_running_loop
                except RuntimeError:
                    asyncio.set_event_loop(asyncio.new_event_loop())

                return GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY")
                )
            except Exception as e:
                log.error("Error loading embedding model", error = str(e))
                raise ProductAssistantExeption("Failed to load embedding model", sys)




