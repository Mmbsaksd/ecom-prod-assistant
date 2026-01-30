import os
import json
import sys
import asyncio
from dotenv import load_dotenv
from prod_assistant.utils.config_loader import load_config

from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import AzureOpenAIEmbeddings


from prod_assistant.logger import GLOBAL_LOGGER as log
from prod_assistant.exception.custom_exception import ProductAssistantException

class ApiKeyManager:
    PROVIDER_KEYS = {
            "groq": ["GROQ_API_KEY"],
            "google": ["GOOGLE_API_KEY"],
            "azure": [
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_DEPLOYMENT_NAME",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
                "AZURE_OPENAI_API_VERSION",
            ],
        }

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")
        self.provider = os.getenv("LLM_PROVIDER", "azure").lower()

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed,dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))
        required_keys = self.PROVIDER_KEYS.get(self.provider)

        if not required_keys:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")
        
        for key in required_keys:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")
            
        missing = [k for k in required_keys if not self.api_keys.get(k)]
        if missing:
            if self.provider == "azure":
                log.error("Azure API keys missing", missing_keys=missing)
                raise ProductAssistantException("Azure API keys missing", sys)
            else:
                log.warning("Non-primary provider keys missing", missing_keys=missing)


        log.info("API keys loaded", provider=self.provider, available_keys=list(self.api_keys.keys()))
    def get(self, key:str)->str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val

class ModelLoader:
    def __init__(self):
        if os.getenv("ENV","local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys = list(self.config.keys()))


    def load_embeddings(self):
        try:
            log.info("Loading Azure OpenAI embedding model")

            # Ensure event loop exists (Windows / async safety)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            return AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                api_version=self.api_key_mgr.get("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=self.api_key_mgr.get("AZURE_OPENAI_ENDPOINT"),
                api_key=self.api_key_mgr.get("AZURE_OPENAI_API_KEY"),
            )

        except Exception as e:
            log.error("Error loading Azure embedding model", error=str(e))
            raise ProductAssistantException("Failed to load Azure embedding model", sys)




    def load_llm(self):
        llm_block = self.config['llm']
        provider_key = os.getenv("LLM_PROVIDER", "azure").lower()

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider = provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature",0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider = provider, model = model_name)

        if provider == "azure":
            
            return AzureChatOpenAI(
                azure_deployment=self.api_key_mgr.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version=self.api_key_mgr.get("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=self.api_key_mgr.get("AZURE_OPENAI_ENDPOINT"),
                api_key=self.api_key_mgr.get("AZURE_OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
            )

        
        elif provider=="google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key = self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature = temperature,
                max_output_tokens = max_tokens
            )
        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                temperature=temperature,
            )
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")

if __name__=="__main__":
    loader = ModelLoader()

    #Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    #Test LLM
    llm = loader.load_llm()
    print(f"LLM loaded: {llm}")
    result = llm.invoke("Hello how are you")
    print(f"LLM Result: {result.content}")