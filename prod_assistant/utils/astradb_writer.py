import os
import datetime
from typing import Optional
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config
from prod_assistant.logger import GLOBAL_LOGGER as log


class AstraWriter:
    def __init__(self):
        self.enabled = False
        try:
            # Load config and env
            self.config = load_config()
            required = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
            missing = [v for v in required if os.getenv(v) is None]
            if missing:
                log.info("AstraWriter disabled - missing env vars", missing=missing)
                return

            self.model_loader = ModelLoader()
            collection_name = self.config["astra_db"]["collection_name"]

            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
                token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
                namespace=os.getenv("ASTRA_DB_KEYSPACE"),
            )
            self.enabled = True
            log.info("AstraWriter initialized", collection=collection_name)
        except Exception as e:
            log.warning("Failed to initialize AstraWriter, continuing without persistence", error=str(e))
            self.enabled = False

    def _is_duplicate(self, question: str, final_answer: str) -> bool:
        try:
            # conservative duplicate check: search by question and compare stored answer
            results = self.vstore.similarity_search(question, k=5)
            for r in results:
                stored_q = r.metadata.get("user_question")
                stored_ans = r.page_content
                if stored_q and stored_ans and stored_q.strip() == question.strip() and stored_ans.strip() == final_answer.strip():
                    return True
            return False
        except Exception as e:
            log.warning("AstraWriter duplicate check failed, assuming not duplicate", error=str(e))
            return False

    def save_interaction(self, question: str, retrieved_context: Optional[str], final_answer: str, thread_id: Optional[str] = None) -> bool:
        """Save a single Q/A interaction to AstraDB. Returns True if written, False otherwise."""
        if not self.enabled:
            log.info("AstraWriter disabled - skipping save")
            return False

        try:
            if self._is_duplicate(question, final_answer):
                log.info("Duplicate interaction found - skipping insert", question=question)
                return False

            metadata = {
                "user_question": question,
                "retrieved_context": retrieved_context or "",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "source": "agentic_rag",
            }
            if thread_id:
                metadata["thread_id"] = thread_id

            doc = Document(page_content=final_answer, metadata=metadata)
            inserted = self.vstore.add_documents([doc])
            log.info("Saved interaction to AstraDB", inserted_count=len(inserted))
            return True
        except Exception as e:
            log.warning("Failed to save interaction to AstraDB", error=str(e))
            return False
