from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from prod_assistant.evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy

class AgenticRAG:
    class AgentState(TypedDict):
        message: Annotated[Sequence[BaseMessage],add_messages]

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()

    def _ai_assistant(self, state:AgentState):
        print("---CALL ASSISTANT---")
        messages = state["messages"]
        last_message = messages[-1].content

        if any(word in last_message.lower() for word in ["price","review","product"]):
            return {"messages":[HumanMessage(content="TOOL: retriever")]}
        
        prompt = ChatPromptTemplate.from_template(
            "You are helpful assistant. Answer user directly. \n\nQuestion: {question}\nAnswer:"
        )
        chain = prompt| self.llm| StrOutputParser()
        response = chain.invoke({"question": last_message})
        return {"messages": [HumanMessage(content=response)]}

    


    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)