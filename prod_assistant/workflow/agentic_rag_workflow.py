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

    def format_docs(self, docs)->str:
        if not docs:
            return "No relevant documents found"
        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title:{meta.get('product_title','N/A')}\n"
                f"Price:{meta.get('price','N/A')}\n"
                f"Rating:{meta.get('rating','N/A')}\n"
                f"Review:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)
        return "\n\n---\n\n".join(formatted_chunks)

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
    
    def _vector_retriever(self, state: AgentState):
        print("---RETRIEVER---")
        query = state["messages"][-1].content
        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(query)
        context = self.format_docs(docs)
        return {"messages":[HumanMessage(content=context)]}
    
    def _grade_documnets(self, state:AgentState)->Literal["generator","rewriter"]:
        print("---GRADER---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grager. Question:{question}\nDocs:{docs}\n
            Are relevant to the question? Answer yes or no""",
            input_variables=["question","docs"]
        )
        chain = prompt |self.llm | StrOutputParser()
        score = chain.invoke({"question":question,"docs":docs})
        return "generator" if "yes" in score.lower() else "rewriter"

    def _generate(self, state: AgentState):
        print("---GENERATE---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_messages(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context":docs, "question":question})
        return {"messages":[HumanMessage(content=response)]}
    
    def _rewrite(self, state:AgentState):
        print("--- REWRITE --")
        question = state["messages"][0].content
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be more clear: {question}")]
        )
        return {"messages":[HumanMessage(content=new_q.content)]}


    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewrite", self._rewrite)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END
            {"Retriever":"Retriever",END:END}
        )
        workflow.add_conditional_edges(
            workflow.add_conditional_edges(
                "Rereiever",
                self._grade_documnets,
                {"generator":"Generator","rewriter":"Rewriter"}
            )
        )
        workflow.add_edge("Generator",END)
        workflow.add_edge("Rewriter","Assistant")
        return workflow
