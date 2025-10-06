import sys
import re
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
from langchain_mcp_adapters.client import MultiServerMCPClient

class AgenticRAG:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage],add_messages]

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()

        self.mcp_client = MultiServerMCPClient(
            {
                "product_retriever":{
                    "command":sys.executable,
                    "args": ["-m", "prod_assistant.mcp_server.product_search_server"],
                    "transport":"stdio"
                }
                

            }
        )

        self.mcp_tools = asyncio.run(self.mcp_client.get_tools())
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

        
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
        last_message = messages[-1].content if messages else ""

        if any(word in last_message.lower() for word in ["price","review","product"]):
            return {"messages":[HumanMessage(content=f"TOOL: retriever||{last_message}")]}
        
        prompt = ChatPromptTemplate.from_template(
            """You are a product assistant. Only return the direct, final answer to the user's question, without explanations or alternative suggestions.

            User Question: {question}
            Context: {context}

            Final Answer:
            """
        )
        chain = prompt| self.llm| StrOutputParser()
        safe_context = ""
        try:
            response = chain.invoke({"question": last_message, "context": safe_context})
        except Exception as e:
            print("Error invoking assistant chain:", e)
            response = "Sorry — I couldn't generate an answer right now."
        return {"messages": [HumanMessage(content=response)]}
    
    def _vector_retriever(self, state: AgentState):
        print("---RETRIEVER(MCP)---")
        raw = state["messages"][-1].content

        if raw.startswith("TOOL: retriever||"):
            query = raw.split("||", 1)[1].strip()
        else:
            query = raw

        product_tool = next(t for t in self.mcp_tools if t.name=="get_product_info")
        web_tool = next(t for t in self.mcp_tools if t.name=="web_search")

        result = asyncio.run(product_tool.ainvoke({"query":query}))

        if not result or not any(keyword in result.lower() for keyword in [
        "price", "product", "model", "details", "specification", "features", "buy", "cost"
        ]):
            print("---No database result, running web search---")
            web_result = asyncio.run(web_tool.ainvoke({"query":query}))
            context = web_result
        else:
            context = result

        return {"messages":[HumanMessage(content=context)]}
    
    def _grade_documents(self, state:AgentState)->Literal["generator","rewriter"]:
        print("---GRADER---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question:{question}\nDocs:{docs}\n
            Are docs relevant to the question? Answer yes or no""",
            input_variables=["question","docs"]
        )
        chain = prompt |self.llm | StrOutputParser()
        score = chain.invoke({"question":question,"docs":docs})

        
        if "yes" in score.lower() or any(keyword in docs.lower() for keyword in [
        "price", "product", "model", "specification", "details", "features", "buy", "cost"
    ]):
            return "generator"
        return "rewriter"

    def clean_response(self,text, max_chars=None):
        """
        Clean up model output by removing markdown symbols, stars, and extra spacing.
        """
        if not text:
            return ""
        text = re.sub(r'^[\*\-\+\s]*', '', text, flags=re.M)
        text = re.sub(r'\n+', ' ', text).strip()
        text = text.replace('*', '').replace('`', '')
        if max_chars is not None and len(text)>max_chars:
            text = text[:max_chars].rsplit(' ', 1)[0] + '...'
        return text
 
    def _generate(self, state: AgentState):
        print("---GENERATE---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context":docs, "question":question})
        safe_response = self.clean_response(response, max_chars=250)
        return {"messages":[HumanMessage(content=safe_response)]}
    

    def _rewrite(self, state: AgentState):
        print("--- REWRITE --")
        question = state["messages"][0].content

        rewrite_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that rewrites user queries 
            to make them more specific and clear for product searches.

            Original query: {question}

            Rewrite the query in one line. 
            Only return the rewritten query — no explanations, 
            no examples, and no lists.
            """
        )

        chain = rewrite_prompt | self.llm | StrOutputParser()
        rewritten_query = chain.invoke({"question": question})
        cleaned_q = self.clean_response(rewritten_query, max_chars=200)
        cleaned_q = cleaned_q.split("\n")[0].strip()  # keep only first line

        return {"messages": [HumanMessage(content=f"TOOL: retriever||{cleaned_q}")] }


    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever":"Retriever",END:END},
        )

        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator":"Generator","rewriter":"Rewriter"}
        )

        workflow.add_edge("Generator",END)
        workflow.add_edge("Rewriter","Retriever")
        return workflow
    
    def run(self, query:str, thread_id: str="default_thread")->str:
        result = self.app.invoke({"messages":[HumanMessage(content=query)]},
                                 config = {"configurable": {"thread_id": thread_id}})
        return result["messages"][-1].content
    
if __name__=="__main__":
    rag_agent = AgenticRAG()
    answer = rag_agent.run("What is the price of iPhone 15?")
    print("\nFinal Answer: \n",answer)
