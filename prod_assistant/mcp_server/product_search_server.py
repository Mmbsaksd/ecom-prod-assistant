from mcp.server.fastmcp import FastMCP
from prod_assistant.retriever.retrieval import Retriever
from langchain_community.tools import DuckDuckGoSearchRun
import re
mcp = FastMCP("hybrid_search")

retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

duckduckgo = DuckDuckGoSearchRun()


def format_doc(docs)-> str:
    if not docs:
        return ""
    formatted_chunk = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title','N/A')}\n"
            f"Price: {meta.get('price','N/A')}\n"
            f"Rating: {meta.get('rating','N/A')}\n"
            f"Review: \n{d.page_content.strip()}"
        )
        formatted_chunk.append(formatted)
        
    return "\n\n---\n\n".join(formatted_chunk)



@mcp.tool()
async def get_product_info(query:str)-> str:
    try:
        docs = retriever.invoke(query)

        filtered_docs = [
            d for d in docs 
            if any(word in d.metadata.get("product_title","").lower()
                   for word in query.lower().split())
        ]
        if not filtered_docs:
            return "No exact result found"
        context = format_doc(filtered_docs)
        return context
    except Exception as e:
        return f"Error retriving product info: {str(e)}"
    
@mcp.tool()
async def web_search(query:str)->str:
    try:
        return duckduckgo.run(query)
    except Exception as e:
        return f"Error during web search: {str(e)}"

if __name__== "__main__":
    mcp.run(transport="stdio")    