from mcp.server.fastmcp import FastMCP
from retriever.retrieval import Retriever
from langchain_community.tools import DuckDuckGoSearchRun

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
        context = format_doc(docs)

        if not context.strip():
            return "No local result found"
        return context
    except Exception as e:
        return f"Error retriving product info: {str(e)}"
    
@mcp.tool()
async def web_search(query:str)->str:
    try:
        pass
    except Exception as e:
        return f"Error during web search: {str(e)}"

if __name__== "__main__":
    mcp.run(transport="streamable-http")    