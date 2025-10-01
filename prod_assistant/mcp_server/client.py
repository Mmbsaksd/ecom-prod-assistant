import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys

async def main():
    client = MultiServerMCPClient({
        "hybrid_search": {
            "command": sys.executable,  # use same Python from venv
            "args": ["-m", "prod_assistant.mcp_server.product_search_server"],  # run server as module
            "transport": "stdio",
        }
    })

    tools = await client.get_tools()
    print("Available tools: ",[t.name for t in tools])

    retriever_tools = next(t for t in tools if t.name=="get_product_info")
    web_tool = next(t for t in tools if t.name=="web_search")

    query = "what is the price of iphone 16"
    retrieved_result = await retriever_tools.ainvoke({"query":query})
    print("\nRetriever Result: \n", retrieved_result)

    if not retrieved_result.strip() or "No exact result found" in retrieved_result:
        print("\n No results, falling back to web search...\n")
        web_result = await web_tool.ainvoke({"query":query})
        print("Web Search Result:\n",web_result)

if __name__ == "__main__":
    asyncio.run(main())