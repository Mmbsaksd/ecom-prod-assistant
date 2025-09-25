from langchain.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatMessagePromptTemplate

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy

retriever_obj = Retriever()
model_loader = ModelLoader()

def format_docs(docs)-> str:
    if not docs:
        return "No relevants documents found"
    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title','NA')}\n"
            f"Price: {meta.get('price','N/A')}\n"
            f"Rating: {meta.get('rating','N/A')}\n"
            f"Reviews: {meta.get(d.page_content.strip())}"

        )
        formatted_chunks.append(formatted)
    return "\n\n--\n\n".join(formatted_chunks)

def build_chain(query):
    retriever = retriever_obj.load_retriever()
    retrieved_docs = retriever.invoke(query)

    retrived_context = [format_docs(retrieved_docs)]

    llm = model_loader.load_llm()
    prompt = ChatMessagePromptTemplate.from_template(
        PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
    )
    chain = (
        {"context":retriever | format_docs,"question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retrived_context

def invoke_chain(query:str, debug: bool=False):
    chain, retrieved_contexts = build_chain(query)
    if debug:
        docs = retriever_obj.load_retriever().invoke(query)
        print("\nRetrieved Documents: ")
        print(format_docs(query))
        print("\n---\n")
    response = chain.invoke(query)

    return retrieved_contexts, response

if __name__ == "__main__":
    user_query = "Can you suggest good budget iphone"
    retrieved_context, response = invoke_chain(user_query)
    
    context_score = evaluate_context_precision(user_query,response,retrieved_context)
    relevancy_score = evaluate_response_relevancy(user_query,response,retrieved_context)

    print("\n--Evaluation metrics--\n")
    print("Context Precision Score: ", context_score)
    print("Response Relavency Score: ", relevancy_score)


