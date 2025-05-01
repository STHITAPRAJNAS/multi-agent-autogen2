from langchain_community.vectorstores.pgvector import PGVector
from langchain.embeddings import VertexAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from src.config_loader import ConfigLoader
import os
import logging
from autogen import AssistantAgent
from genkit import get_project_id
from typing import Dict, Any


config_loader = ConfigLoader()

def get_vector_store(collection_name, connection_string):
    """
    Get the vector store.
    """
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001", project=get_project_id())
    vector_store = PGVector(
        collection_name=collection_name,
        connection_string=connection_string,
        embedding_function=embeddings,
    )
    return vector_store

def knowledge_retriever_helper(query: str, collection_name:str, pg_db_url: str) -> str:
    """Helper function for the knowledge retriever agent. Now call rag method from genkit"""
    logging.info(f"Getting knowledge from pgvector with collection_name: {collection_name}")
    try:
        vector_store = get_vector_store(collection_name, pg_db_url)
        # Get the vector store
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        llm = VertexAI(model_name="gemini-pro", project=get_project_id(), temperature=0)
        # Create the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        # Run the chain
        result = qa_chain(query)

        return result["result"]
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {e}"


def create_knowledge_retriever_agent(llm_config: Dict[str, Any], pg_db_url, model_type) -> AssistantAgent:
    """Create and configure the knowledge retriever agent."""
    prompt_loader = ConfigLoader()
    knowledge_retriever_prompt = prompt_loader.get_prompt("knowledge_retriever")

    knowledge_retriever = AssistantAgent(
        name="knowledge_retriever",
        system_message=knowledge_retriever_prompt,
        llm_config=llm_config,
    )
    return knowledge_retriever
