from typing import Dict, Any
import autogen
from src.utils.prompt_loader import PromptLoader
import logging
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import AssistantAgent

def knowledge_retriever_helper(user_proxy, query: str) -> str:
    """Helper function for the knowledge retriever agent."""
    try:        
        docs = user_proxy.retrieve_docs(query)        
        content = ""
        for doc in docs:
            content = content + doc.content

        return content
    except Exception as e:
        logging.error(f"Error during knowledge retrieval: {e}")
        return "Error during knowledge retrieval."


def create_knowledge_retriever_agent(llm_config: Dict[str, Any], cache_seed_path: str) -> AssistantAgent:
    """Create and configure the knowledge retriever agent."""
    prompt_loader = PromptLoader()
    knowledge_retriever_prompt = prompt_loader.get_prompt("knowledge_retriever")

    knowledge_retriever = AssistantAgent(
        name="knowledge_retriever",
        system_message=knowledge_retriever_prompt,
        llm_config=llm_config,
    )

    return knowledge_retriever
