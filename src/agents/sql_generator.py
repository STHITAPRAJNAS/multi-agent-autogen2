from typing import List, Dict, Any
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from src.config_loader import logger
from src.utils.prompt_loader import PromptLoader
import json


def sql_generator_helper(agent: ConversableAgent, user_query: str) -> str:
    """
    Helper function for the SQL generator agent.
    This function generates a single SQL query based on the user's query and the conversation history.
    """
    user_proxy = UserProxyAgent(
        name="sql_user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
        system_message="""You are the user proxy. Your task is to get the sql query from the user.
        You will send the query to the sql agent and return the sql query as is.
        """
    )
    user_proxy.initiate_chat(agent, message=f"User query: {user_query}")
    return user_proxy.last_message()["content"]


def create_sql_generator_agent(llm_config: Dict[str, Any], cache_seed_path: str) -> AssistantAgent:
    """Create and configure the SQL generator agent."""
    try:
        prompt_loader = PromptLoader()
        sql_generator_prompt = prompt_loader.get_prompt("sql_generator")
    sql_generator = AssistantAgent(
        name="sql_generator",
        system_message=sql_generator_prompt,
        max_consecutive_auto_reply=0,
        llm_config=llm_config
    )
    except Exception as e:
      logger.error(f"Error creating sql agent : {e}")
      raise e
    return sql_generator