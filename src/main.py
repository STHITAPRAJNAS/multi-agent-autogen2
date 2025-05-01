import logging
from typing import Dict, Any, List
from autogen import config_list_from_models,config_list_bedrock
from src.config_loader import ConfigLoader
from src.utils.conversation_manager import ConversationState
from src.agents.knowledge_retriever import create_knowledge_retriever_agent, knowledge_retriever_helper
from src.agents.sql_generator import create_sql_generator_agent, sql_generator_helper
from src.agents.code_generator import create_code_generator_agent, code_generator_helper
from src.agents.user_proxy import create_user_proxy_agent

config_loader = ConfigLoader()

def create_agents(config_list: List[Dict], bedrock_config: Dict[str, str]) -> Dict[str, Any]:
    """Creates and configures the multi-agent system."""
    cache_seed_path = config_loader.get_config("CACHE_SEED_PATH")
    model_type = config_loader.get_config("MODEL_TYPE")
    
    if model_type == "bedrock":
        llm_config = {
            "model": bedrock_config.get("ANTHROPIC_MODEL_ID"),
            "region_name": bedrock_config.get("BEDROCK_REGION"),
            "config_list": config_list,
        }
    elif model_type in ["anthropic", "gemini"]:
        llm_config = {
            "config_list": config_list,
        }
    else:
      llm_config = {"config_list": config_list}


    try:
      
        user_proxy_llm_config = llm_config
        user_proxy = create_user_proxy_agent(user_proxy_llm_config, cache_seed_path)

        knowledge_retriever_llm_config = llm_config_bedrock if model_type == "bedrock" else llm_config
        knowledge_retriever = create_knowledge_retriever_agent(
            knowledge_retriever_llm_config,
            cache_seed_path
        )

        sql_generator_llm_config = llm_config
        sql_generator = create_sql_generator_agent(
            sql_generator_llm_config, cache_seed_path
        )

        code_generator_llm_config = llm_config
        code_generator = create_code_generator_agent(            
            code_generator_llm_config, cache_seed_path
        )


        return {
            "user_proxy": user_proxy,
            "knowledge_retriever": knowledge_retriever,
            "sql_generator": sql_generator,
            "code_generator": code_generator,
        }
    except Exception as e:
        logging.error(f"Error creating agents: {e}")
        return {}


def run_task(agents: Dict[str, Any], user_query: str, conversation_state: ConversationState) -> Dict[str, Any]:
    """Runs the three tasks using the agents and returns the results."""
    if not agents:
        logging.error("Agents not initialized.")
        return {}

    user_proxy = agents["user_proxy"]
    knowledge_retriever = agents["knowledge_retriever"]
    sql_generator = agents["sql_generator"]
    code_generator = agents["code_generator"]

    try:
        # Knowledge retrieval
        knowledge_query = f"Retrieve knowledge about: {user_query}"        
        user_proxy.initiate_chat(
            knowledge_retriever, message=knowledge_query
        )
        knowledge_result = knowledge_retriever_helper(knowledge_retriever, user_query)
        conversation_state.append_message(user_query, knowledge_result, "knowledge_retrieval")

        # SQL query generation
        sql_query_task = (
            f"Based on this user query: {user_query} , generate sql query, also the conversation history is: {conversation_state.get_history()}"
        )

        user_proxy.initiate_chat(
            sql_generator, message=f"Generate a SQL query for: {sql_query_task}",
        )
        sql_query_result = sql_generator_helper(sql_generator, sql_query_task)
        conversation_state.append_message(user_query, sql_query_result, "sql_generation")


        # Code generation
        code_generation_task = (
            f"Based on this user query: {user_query} , generate python code, also the conversation history is: {conversation_state.get_history()}"
        )
        user_proxy.initiate_chat(
            code_generator, message=f"Generate Python code for: {code_generation_task}",
        )
        conversation_state.append_message(user_query, code_result, "code_generation")

        conversation_state.calculate_usage_cost(user_query, knowledge_result, sql_query_result, code_result)

        return {
            "knowledge_result": knowledge_result,
            "sql_query_result": sql_query_result,
            "code_result": code_result,
        }
    except Exception as e:
        logging.error(f"Error running task: {e}")
        return {
            "knowledge_result": "Error during knowledge retrieval.",
            "sql_query_result": "Error during SQL query generation.",
            "code_result": "Error during code generation.",
        }