from typing import List, Dict, Any, Optional
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from src.utils.conversation_manager import ConversationState
from src.utils.prompt_loader import PromptLoader
import json


def code_generator_helper(    
    agent: ConversableAgent, user_query: str, conversation_state: ConversationState
) -> str:
    """
    Helper function for the code generator agent.
    This function generates Python code based on the user's query and conversation history.
    """
    conversation_history = conversation_state.get_history()
    system_message = f"""
        You are a code generator assistant. Your task is to generate a single, executable Python code snippet based on the user's request and the provided conversation history.
        Here is the conversation history: {json.dumps(conversation_history)}.
        The user query is : {user_query}
        - Generate only the code in a code block.
        - Do not provide explanations or comments outside the code block.
        - Ensure that the code is executable and complete.
    """
    user_proxy = UserProxyAgent(
        name="code_user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,  # Ensure only one reply is expected
        code_execution_config={"work_dir": "coding"},
        system_message=system_message,
    )
    user_proxy.initiate_chat(agent, message=user_query)
    return user_proxy.last_message()["content"]


def create_code_generator_agent(llm_config: Dict, cache_seed_path: str) -> AssistantAgent:
    """Create and configure the code generator agent."""
    prompt_loader = PromptLoader()
    code_generator_prompt = prompt_loader.get_prompt("code_generator")
    code_generator = AssistantAgent(
        max_consecutive_auto_reply=0,
        name="code_generator",        
        llm_config=llm_config,
        system_message=code_generator_prompt,
    )
    return code_generator