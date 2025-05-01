import autogen
from autogen import UserProxyAgent
from typing import Dict, List, Any
from src.utils.prompt_loader import PromptLoader
from src.config_loader import ConfigLoader

config_loader = ConfigLoader()


def create_user_proxy_agent(
    llm_config: Dict, cache_seed_path: str = "./cache_seed"
) -> autogen.UserProxyAgent:
    """
    Create and configure the user proxy agent.

    Args:
        llm_config: The configuration list for the LLM.
        cache_seed_path: the path to the cache
    Returns:
        The user proxy agent.
    """
    prompt_loader = PromptLoader()
    user_proxy_prompt = prompt_loader.get_prompt("user_proxy")
    return UserProxyAgent(        
        name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=int(config_loader.get_config("MAX_CONSECUTIVE_AUTO_REPLY")) if config_loader.get_config("MAX_CONSECUTIVE_AUTO_REPLY") else 10, code_execution_config={"work_dir": cache_seed_path}, llm_config=llm_config, system_message=user_proxy_prompt
    )
