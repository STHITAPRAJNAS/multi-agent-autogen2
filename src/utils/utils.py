import logging.config
import os
from typing import List, Dict, Optional
from src.config_loader import ConfigLoader

CACHE_SEED_PATH = "./cache_seed"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "default",
            "filename": "app.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)


def config_list_from_models(model_ids: List[str], api_key: Optional[str] = None, llm_config:Optional[Dict] = None ) -> List[Dict]:
    """
    Creates a configuration list for Autogen from a list of model IDs, supporting Anthropic, Gemini, and Bedrock.

    Args:
        model_ids: List of model IDs to create configuration for.
        api_key: API key for models like Anthropic and Gemini.

    Returns:
        A list of dictionaries, where each dictionary is a configuration for a model.    
    """
    config_list = []
    config_loader = ConfigLoader()
    for model_id in model_ids:
        config: Dict[str, str | Dict] = {}
        if model_id.startswith("anthropic"):
            config["model"] = model_id
            api_key = config_loader.get_config("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(f"API key for Anthropic model {model_id} is not set.")
            config["api_key"] = api_key
            config["api_type"] = "anthropic"
        elif model_id.startswith("google_gemini_pro"):
            config["model"] = model_id
            api_key = config_loader.get_config("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(f"API key for Gemini model {model_id} is not set.")
            config["api_key"] = api_key
            config["api_type"] = "gemini"
        else:
            # For Bedrock, continue with the existing logic
            config = {
                "model": model_id,
                "region_name": config_loader.get_config("BEDROCK_REGION"),
            }

        config_list.append(config)
    return config_list

def cost_per_token(model_type: str, num_tokens: int) -> float:
    """
    Calculates the cost per token based on the model type.

    Args:
        model_type: The type of the model ('openai', 'gemini', 'anthropic').
        num_tokens: The number of tokens.

    Returns:
        The cost per token.

    Raises:
        ValueError: If the model type is unknown.
    """
    if model_type == "openai":
        return num_tokens * 0.000002
    elif model_type == "gemini":
        return num_tokens * 0.00000025
    elif model_type == "anthropic":
        return num_tokens * 0.000008
    else:
        raise ValueError(f"Unknown model type: {model_type}")