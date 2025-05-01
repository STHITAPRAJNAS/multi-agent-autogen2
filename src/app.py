# src/app.py
import logging.config
from typing import AsyncIterator, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from autogen import config_list_from_json, config_list_bedrock, config_list_from_models
from src.config_loader import ConfigLoader

from src.main import run_task, create_agents
from src.utils.conversation_manager import ConversationManager

# Configure logging
from src.utils.utils import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# Load configuration
config_loader = ConfigLoader()
app_config = config_loader.get_app_config()
llm_config = config_loader.get_llm_config()
bedrock_config = config_loader.get_bedrock_config()

# Global variable to store the agents
agents: Dict[str, Any] = {}


# Load LLM configuration
model_type = app_config.get("MODEL_TYPE")
if model_type not in ["openai", "bedrock"]:
    logger.error(f"Invalid model type: {model_type}")
    raise ValueError(f"Invalid model type: {model_type}")

if model_type == "openai":
    
    try:
        config_list = config_list_from_json(env_or_file=llm_config.get("OAI_CONFIG_LIST"), filter_dict={"model": [llm_config.get("DEFAULT_MODEL")]})
    except Exception as e:
        logger.error(f"Failed to load config list: {e}")
        raise
elif model_type == "bedrock":
    
    try:
        config_list = config_list_bedrock(region_name=bedrock_config.get("BEDROCK_REGION"))
    except Exception as e:
        logger.error(f"Failed to load bedrock config list: {e}")
        raise
elif model_type in ["anthropic","gemini"]:
    try:
        models = [bedrock_config.get("ANTHROPIC_MODEL_ID") if model_type == "anthropic" else bedrock_config.get("GEMINI_MODEL_ID")]
        config_list = config_list_from_models(models)
    except Exception as e:
        logger.error(f"Failed to load bedrock config list: {e}")
        raise
else:
    logger.error(f"Invalid model type: {model_type}")
    raise ValueError(f"Invalid model type: {model_type}")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Context manager for FastAPI lifespan events.
    Initializes and cleans up resources like agents.
    """
    logger.info("Starting up...")
    global agents
    try:
        agents = create_agents(config_list, bedrock_config)  # Initialize agents
        logger.info("Agents initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        raise  # Re-raise the exception to prevent app startup
    yield
    logger.info("Shutting down...")
    agents.clear()
    logger.info("Agents cleared successfully")




conversation_manager = ConversationManager()


class ConversationRequest(BaseModel):
    query: str = Field(..., description="User query")
    conversation_id: str


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(request: ConversationRequest) -> Dict[str, Any]:
    logger.info(f"Received chat request: {request.query}, conversation_id: {request.conversation_id}")
    conversation_state = conversation_manager.get_or_create_conversation_state(request.conversation_id)

    try:
        response = run_task(agents, request.query, conversation_state)        
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/clear_conversation")
async def clear_conversation(conversation_id: str):
    logger.info(f"Clearing conversation state for: {conversation_id}")
    conversation_manager.clear_conversation_state(conversation_id)
    return {"message": f"Conversation {conversation_id} cleared."}

@app.get("/get_conversation_history")
async def get_conversation_history(conversation_id: str) -> List[Dict[str, str]]:
    """Returns the conversation history for a given conversation ID."""
    logger.info(f"Getting conversation history for: {conversation_id}")
    history = conversation_manager.get_conversation_history(conversation_id)
    return history
