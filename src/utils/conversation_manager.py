import logging
from typing import Dict, List, Any, Optional
from collections import deque
from src.config_loader import ConfigLoader
from src.utils.utils import cost_per_token
logger = logging.getLogger(__name__)


class ConversationState:
    """
    Manages the state of a single conversation, including history and costs.
    """

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.history: List[Dict[str, Any]] = []
        self.last_message_cost: float = 0.0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        config_loader = ConfigLoader()
        self.cost_per_token: float = float(config_loader.get_config("DEFAULT_COST_PER_TOKEN"))  # Sample cost per token for gpt-3.5-turbo
        self.max_history_length: int = int(config_loader.get_config("MAX_HISTORY_LENGTH")) if config_loader.get_config("MAX_HISTORY_LENGTH") else 10        
        self.config_loader: ConfigLoader = config_loader


    def append_message(self, content: str, message_type: str):
        """
        Appends a message to the conversation history with the message type.
        If the history exceeds the maximum length, removes the oldest messages.
        """
        self.history.append({"content": content, "type": message_type})        
        if len(self.history) > self.max_history_length:
            self.history.pop(0)  # Remove the oldest message
        logger.info(
            f"Conversation {self.conversation_id}: Appended message of type {message_type}"
        )
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """
        Returns the last n messages from the conversation history.
        If n is greater than the history length, returns the entire history.
        """
        if n >= len(self.history):
            return self.history
        return self.history[-n:]
        )

    def clear_history(self):
        """Clears the conversation history and resets the usage cost."""
        self.history = []
        self.total_cost = 0.0
        self.total_tokens = 0
        self.last_message_cost = 0.0
        logger.info(f"Conversation {self.conversation_id}: History cleared.")

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the conversation history."""
        return self.history

    def get_last_message_cost(self) -> float:
        """Returns the cost of the last message."""
        return self.last_message_cost

    def get_conversation_id(self) -> str:
        """Returns the conversation id."""
        return self.conversation_id

    def get_total_tokens(self) -> int:
        """Returns the total number of tokens used in the conversation."""
        return self.total_tokens

    def get_total_cost(self) -> float:
        """Returns the total cost of the conversation."""
        return self.total_cost

    def calculate_usage_cost(self, user_query: str, knowledge_result: str, sql_result: str, code_result:str):
        """Calculates the total usage cost based on content length."""        
        content = user_query + knowledge_result + sql_result + code_result
        
        num_tokens = len(content.split())
        model_type = self.config_loader.get_config("MODEL_TYPE")
        self.last_message_cost = cost_per_token(model_type,num_tokens)        
        self.total_tokens += num_tokens
        self.total_cost += self.last_message_cost
        logger.info(
            f"Conversation {self.conversation_id}: Estimated {num_tokens} tokens, updated cost to ${self.last_message_cost:.6f}, updated total cost to {self.total_cost:.6f}, updated total tokens to {self.total_tokens}"
        )


class ConversationManager:
    def __init__(self):
        self.conversation_states: Dict[str, ConversationState] = {}

    def get_or_create_conversation_state(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = ConversationState(
                conversation_id
            )
            logger.info(f"Created new conversation state for ID: {conversation_id}")
        return self.conversation_states[conversation_id]

    def clear_conversation_state(self, conversation_id: str):
        if conversation_id in self.conversation_states:
            self.conversation_states[conversation_id].clear_history()

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Returns the conversation history for a given conversation ID.
        """
        if conversation_id in self.conversation_states:
            return self.conversation_states[conversation_id].get_history()
        else:
            return []
