"""
Specialized agents for the multi-agent system.
"""

from typing import Dict, Any, List
import autogen
from .vector_store import get_vector_store, load_config, VectorStore
from .llm_provider import get_llm_provider
from langchain.schema import Document
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class KnowledgeRetrieverAgent(autogen.AssistantAgent):
    """Agent for retrieving information from knowledge bases."""
    
    def __init__(self, config: Dict[str, Any], vector_store: VectorStore = None):
        super().__init__(
            name="knowledge_retriever",
            system_message=config['agents']['knowledge_retriever']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        logger.debug("Initializing KnowledgeRetrieverAgent with config: %s", config)
        self.vector_store = vector_store or get_vector_store(config['vector_stores']['knowledge_base'])
        logger.debug("Vector store initialized: %s", self.vector_store)
    
    def retrieve_knowledge(self, query: str) -> List[Document]:
        """Retrieve relevant knowledge from the vector store."""
        logger.debug("Retrieving knowledge for query: %s", query)
        results = self.vector_store.similarity_search(query)
        logger.debug("Retrieved %d documents", len(results))
        return results

class SQLGeneratorAgent(autogen.AssistantAgent):
    """Agent for generating SQL queries."""
    
    def __init__(self, config: Dict[str, Any], vector_store: VectorStore = None):
        super().__init__(
            name="sql_generator",
            system_message=config['agents']['sql_generator']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        logger.debug("Initializing SQLGeneratorAgent with config: %s", config)
        self.vector_store = vector_store or get_vector_store(config['vector_stores']['databricks_schema'])
        logger.debug("Vector store initialized: %s", self.vector_store)
    
    def get_schema_context(self, query: str) -> List[Document]:
        """Retrieve relevant schema information."""
        logger.debug("Retrieving SQL schema for query: %s", query)
        results = self.vector_store.similarity_search(query)
        logger.debug("Retrieved %d documents", len(results))
        return results

class GraphQLGeneratorAgent(autogen.AssistantAgent):
    """Agent for generating GraphQL queries."""
    
    def __init__(self, config: Dict[str, Any], vector_store: VectorStore = None):
        super().__init__(
            name="graphql_generator",
            system_message=config['agents']['graphql_generator']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        logger.debug("Initializing GraphQLGeneratorAgent with config: %s", config)
        self.vector_store = vector_store or get_vector_store(config['vector_stores']['graphql_schema'])
        logger.debug("Vector store initialized: %s", self.vector_store)
    
    def get_schema_context(self, query: str) -> List[Document]:
        """Retrieve relevant schema information."""
        logger.debug("Retrieving GraphQL schema for query: %s", query)
        results = self.vector_store.similarity_search(query)
        logger.debug("Retrieved %d documents", len(results))
        return results

class CodeGeneratorAgent(autogen.AssistantAgent):
    """Agent for generating code and architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="code_generator",
            system_message=config['agents']['code_generator']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )

class OrchestratorAgent(autogen.AssistantAgent):
    """Agent for orchestrating other agents."""
    
    def __init__(self, config: Dict[str, Any], vector_stores: Dict[str, VectorStore] = None):
        super().__init__(
            name="orchestrator",
            system_message=config['agents']['orchestrator']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        logger.debug("Initializing OrchestratorAgent with config: %s", config)
        vector_stores = vector_stores or {}
        self.knowledge_retriever = KnowledgeRetrieverAgent(
            config,
            vector_store=vector_stores.get('knowledge_base')
        )
        self.sql_generator = SQLGeneratorAgent(
            config,
            vector_store=vector_stores.get('databricks_schema')
        )
        self.graphql_generator = GraphQLGeneratorAgent(
            config,
            vector_store=vector_stores.get('graphql_schema')
        )
        self.code_generator = CodeGeneratorAgent(config)

def create_agents(vector_stores: Dict[str, VectorStore] = None) -> Dict[str, autogen.AssistantAgent]:
    """Create all agents with their configurations."""
    config = load_config()
    vector_stores = vector_stores or {}
    
    return {
        "knowledge_retriever": KnowledgeRetrieverAgent(
            config,
            vector_store=vector_stores.get('knowledge_base')
        ),
        "sql_generator": SQLGeneratorAgent(config, vector_store=vector_stores.get('databricks_schema')),
        "graphql_generator": GraphQLGeneratorAgent(config, vector_store=vector_stores.get('graphql_schema')),
        "code_generator": CodeGeneratorAgent(config),
        "orchestrator": OrchestratorAgent(config, vector_stores)
    } 