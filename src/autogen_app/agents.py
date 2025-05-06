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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeRetrieverAgent(autogen.AssistantAgent):
    """Agent for retrieving information from knowledge bases."""
    
    def __init__(self, config: Dict[str, Any], vector_store: VectorStore = None):
        super().__init__(
            name="knowledge_retriever",
            system_message=config['agents']['knowledge_retriever']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        logger.info("Initializing KnowledgeRetrieverAgent with config: %s", config)
        self.vector_store = vector_store or get_vector_store(config['vector_stores']['knowledge_base'])
        logger.info("Vector store initialized: %s", self.vector_store)
    
    def retrieve_knowledge(self, query: str) -> List[Document]:
        """Retrieve relevant knowledge from the vector store."""
        logger.info("Retrieving knowledge for query: %s", query)
        results = self.vector_store.similarity_search(query)
        logger.info("Retrieved %d documents", len(results))
        return results

class SQLGeneratorAgent(autogen.AssistantAgent):
    """Agent for generating SQL queries."""
    
    def __init__(self, config: Dict[str, Any], vector_store: VectorStore = None):
        super().__init__(
            name="sql_generator",
            system_message=config['agents']['sql_generator']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        logger.info("Initializing SQLGeneratorAgent with config: %s", config)
        self.vector_store = vector_store or get_vector_store(config['vector_stores']['databricks_schema'])
        logger.info("Vector store initialized: %s", self.vector_store)
    
    def get_schema_context(self, query: str) -> List[Document]:
        """Retrieve relevant schema information."""
        logger.info("Retrieving SQL schema for query: %s", query)
        results = self.vector_store.similarity_search(query)
        logger.info("Retrieved %d documents", len(results))
        return results

class GraphQLGeneratorAgent(autogen.AssistantAgent):
    """Agent for generating GraphQL queries."""
    
    def __init__(self, config: Dict[str, Any], vector_store: VectorStore = None):
        super().__init__(
            name="graphql_generator",
            system_message=config['agents']['graphql_generator']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        logger.info("Initializing GraphQLGeneratorAgent with config: %s", config)
        self.vector_store = vector_store or get_vector_store(config['vector_stores']['graphql_schema'])
        logger.info("Vector store initialized: %s", self.vector_store)
    
    def get_schema_context(self, query: str) -> List[Document]:
        """Retrieve relevant schema information."""
        logger.info("Retrieving GraphQL schema for query: %s", query)
        results = self.vector_store.similarity_search(query)
        logger.info("Retrieved %d documents", len(results))
        return results

class CodeGeneratorAgent(autogen.AssistantAgent):
    """Agent for generating code and architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="code_generator",
            system_message=config['agents']['code_generator']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )

def create_agents(vector_stores: Dict[str, VectorStore] = None) -> Dict[str, autogen.AssistantAgent]:
    """Create all agents with their configurations."""
    config = load_config()
    vector_stores = vector_stores or {}
    
    # Create specialized agents
    knowledge_retriever = KnowledgeRetrieverAgent(
        config,
        vector_store=vector_stores.get('knowledge_base')
    )
    sql_generator = SQLGeneratorAgent(
        config,
        vector_store=vector_stores.get('databricks_schema')
    )
    graphql_generator = GraphQLGeneratorAgent(
        config,
        vector_store=vector_stores.get('graphql_schema')
    )
    code_generator = CodeGeneratorAgent(config)
    
    # Define tool functions with proper schemas
    def retrieve_knowledge(query: str) -> str:
        """Retrieve relevant knowledge from the knowledge base.
        
        Args:
            query: The search query to find relevant information.
            
        Returns:
            str: The retrieved knowledge in a formatted string.
        """
        results = knowledge_retriever.retrieve_knowledge(query)
        return "\n\n".join([doc.page_content for doc in results])
    
    def get_sql_schema(query: str) -> str:
        """Get relevant SQL schema information.
        
        Args:
            query: The search query to find relevant schema information.
            
        Returns:
            str: The retrieved schema information in a formatted string.
        """
        results = sql_generator.get_schema_context(query)
        return "\n\n".join([doc.page_content for doc in results])
    
    def get_graphql_schema(query: str) -> str:
        """Get relevant GraphQL schema information.
        
        Args:
            query: The search query to find relevant schema information.
            
        Returns:
            str: The retrieved schema information in a formatted string.
        """
        results = graphql_generator.get_schema_context(query)
        return "\n\n".join([doc.page_content for doc in results])
    
    # Create supervisor agent with tool calling capabilities
    supervisor = autogen.AssistantAgent(
        name="supervisor",
        system_message="""You are a supervisor agent responsible for analyzing user queries and determining which specialized agents should handle them.
        You have access to the following tools:
        - retrieve_knowledge(query): Retrieve information from documentation
        - get_sql_schema(query): Get database schema information
        - get_graphql_schema(query): Get GraphQL schema information
        
        Your tasks are:
        1. Analyze the query to understand what information is needed
        2. Use the appropriate tools to gather information
        3. Synthesize the results into a coherent response
        
        Always explain your reasoning for using specific tools and how the information will be combined.
        
        When using tools, follow these steps:
        1. First, explain why you're using a particular tool
        2. Then call the tool with the appropriate query
        3. Finally, analyze the results and decide if you need more information
        
        Example:
        User: "What is the user data structure in our system?"
        Assistant: "I'll help you understand the user data structure. First, I'll check our documentation for any relevant information about the user model."
        Assistant: [Calls retrieve_knowledge with query "user data structure model"]
        Assistant: "Based on the documentation, I see that we need to understand both the database schema and GraphQL interface. Let me check those."
        Assistant: [Calls get_sql_schema with query "user table schema"]
        Assistant: [Calls get_graphql_schema with query "user type definition"]
        Assistant: "Now I can provide a complete picture of the user data structure..." """,
        llm_config={
            **get_llm_provider(config['llm']).get_config(),
            "functions": [
                {
                    "name": "retrieve_knowledge",
                    "description": "Retrieve relevant knowledge from the knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant information"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_sql_schema",
                    "description": "Get relevant SQL schema information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant schema information"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_graphql_schema",
                    "description": "Get relevant GraphQL schema information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant schema information"
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
        },
        function_map={
            "retrieve_knowledge": retrieve_knowledge,
            "get_sql_schema": get_sql_schema,
            "get_graphql_schema": get_graphql_schema
        }
    )
    
    return {
        "knowledge_retriever": knowledge_retriever,
        "sql_generator": sql_generator,
        "graphql_generator": graphql_generator,
        "code_generator": code_generator,
        "supervisor": supervisor
    } 