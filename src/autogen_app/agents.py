"""
Specialized agents for the multi-agent system.
"""

from typing import Dict, Any, List
import autogen
from .vector_store import get_vector_store, load_config
from .llm_provider import get_llm_provider
from langchain.schema import Document

class KnowledgeRetrieverAgent(autogen.AssistantAgent):
    """Agent for retrieving information from knowledge bases."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="knowledge_retriever",
            system_message=config['agents']['knowledge_retriever']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        self.vector_store = get_vector_store(config['vector_stores']['knowledge_base'])
    
    def retrieve_knowledge(self, query: str) -> List[Document]:
        """Retrieve relevant knowledge from the vector store."""
        return self.vector_store.similarity_search(query)

class SQLGeneratorAgent(autogen.AssistantAgent):
    """Agent for generating SQL queries."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="sql_generator",
            system_message=config['agents']['sql_generator']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        self.vector_store = get_vector_store(config['vector_stores']['databricks_schema'])
    
    def get_schema_context(self, query: str) -> List[Document]:
        """Retrieve relevant schema information."""
        return self.vector_store.similarity_search(query)

class GraphQLGeneratorAgent(autogen.AssistantAgent):
    """Agent for generating GraphQL queries."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="graphql_generator",
            system_message=config['agents']['graphql_generator']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        self.vector_store = get_vector_store(config['vector_stores']['graphql_schema'])
    
    def get_schema_context(self, query: str) -> List[Document]:
        """Retrieve relevant schema information."""
        return self.vector_store.similarity_search(query)

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
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="orchestrator",
            system_message=config['agents']['orchestrator']['system_prompt'],
            llm_config=get_llm_provider(config['llm']).get_config()
        )
        self.knowledge_retriever = KnowledgeRetrieverAgent(config)
        self.sql_generator = SQLGeneratorAgent(config)
        self.graphql_generator = GraphQLGeneratorAgent(config)
        self.code_generator = CodeGeneratorAgent(config)

def create_agents() -> Dict[str, autogen.AssistantAgent]:
    """Create all agents with their configurations."""
    config = load_config()
    return {
        "knowledge_retriever": KnowledgeRetrieverAgent(config),
        "sql_generator": SQLGeneratorAgent(config),
        "graphql_generator": GraphQLGeneratorAgent(config),
        "code_generator": CodeGeneratorAgent(config),
        "orchestrator": OrchestratorAgent(config)
    } 