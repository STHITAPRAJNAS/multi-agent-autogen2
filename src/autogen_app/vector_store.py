"""
Vector store interface for knowledge bases.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import yaml
from pathlib import Path
import os
from langchain_community.vectorstores import PGVector, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        pass

class MemoryVectorStore(VectorStore):
    """In-memory vector store using FAISS."""
    
    def __init__(self, config: Dict[str, Any]):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["embedding_model"]
        )
        self.vectorstore = None
    
    def add_documents(self, documents: List[Document]) -> None:
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)

class PGVectorStore(VectorStore):
    """PostgreSQL vector store using pgvector."""
    
    def __init__(self, config: Dict[str, Any]):
        connection_string = f"postgresql+psycopg2://{config["user"]}:{config["password"]}@{config["host"]}:{config["port"]}/{config["database"]}"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["embedding_model"]
        )
        self.vectorstore = PGVector(
            collection_name=config["collection_name"],
            connection_string=connection_string,
            embedding_function=self.embeddings
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        self.vectorstore.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)

def get_vector_store(config: Dict[str, Any]) -> VectorStore:
    """Factory function to create appropriate vector store."""
    store_type = config["type"]
    if store_type == "memory":
        return MemoryVectorStore(config)
    elif store_type == "pgvector":
        return PGVectorStore(config)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    env = os.getenv("ENV", "local")
    config_path = Path(__file__).parent.parent.parent / "config" / f"settings.{env}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
