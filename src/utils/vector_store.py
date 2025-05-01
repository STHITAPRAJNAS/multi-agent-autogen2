import os
from langchain_community.vectorstores.pgvector import PGVector
from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader
from genkit import get_project_id
from dotenv import load_dotenv
from src.config_loader import ConfigLoader

load_dotenv()
config_loader = ConfigLoader()


CONNECTION_STRING = config_loader.get_pg_config().get("pg_db_url")
COLLECTION_NAME = config_loader.get_pg_config().get("pg_collection_name")


def prepare_vector_store():
  """
    Create the collection in pg vector and add the data.
  """
  try:
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001", project=get_project_id())
    vector_store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

    loaders = [
        TextLoader("./knowledge_base/doc1.txt"),
        TextLoader("./knowledge_base/doc2.txt"),
        TextLoader("./knowledge_base/doc3.txt"),
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    texts = text_splitter.split_documents(documents)
    vector_store.add_documents(texts)
    return True
  except Exception as e:
      print(f"Error adding data to pgvector: {e}")
      return False


def get_vector_store():
  """
  Get the vector store.
  """
  embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001", project=get_project_id())
  vector_store = PGVector(
      collection_name=COLLECTION_NAME,
      connection_string=CONNECTION_STRING,
      embedding_function=embeddings,
  )
  return vector_store