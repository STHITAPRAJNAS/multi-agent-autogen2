import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Testing Agentic System\n",
                "\n",
                "This notebook tests the multi-agent system by:\n",
                "1. Setting up the environment\n",
                "2. Hydrating the knowledge base with example documents\n",
                "3. Testing each agent's functionality\n",
                "4. Testing the orchestrator with complex queries"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "from dotenv import load_dotenv\n",
                "\n",
                "# Add src to Python path\n",
                "src_path = str(Path.cwd() / 'src')\n",
                "if src_path not in sys.path:\n",
                "    sys.path.append(src_path)\n",
                "\n",
                "from autogen_app.agents import create_agents\n",
                "from autogen_app.vector_store import get_vector_store, load_config\n",
                "from langchain.schema import Document"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Set up environment variables from .env file"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load environment variables from .env file\n",
                "load_dotenv()\n",
                "\n",
                "# Verify environment variables are loaded\n",
                "print(f\"Environment: {os.getenv('ENV')}\")\n",
                "print(f\"AWS Access Key ID: {'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not Set'}\")\n",
                "print(f\"AWS Secret Access Key: {'Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not Set'}\")\n",
                "print(f\"Gemini API Key: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not Set'}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Hydrate the knowledge base with example documents"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load configuration\n",
                "config = load_config()\n",
                "\n",
                "# Get vector stores\n",
                "knowledge_store = get_vector_store(config['vector_stores']['knowledge_base'])\n",
                "sql_store = get_vector_store(config['vector_stores']['databricks_schema'])\n",
                "graphql_store = get_vector_store(config['vector_stores']['graphql_schema'])\n",
                "\n",
                "# Example Confluence knowledge documents\n",
                "confluence_docs = [\n",
                "    Document(\n",
                "        page_content=\"\"\"\n",
                "        Project Overview\n",
                "        Our project uses a microservices architecture with the following components:\n",
                "        - User Service: Handles user authentication and profile management\n",
                "        - Order Service: Manages order processing and fulfillment\n",
                "        - Payment Service: Handles payment processing and transactions\n",
                "        \"\"\",\n",
                "        metadata={\"source\": \"confluence\", \"title\": \"Project Architecture\"}\n",
                "    ),\n",
                "    Document(\n",
                "        page_content=\"\"\"\n",
                "        API Documentation\n",
                "        The User Service exposes the following endpoints:\n",
                "        - POST /api/users: Create new user\n",
                "        - GET /api/users/{id}: Get user details\n",
                "        - PUT /api/users/{id}: Update user information\n",
                "        \"\"\",\n",
                "        metadata={\"source\": \"confluence\", \"title\": \"User Service API\"}\n",
                "    )\n",
                "]\n",
                "\n",
                "# Example Databricks schema documents\n",
                "databricks_docs = [\n",
                "    Document(\n",
                "        page_content=\"\"\"\n",
                "        Users Table Schema\n",
                "        CREATE TABLE users (\n",
                "            user_id INT PRIMARY KEY,\n",
                "            username VARCHAR(50),\n",
                "            email VARCHAR(100),\n",
                "            created_at TIMESTAMP\n",
                "        )\n",
                "        \"\"\",\n",
                "        metadata={\"source\": \"databricks\", \"table\": \"users\"}\n",
                "    ),\n",
                "    Document(\n",
                "        page_content=\"\"\"\n",
                "        Orders Table Schema\n",
                "        CREATE TABLE orders (\n",
                "            order_id INT PRIMARY KEY,\n",
                "            user_id INT,\n",
                "            total_amount DECIMAL(10,2),\n",
                "            status VARCHAR(20),\n",
                "            created_at TIMESTAMP,\n",
                "            FOREIGN KEY (user_id) REFERENCES users(user_id)\n",
                "        )\n",
                "        \"\"\",\n",
                "        metadata={\"source\": \"databricks\", \"table\": \"orders\"}\n",
                "    )\n",
                "]\n",
                "\n",
                "# Example GraphQL schema documents\n",
                "graphql_docs = [\n",
                "    Document(\n",
                "        page_content=\"\"\"\n",
                "        User Type\n",
                "        type User {\n",
                "            id: ID!\n",
                "            username: String!\n",
                "            email: String!\n",
                "            orders: [Order!]!\n",
                "        }\n",
                "        \"\"\",\n",
                "        metadata={\"source\": \"graphql\", \"type\": \"User\"}\n",
                "    ),\n",
                "    Document(\n",
                "        page_content=\"\"\"\n",
                "        Order Type\n",
                "        type Order {\n",
                "            id: ID!\n",
                "            user: User!\n",
                "            totalAmount: Float!\n",
                "            status: String!\n",
                "            createdAt: DateTime!\n",
                "        }\n",
                "        \"\"\",\n",
                "        metadata={\"source\": \"graphql\", \"type\": \"Order\"}\n",
                "    )\n",
                "]\n",
                "\n",
                "# Add documents to vector stores\n",
                "knowledge_store.add_documents(confluence_docs)\n",
                "sql_store.add_documents(databricks_docs)\n",
                "graphql_store.add_documents(graphql_docs)\n",
                "\n",
                "print(\"Knowledge base hydrated successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Create and test agents"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create all agents\n",
                "agents = create_agents()\n",
                "\n",
                "# Test knowledge retriever\n",
                "print(\"Testing Knowledge Retriever:\")\n",
                "knowledge = agents['knowledge_retriever'].retrieve_knowledge(\"What are the main components of the project?\")\n",
                "for doc in knowledge:\n",
                "    print(f\"\\nSource: {doc.metadata['source']}\")\n",
                "    print(f\"Title: {doc.metadata['title']}\")\n",
                "    print(f\"Content: {doc.page_content}\")\n",
                "\n",
                "# Test SQL generator\n",
                "print(\"\\nTesting SQL Generator:\")\n",
                "schema_context = agents['sql_generator'].get_schema_context(\"users table schema\")\n",
                "for doc in schema_context:\n",
                "    print(f\"\\nSource: {doc.metadata['source']}\")\n",
                "    print(f\"Table: {doc.metadata['table']}\")\n",
                "    print(f\"Content: {doc.page_content}\")\n",
                "\n",
                "# Test GraphQL generator\n",
                "print(\"\\nTesting GraphQL Generator:\")\n",
                "schema_context = agents['graphql_generator'].get_schema_context(\"user type definition\")\n",
                "for doc in schema_context:\n",
                "    print(f\"\\nSource: {doc.metadata['source']}\")\n",
                "    print(f\"Type: {doc.metadata['type']}\")\n",
                "    print(f\"Content: {doc.page_content}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Test complex queries with orchestrator"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test complex query that requires multiple agents\n",
                "print(\"Testing Orchestrator with Complex Query:\")\n",
                "orchestrator = agents['orchestrator']\n",
                "\n",
                "# Example complex query\n",
                "complex_query = \"\"\"\n",
                "I need to understand how user data is structured in our system.\n",
                "Please provide:\n",
                "1. The database schema for users\n",
                "2. The GraphQL type definition for users\n",
                "3. Any relevant API documentation\n",
                "\"\"\"\n",
                "\n",
                "# Get knowledge from all relevant sources\n",
                "knowledge = orchestrator.knowledge_retriever.retrieve_knowledge(\"user service API documentation\")\n",
                "sql_schema = orchestrator.sql_generator.get_schema_context(\"users table schema\")\n",
                "graphql_schema = orchestrator.graphql_generator.get_schema_context(\"user type definition\")\n",
                "\n",
                "print(\"\\nKnowledge from Confluence:\")\n",
                "for doc in knowledge:\n",
                "    print(f\"\\nSource: {doc.metadata['source']}\")\n",
                "    print(f\"Title: {doc.metadata['title']}\")\n",
                "    print(f\"Content: {doc.page_content}\")\n",
                "\n",
                "print(\"\\nSQL Schema:\")\n",
                "for doc in sql_schema:\n",
                "    print(f\"\\nSource: {doc.metadata['source']}\")\n",
                "    print(f\"Table: {doc.metadata['table']}\")\n",
                "    print(f\"Content: {doc.page_content}\")\n",
                "\n",
                "print(\"\\nGraphQL Schema:\")\n",
                "for doc in graphql_schema:\n",
                "    print(f\"\\nSource: {doc.metadata['source']}\")\n",
                "    print(f\"Type: {doc.metadata['type']}\")\n",
                "    print(f\"Content: {doc.page_content}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Test code generation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test code generation with context from other agents\n",
                "print(\"Testing Code Generator:\")\n",
                "code_generator = agents['code_generator']\n",
                "\n",
                "# Get context from other agents\n",
                "knowledge = orchestrator.knowledge_retriever.retrieve_knowledge(\"project architecture\")\n",
                "sql_schema = orchestrator.sql_generator.get_schema_context(\"users and orders tables\")\n",
                "\n",
                "# Combine context for code generation\n",
                "context = \"\"\"\n",
                "Based on the following information:\n",
                "\n",
                "Project Architecture:\n",
                "{}\n",
                "\n",
                "Database Schema:\n",
                "{}\n",
                "\n",
                "Please generate a Python class for the User Service that:\n",
                "1. Implements user CRUD operations\n",
                "2. Uses SQLAlchemy for database access\n",
                "3. Follows best practices for error handling and logging\n",
                "\"\"\".format(\n",
                "    knowledge[0].page_content if knowledge else \"\",\n",
                "    sql_schema[0].page_content if sql_schema else \"\"\n",
                ")\n",
                "\n",
                "# Generate code\n",
                "print(\"\\nGenerated Code:\")\n",
                "print(context)  # In a real implementation, this would be sent to the code generator"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to a file
with open('notebooks/test_agentic_system.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 