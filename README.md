# Agentic System with AutoGen

A multi-agent system built with AutoGen that can:
- Retrieve knowledge from Confluence (stored in pgvector)
- Generate SQL queries for Databricks
- Generate GraphQL queries
- Generate code and architecture
- Orchestrate multiple agents to handle complex queries

## Features

- **Knowledge Retrieval**: Uses pgvector to store and retrieve Confluence knowledge
- **SQL Generation**: Generates SQL queries for Databricks with schema awareness
- **GraphQL Generation**: Creates GraphQL queries based on schema knowledge
- **Code Generation**: Expert code and architecture generation
- **Multi-Agent Orchestration**: Intelligent task delegation to appropriate agents
- **Flexible LLM Support**: Works with Bedrock (Claude) and Gemini
- **Local Testing**: Memory-based vector stores for local development
- **Production Ready**: pgvector integration for production use

## Prerequisites

- Python 3.8 or higher
- `uv` package manager
- AWS Account (for Bedrock)
- Google Cloud Account (for Gemini)
- PostgreSQL with pgvector extension (for production)

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd multi-agent-autogen2
   ```

2. **Install Dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

3. **Configure Environment**:
   Create a `.env` file in the project root:
   ```env
   ENV=local  # or 'prod' for production
   AWS_ACCESS_KEY_ID=your_aws_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret
   GEMINI_API_KEY=your_gemini_key  # if using Gemini
   ```

4. **Configure Vector Stores**:
   - For local development: Uses in-memory FAISS
   - For production: Update `config/settings.prod.yaml` with your PostgreSQL details

## Testing

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open Test Notebook**:
   Navigate to `notebooks/test_agentic_system.ipynb`

3. **Run Tests**:
   - Hydrate the knowledge base with example documents
   - Test knowledge retrieval
   - Test SQL generation
   - Test GraphQL generation
   - Test code generation
   - Test orchestrator with complex queries

## Configuration

The system uses two configuration files:
- `config/settings.local.yaml`: For local development
- `config/settings.prod.yaml`: For production

Key configuration sections:
- `llm`: LLM provider settings (Bedrock or Gemini)
- `vector_stores`: Vector store configurations
- `agents`: Agent-specific settings and prompts

## Architecture

The system consists of several components:

1. **Vector Stores**:
   - Knowledge base (Confluence content)
   - Databricks schema
   - GraphQL schema

2. **Agents**:
   - Knowledge Retriever
   - SQL Generator
   - GraphQL Generator
   - Code Generator
   - Orchestrator

3. **LLM Providers**:
   - Bedrock (Claude)
   - Gemini

## Usage

1. **Basic Usage**:
   ```python
   from autogen_app.agents import create_agents
   
   # Create agents
   agents = create_agents()
   
   # Use specific agent
   knowledge = agents['knowledge_retriever'].retrieve_knowledge("your query")
   ```

2. **Complex Queries**:
   ```python
   # Use orchestrator for complex queries
   orchestrator = agents['orchestrator']
   result = orchestrator.handle_query("your complex query")
   ```

## Development

1. **Adding New Knowledge**:
   - Add documents to the appropriate vector store
   - Use the test notebook to verify retrieval

2. **Customizing Agents**:
   - Modify prompts in `config/settings.*.yaml`
   - Update agent logic in `src/autogen_app/agents.py`

3. **Adding New LLM Provider**:
   - Implement new provider in `src/autogen_app/llm_provider.py`
   - Update configuration files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License