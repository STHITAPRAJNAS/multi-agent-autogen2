# Multi-Agent Autogen Application

This is a multi-agent application built using Autogen and FastAPI, designed to perform tasks like knowledge retrieval, SQL query generation, and code generation using various Large Language Models (LLMs) like OpenAI, Anthropic (via API key or Bedrock), and Gemini (via API key or Bedrock). The application follows best practices for modularity, configuration management, and production readiness.

## Features

*   Multi-agent system using Autogen.
*   Knowledge Retrieval (RAG) capabilities.
*   Text-to-SQL generation against Databricks (simulated).
*   Code generation (Python).
*   Modular agent design.
*   FastAPI-based API for chatbot interaction.
*   Conversation state management with history and cost tracking.
*   Configurable LLM providers (OpenAI, Anthropic, Gemini via API Key or Bedrock).
*   Environment-specific configuration loading (local vs. production).
*   Robust logging and error handling.
*   Jupyter Notebook for step-by-step testing and debugging of agent logic.
*   `uv` as the package manager.

## Prerequisites

Before you begin, ensure you have the following installed:

*   Python 3.9 or higher
*   `uv` package manager
*   An IDE with Python and Jupyter Notebook support (optional, but recommended)
*   AWS Account (required for Bedrock models)
*   API Keys for OpenAI, Anthropic, or Gemini (depending on your chosen `MODEL_TYPE`)

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your_repository_url>
    cd multi-agent-autogen
    ```
    Replace `<your_repository_url>` with the actual URL of your GitHub repository.

2.  **Install Dependencies:**
    Use `uv` to install the project dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    The application uses environment variables for critical settings and API keys. Set the following variables in your terminal or in a `.env` file (make sure `.env` is in your `.gitignore`):

    *   `ENV`: Set to `local` for local development or `prod` for production. This determines which configuration file (`config/settings.local.yaml` or `config/settings.prod.yaml`) is loaded.
    *   `MODEL_TYPE`: Set to one of `openai`, `anthropic`, `gemini`, or `bedrock`. This determines which LLM provider is used.
    *   `ANTHROPIC_API_KEY`: Your API key for Anthropic models (required if `MODEL_TYPE` is `anthropic`).
    *   `GEMINI_API_KEY`: Your API key for Gemini models (required if `MODEL_TYPE` is `gemini`).
    *   `OAI_CONFIG_LIST`: A JSON string or path to a JSON file for OpenAI configurations (required if `MODEL_TYPE` is `openai`).
    *   `DEFAULT_MODEL`: The default OpenAI model to use (required if `MODEL_TYPE` is `openai`).
    *   `AWS_ACCESS_KEY_ID`: Your AWS access key ID (required if `MODEL_TYPE` is `bedrock`).
    *   `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key (required if `MODEL_TYPE` is `bedrock`).
    *   `BEDROCK_REGION`: The AWS region for Bedrock (e.g., `us-east-1`, required if `MODEL_TYPE` is `bedrock`).

    Example using a `.env` file (create `.env` in the project root):
    ```env
    ENV=local
    MODEL_TYPE=anthropic
    ANTHROPIC_API_KEY=your_anthropic_api_key
    # GEMINI_API_KEY=your_gemini_api_key
    # OAI_CONFIG_LIST={"your":"openai_config"}
    # DEFAULT_MODEL=gpt-4
    # AWS_ACCESS_KEY_ID=your_aws_access_key_id
    # AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
    # BEDROCK_REGION=us-east-1
    ```
    **Note:** Uncomment and set the variables relevant to your chosen `MODEL_TYPE`.

4.  **Configure Settings Files:**
    Review and update the configuration files in the `config/` directory (`settings.local.yaml` and `settings.prod.yaml`) to match your specific needs (e.g., database URLs, default models, cost per token, history length). Environment variables will override values in these files.

5.  **Database Setup (Local - SQLite):**
    For local development with `ENV=local`, the application uses a SQLite database file (`local.db` by default, configured in `config/settings.local.yaml`). This file will be created automatically the first time you run the application.

6.  **Database Setup (Production - PostgreSQL):**
    For production with `ENV=prod`, the application is configured to use PostgreSQL (as specified in `config/settings.prod.yaml`). You need to have a PostgreSQL database running and update the `db_url` in `config/settings.prod.yaml` with your connection string. Database migration tools (like Alembic) would typically be used in a production setup, but are not included in this initial version.

## Running the Application

Once the setup is complete, you can run the FastAPI application using Uvicorn:

```bash
vicorn src.app:app --reload
```

The `--reload` flag is useful for local development as it restarts the server automatically when code changes are detected. The application will run on `http://127.0.0.1:8000` by default.

## Testing the Application

You can test the application's API endpoints using tools like `curl` or Postman, or by accessing the automatically generated OpenAPI documentation at `http://127.0.0.1:8000/docs`.

### Testing API Endpoints

1.  **`/chat` (POST):** Send a user query to the multi-agent system.
    ```bash
    curl -X POST \
      http://127.0.0.1:8000/chat \
      -H 'Content-Type: application/json' \
      -d '{
        "query": "What is the capital of Italy?",
        "conversation_id": "testuser1"
      }'
    ```
    Send multiple requests with the same `conversation_id` to test follow-up questions.

2.  **`/get_conversation_history` (GET):** Retrieve the conversation history for a specific `conversation_id`.
    ```bash
    curl "http://127.0.0.1:8000/get_conversation_history?conversation_id=testuser1"
    ```

3.  **`/clear_conversation` (POST):** Clear the conversation history for a specific `conversation_id`.
    ```bash
    curl -X POST \
      http://127.0.0.1:8000/clear_conversation \
      -H 'Content-Type: application/json' \
      -d '{
        "conversation_id": "testuser1"
      }'
    ```

### Testing Agent Logic Step-by-Step (Jupyter Notebook)

You can use the provided Jupyter Notebook (`notebook/test_agents.ipynb`) to test the individual agent logic and prompts without running the full FastAPI application.

1.  **Install Jupyter:** If you don't have Jupyter installed, you can install it using `uv`:
    ```bash
    uv pip install notebook
    ```
2.  **Start Jupyter:**
    ```bash
    jupyter notebook
    ```
3.  **Open Notebook:** Navigate to the `notebook/` directory and open `test_agents.ipynb`.
4.  **Run Cells:** Execute the notebook cells step by step to:
    *   Load configurations and create agents.
    *   Test knowledge retrieval.
    *   Test SQL generation.
    *   Test code generation.
    *   Inspect the conversation state.

This notebook is valuable for debugging and refining agent prompts and behaviors.

## Configuration Details

The application loads configuration from `config/settings.<ENV>.yaml`. The `ENV` environment variable determines which file is used (`local` or `prod`). Environment variables also override any settings found in the YAML files.

The `ConfigLoader` class (`src/config_loader.py`) handles this loading process.

## Logging

The application uses Python's built-in logging. The configuration is defined in `src/utils/utils.py` and includes logging to the console and a file (`app.log` by default). Check these logs for information, warnings, and errors.

---

Feel free to contribute to this project by submitting issues or pull requests!