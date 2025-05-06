"""
Test script for the supervisor agent with tool calling capabilities.
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    import autogen
    from autogen_app.agents import create_agents
    from autogen_app.vector_store import get_vector_store, load_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you have installed all required dependencies.")
    sys.exit(1)

def main():
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config()
    
    # Create vector stores
    vector_stores = {
        'knowledge_base': get_vector_store(config['vector_stores']['knowledge_base']),
        'databricks_schema': get_vector_store(config['vector_stores']['databricks_schema']),
        'graphql_schema': get_vector_store(config['vector_stores']['graphql_schema'])
    }
    
    # Create agents
    agents = create_agents(vector_stores)
    supervisor = agents['supervisor']
    
    # Create a user proxy agent
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "workspace",
            "use_docker": False,
        },
    )
    
    # Test queries
    test_queries = [
        "What is the user data structure in our system?",
        "How do I create a new user and place an order using the GraphQL API?",
        "What are the main components of our microservices architecture and how do they interact?",
        "Show me the database schema for the orders table and how it relates to users",
        "What are the available GraphQL mutations for user management?"
    ]
    
    # Test each query
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Testing query: {query}")
        print(f"{'='*80}\n")
        
        try:
            # Create a chat between user proxy and supervisor
            chat = user_proxy.initiate_chat(
                recipient=supervisor,
                message=query
            )
            
            # Print the response
            print("Response:")
            print(chat.last_message()["content"])
            print("\n")
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            print("\n")

if __name__ == "__main__":
    main() 