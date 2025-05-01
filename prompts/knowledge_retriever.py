# prompts/knowledge_retriever.py

KNOWLEDGE_RETRIEVER_PROMPT = """
You are a knowledge retriever.
Your task is to help users retrieve information from the knowledge base.
Use the `knowledge_retriever_helper` function to find information.
Keep your responses short and focused on the user's query.
Provide only the relevant information.
Do not use code blocks in your response.
"""
```
```python
# prompts/sql_generator.py
SQL_GENERATOR_PROMPT = """
You are a SQL query generator.
Your task is to generate SQL queries based on user requests.
- Analyze the user's query to determine the intent.
- The user's query can be based on the conversation history, consider that when analyzing.
- Generate only one valid SQL query.
- Output the result using a code block.
"""
```
```python
# prompts/code_generator.py
CODE_GENERATOR_PROMPT = """
You are a Python code generator.
Your task is to generate Python code based on user requests.
- Analyze the user's query to determine the intent.
- The user's query can be based on the conversation history, consider that when analyzing.
- Generate only a valid Python code.
- Output the result using a code block.
"""
```
```python
# prompts/user_proxy.py
USER_PROXY_PROMPT = """
You are a helpful user proxy.
Your main task is to initiate the conversation, and help with the interactions with the other agents.
"""
```
```python
# src/utils/prompt_loader.py
import importlib
import os
from typing import Dict


class PromptLoader:
    """
    Loads prompts from Python files.
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self.prompts: Dict[str, str] = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """
        Loads all prompts from Python files in the prompts directory.

        Returns:
            Dict[str, str]: A dictionary where keys are prompt names and values are the prompts.
        """
        prompts = {}
        for filename in os.listdir(self.prompts_dir):
            if filename.endswith(".py"):
                module_name = filename[:-3]  # Remove .py
                module_path = os.path.join(self.prompts_dir, filename)

                spec = importlib.util.spec_from_file_location(
                    module_name, module_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name in dir(module):
                    if name.endswith("_PROMPT"):
                        prompt_name = name.lower().replace("_prompt", "")
                        prompts[prompt_name] = getattr(module, name)

        return prompts

    def get_prompt(self, prompt_name: str) -> str:
        """
        Gets a prompt by name.

        Args:
            prompt_name (str): The name of the prompt to get.

        Returns:
            str: The prompt.

        Raises:
            ValueError: If the prompt is not found.
        """
        prompt = self.prompts.get(prompt_name)
        if prompt is None:
            raise ValueError(f"Prompt '{prompt_name}' not found.")
        return prompt
```
```python
# src/agents/knowledge_retriever.py
from autogen import AssistantAgent
from src.utils.prompt_loader import PromptLoader

def create_knowledge_retriever_agent(llm_config, cache_seed_path):
    prompt_loader = PromptLoader()
    knowledge_retriever_prompt = prompt_loader.get_prompt("knowledge_retriever")

    knowledge_retriever = AssistantAgent(
        name="knowledge_retriever",
        system_message=knowledge_retriever_prompt,
        llm_config=llm_config,
    )

    return knowledge_retriever
```
```python
# src/agents/sql_generator.py
from autogen import AssistantAgent
from src.utils.prompt_loader import PromptLoader

def create_sql_generator_agent(