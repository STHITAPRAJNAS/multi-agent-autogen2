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