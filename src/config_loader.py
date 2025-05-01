import yaml  # Corrected import name
import os
import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Loads configuration from a YAML file and allows overriding with environment variables.
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        env = os.environ.get("ENV", "local")
        if env == "prod":
          self.config_path = os.path.join(self.config_dir, "settings.prod.yaml")
        elif env == "local":
          self.config_path = os.path.join(self.config_dir, "settings.local.yaml")
        
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads the configuration from the YAML file.

        Returns:
            Dict[str, Any]: The loaded configuration.
        """
        try:
            with open(self.config_path, "r") as file:  # Added "r" for read mode
                config = yaml.safe_load(file)
                logger.info(f"Loaded config from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Config file not found at {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML in {self.config_path}: {e}")
            raise

    def get_config(self, key: str) -> Optional[Any]:
        """
        Gets a config value by key, with environment variable override.

        Args:
            key (str): The config key.

        Returns:
            Optional[Any]: The config value or None if not found.
        """
        env_value = os.environ.get(key)
        if env_value:
            logger.info(f"Using environment variable for {key}")
            return env_value

        if key in self.config.get("app_config", {}).get("app", {}):
            return self.config["app_config"]["app"][key]
        elif key in self.config.get("app_config", {}).get("llm_config", {}):
            return self.config["app_config"]["llm_config"][key]
        elif key in self.config.get("bedrock_config", {}):
            return self.config["bedrock_config"][key]
        elif key in self.config.get("database_config", {}):
            return self.config["database_config"][key]
        elif "bedrock_config" in self.config:          
          if key in self.config["bedrock_config"]:
            return self.config["bedrock_config"][key]          
        else:
            logger.warning(f"Config key {key} not found.")
            return None
    
    def get_bedrock_config(self) -> Optional[Dict[str, Any]]:
        """
        Gets the bedrock configuration.

        Returns:
            Optional[Dict[str, Any]]: The bedrock configuration or None if not found.
        """
        if "bedrock_config" in self.config:
            return self.config["bedrock_config"]
        return None
    
    def get_database_config(self) -> Optional[Dict[str, Any]]:
        """
        Gets the database configuration.

        Returns:
            Optional[Dict[str, Any]]: The database configuration or None if not found.
        """
        if "database_config" in self.config:
            return self.config["database_config"]
        return None
