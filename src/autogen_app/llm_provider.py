"""
LLM provider interface for different model providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import os
import boto3
import google.generativeai as genai
from autogen import OpenAIWrapper

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get LLM configuration for AutoGen."""
        pass

class BedrockProvider(LLMProvider):
    """Bedrock LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.model = config['model']
        self.region = config['region']
        self.bedrock = boto3.client(
            'bedrock-runtime',
            region_name=self.region
        )
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "config_list": [{
                "model": self.model,
                "client": self.bedrock
            }]
        }

class GeminiProvider(LLMProvider):
    """Gemini LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config['api_key']
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "config_list": [{
                "model": "gemini-pro",
                "client": self.model
            }]
        }

def get_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """Factory function to create appropriate LLM provider."""
    provider = config['provider']
    if provider == 'bedrock':
        return BedrockProvider(config)
    elif provider == 'gemini':
        return GeminiProvider(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 