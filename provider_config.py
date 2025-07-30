"""OpenRouter Provider Validator - Provider Configuration

Manages provider-specific agent creation and configuration.
"""

import os
from typing import Dict, List, Optional

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from client import FileSystemClient, ProviderConfig

# Initialize the filesystem client
client = FileSystemClient()

def get_provider_configs() -> List[ProviderConfig]:
    """Load all provider configurations.
    
    Returns:
        List of provider configurations
    """
    return client.load_providers()

def get_enabled_providers() -> List[ProviderConfig]:
    """Get only enabled provider configurations.
    
    Returns:
        List of enabled provider configurations
    """
    providers = client.load_providers()
    return [p for p in providers if p.enabled]

def create_provider_agent(provider_name: str, model_name: str, system_prompt: str) -> Agent:
    """Create a provider-specific agent with appropriate routing.
    
    Args:
        provider_name: Name of the provider to route to
        model_name: Model identifier
        system_prompt: System prompt for the agent
        
    Returns:
        Configured Agent instance
    """
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    # Set up provider-specific routing
    model_settings = {
        "extra_body": {
            "provider": provider_name
        }
    }
    
    # Create the model
    model = OpenAIModel(
        model_name,
        provider=OpenAIProvider(
            base_url='https://openrouter.ai/api/v1',
            api_key=api_key
        ),
        settings=model_settings
    )
    
    # Set up MCP servers
    mcp_servers = [MCPServerStdio('python', ['./mcp_server.py'])]
    
    # Create the agent
    return Agent(model, mcp_servers=mcp_servers, system_prompt=system_prompt)

def save_provider_config(config: ProviderConfig) -> bool:
    """Save a provider configuration.
    
    Args:
        config: Provider configuration to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        providers = client.load_providers()
        
        # Check if provider already exists
        for i, provider in enumerate(providers):
            if provider.name == config.name:
                # Update existing provider
                providers[i] = config
                client.save_providers(providers)
                return True
        
        # Add new provider
        providers.append(config)
        client.save_providers(providers)
        return True
        
    except Exception:
        return False

def delete_provider_config(provider_name: str) -> bool:
    """Delete a provider configuration.
    
    Args:
        provider_name: Name of the provider to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        providers = client.load_providers()
        original_count = len(providers)
        
        providers = [p for p in providers if p.name != provider_name]
        
        if len(providers) < original_count:
            client.save_providers(providers)
            return True
        return False
        
    except Exception:
        return False

def toggle_provider(provider_name: str, enabled: bool) -> bool:
    """Enable or disable a provider.
    
    Args:
        provider_name: Name of the provider to toggle
        enabled: Whether to enable or disable the provider
        
    Returns:
        True if successful, False otherwise
    """
    try:
        providers = client.load_providers()
        
        for i, provider in enumerate(providers):
            if provider.name == provider_name:
                providers[i].enabled = enabled
                client.save_providers(providers)
                return True
        
        return False
        
    except Exception:
        return False
