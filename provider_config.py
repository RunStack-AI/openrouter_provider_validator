"""OpenRouter Provider Validator - Provider Configuration

Manages provider-specific settings and routing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ProviderConfig:
    """Manages provider configuration settings."""
    
    @staticmethod
    def load_providers(filepath: str = "data/providers.json") -> List[Dict[str, Any]]:
        """Load provider configurations from a JSON file.
        
        Args:
            filepath: Path to the provider configuration file
            
        Returns:
            List of provider configurations
        """
        try:
            with open(filepath, "r") as f:
                providers = json.load(f)
                logger.info(f"Loaded {len(providers)} provider configurations")
                return providers
        except Exception as e:
            logger.error(f"Error loading provider configurations: {str(e)}")
            return []
    
    @staticmethod
    def save_providers(providers: List[Dict[str, Any]], filepath: str = "data/providers.json") -> bool:
        """Save provider configurations to a JSON file.
        
        Args:
            providers: List of provider configurations
            filepath: Path to save the configurations
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, "w") as f:
                json.dump(providers, f, indent=2)
                logger.info(f"Saved {len(providers)} provider configurations")
            return True
        except Exception as e:
            logger.error(f"Error saving provider configurations: {str(e)}")
            return False
    
    @staticmethod
    def get_provider(provider_id: str, filepath: str = "data/providers.json") -> Optional[Dict[str, Any]]:
        """Get a specific provider configuration by ID.
        
        Args:
            provider_id: ID of the provider to retrieve
            filepath: Path to the provider configuration file
            
        Returns:
            Provider configuration dictionary or None if not found
        """
        providers = ProviderConfig.load_providers(filepath)
        for provider in providers:
            if provider.get("id") == provider_id:
                return provider
        return None
    
    @staticmethod
    def update_provider(provider_id: str, updates: Dict[str, Any], filepath: str = "data/providers.json") -> bool:
        """Update a specific provider configuration.
        
        Args:
            provider_id: ID of the provider to update
            updates: Dictionary of fields to update
            filepath: Path to the provider configuration file
            
        Returns:
            True if successful, False otherwise
        """
        providers = ProviderConfig.load_providers(filepath)
        updated = False
        
        for i, provider in enumerate(providers):
            if provider.get("id") == provider_id:
                providers[i].update(updates)
                updated = True
                break
        
        if updated:
            return ProviderConfig.save_providers(providers, filepath)
        else:
            logger.error(f"Provider {provider_id} not found for update")
            return False
    
    @staticmethod
    def add_provider(provider_config: Dict[str, Any], filepath: str = "data/providers.json") -> bool:
        """Add a new provider configuration.
        
        Args:
            provider_config: Provider configuration to add
            filepath: Path to the provider configuration file
            
        Returns:
            True if successful, False otherwise
        """
        if "id" not in provider_config:
            logger.error("Cannot add provider without an id field")
            return False
        
        providers = ProviderConfig.load_providers(filepath)
        
        # Check if provider with same ID already exists
        for provider in providers:
            if provider.get("id") == provider_config["id"]:
                logger.error(f"Provider with ID {provider_config['id']} already exists")
                return False
        
        providers.append(provider_config)
        return ProviderConfig.save_providers(providers, filepath)
    
    @staticmethod
    def remove_provider(provider_id: str, filepath: str = "data/providers.json") -> bool:
        """Remove a provider configuration.
        
        Args:
            provider_id: ID of the provider to remove
            filepath: Path to the provider configuration file
            
        Returns:
            True if successful, False otherwise
        """
        providers = ProviderConfig.load_providers(filepath)
        original_count = len(providers)
        
        providers = [p for p in providers if p.get("id") != provider_id]
        
        if len(providers) == original_count:
            logger.error(f"Provider {provider_id} not found for removal")
            return False
        
        return ProviderConfig.save_providers(providers, filepath)
    
    @staticmethod
    def toggle_provider(provider_id: str, enabled: bool, filepath: str = "data/providers.json") -> bool:
        """Enable or disable a provider.
        
        Args:
            provider_id: ID of the provider to toggle
            enabled: Whether the provider should be enabled
            filepath: Path to the provider configuration file
            
        Returns:
            True if successful, False otherwise
        """
        return ProviderConfig.update_provider(provider_id, {"enabled": enabled}, filepath)
    
    @staticmethod
    def find_providers_for_model(model: str, enabled_only: bool = True, filepath: str = "data/providers.json") -> List[Dict[str, Any]]:
        """Find all providers that support a specific model.
        
        Args:
            model: Model identifier (e.g., 'moonshot/kimi-k2')
            enabled_only: Whether to return only enabled providers
            filepath: Path to the provider configuration file
            
        Returns:
            List of provider configurations supporting the model
        """
        providers = ProviderConfig.load_providers(filepath)
        matching_providers = []
        
        for provider in providers:
            # Skip disabled providers if enabled_only is True
            if enabled_only and not provider.get("enabled", True):
                continue
                
            # Check if model is in supported_models
            if "supported_models" in provider and model in provider.get("supported_models", []):
                matching_providers.append(provider)
        
        if matching_providers:
            logger.info(f"Found {len(matching_providers)} providers supporting model {model}")
        else:
            logger.warning(f"No providers found supporting model {model}")
            
        return matching_providers
    
    @staticmethod
    def get_default_provider_for_model(model: str, filepath: str = "data/providers.json") -> Optional[str]:
        """Get the default provider ID for a specific model.
        
        Returns the first enabled provider that supports the model,
        or None if no providers are available.
        
        Args:
            model: Model identifier (e.g., 'moonshot/kimi-k2')
            filepath: Path to the provider configuration file
            
        Returns:
            Provider ID or None if no providers found
        """
        matching_providers = ProviderConfig.find_providers_for_model(model, True, filepath)
        
        if matching_providers:
            # Return the first enabled provider's ID
            return matching_providers[0].get("id")
        
        return None