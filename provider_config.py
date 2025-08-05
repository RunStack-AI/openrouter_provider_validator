#!/usr/bin/env python
"""Provider configuration management for OpenRouter models."""

import json
import os
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

from openrouter_client import OpenRouterClient

class ProviderConfig:
    """Configuration for OpenRouter providers."""
    
    _cached_providers = {}
    _client = None
    
    @classmethod
    def get_client(cls):
        """Get or create the OpenRouter client."""
        if cls._client is None:
            cls._client = OpenRouterClient()
        return cls._client
    
    @classmethod
    def get_providers(cls, refresh_cache=False) -> List[Dict[str, Any]]:
        """Get all configured provider definitions.
        
        For backward compatibility, this will attempt to load from data/providers.json,
        but the API-based dynamic loading is preferred.
        
        Args:
            refresh_cache: Force refresh of the cache
            
        Returns:
            List of provider definitions
        """
        if not cls._cached_providers or refresh_cache:
            providers_file = Path("data/providers.json")
            if providers_file.exists():
                try:
                    with open(providers_file, "r") as f:
                        cls._cached_providers["static"] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load providers from file: {e}")
                    cls._cached_providers["static"] = []
            else:
                # If JSON doesn't exist, initialize with empty list
                cls._cached_providers["static"] = []
        
        return cls._cached_providers.get("static", [])
    
    @classmethod
    def get_provider(cls, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get a provider by ID from the static configuration.
        
        Args:
            provider_id: The ID of the provider to retrieve
            
        Returns:
            Provider configuration dictionary or None if not found
        """
        if not provider_id:
            return None
            
        providers = cls.get_providers()
        for provider in providers:
            if provider.get("id") == provider_id:
                return provider
        
        return None
    
    @classmethod
    async def fetch_providers_for_model(cls, model_id: str) -> List[Dict[str, Any]]:
        """Fetch provider information from the OpenRouter API.
        
        Args:
            model_id: The model ID to retrieve providers for
            
        Returns:
            List of providers supporting the model
        """
        # Check if we have a cached result
        cache_key = f"api_{model_id}"
        if cache_key in cls._cached_providers:
            print(f"Cached: {cache_key}")
            return cls._cached_providers[cache_key]
        
        try:
            # Fetch providers using the API client
            client = cls.get_client()
            providers = await client.get_providers_for_model(model_id, tools_support_only=True)
            print(providers)
            
            # Cache the results
            cls._cached_providers[cache_key] = providers
            
            return providers
        except Exception as e:
            print(f"Warning: Could not fetch providers from API: {e}")
            # Fall back to file if API fails
            return cls.find_providers_for_model_from_file(model_id)
    
    @classmethod
    def find_providers_for_model_from_file(cls, model_id: str, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Find providers for a specific model from the static file configuration.
        
        Args:
            model_id: The model ID to find providers for
            enabled_only: Only return enabled providers
            
        Returns:
            List of provider configurations
        """
        providers = cls.get_providers()
        model_providers = []
        
        for provider in providers:
            supported_models = provider.get("supported_models", [])
            if model_id in supported_models:
                if enabled_only and provider.get("enabled", True) is False:
                    continue
                model_providers.append(provider)
        
        return model_providers
    
    @classmethod
    async def find_providers_for_model(cls, model_id: str, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Find providers for a specific model, preferring API results over file.
        
        Args:
            model_id: The model ID to find providers for
            enabled_only: Only return enabled providers
            
        Returns:
            List of provider configurations
        """
        try:
            # First try to get providers from API
            api_providers = await cls.fetch_providers_for_model(model_id)
            print(model_id)
            print(api_providers)
            print(enabled_only)
            if api_providers:
                if enabled_only:
                    # Filter out any providers explicitly disabled in static config
                    static_providers = cls.get_providers()
                    disabled_ids = [p.get("id") for p in static_providers if p.get("enabled") is False]
                    return [p for p in api_providers if p.get("id") not in disabled_ids]
                return api_providers
        except Exception:
            pass
        
        # Fall back to file-based lookup
        return cls.find_providers_for_model_from_file(model_id, enabled_only)
    
    @classmethod
    async def get_default_provider_for_model(cls, model_id: str) -> Optional[str]:
        """Get the default provider ID for a specific model.
        
        Args:
            model_id: The model ID to get the default provider for
            
        Returns:
            Provider ID string or None if no providers found
        """
        providers = await cls.find_providers_for_model(model_id)
        if not providers:
            return None
            
        # Prioritize by latency if available
        providers_with_latency = [p for p in providers if p.get("latency_ms") is not None]
        if providers_with_latency:
            # Sort by latency (lowest first)
            providers_with_latency.sort(key=lambda p: p.get("latency_ms", float('inf')))
            return providers_with_latency[0].get("id")
        
        # Otherwise, just return the first provider's ID
        return providers[0].get("id")

# Function to run async methods for testing
def run_async(coro):
    """Run an async function synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)

# Main function for testing
def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python provider_config.py <model_id>")
        return
        
    model_id = sys.argv[1]
    
    # Get providers for the model
    providers = run_async(ProviderConfig.find_providers_for_model(model_id, enabled_only=False))
    
    print(f"\nProviders for {model_id}:\n")
    for provider in providers:
        print(f"Provider: {provider['name']} (ID: {provider['id']})")
        if "endpoint_id" in provider:
            print(f"Endpoint ID: {provider['endpoint_id']}")
        print(f"Description: {provider.get('description', 'No description')}")
        if "context_length" in provider:
            print(f"Context Length: {provider['context_length']}")
        if "pricing" in provider and isinstance(provider['pricing'], dict):
            print(f"Pricing: Input ${provider['pricing'].get('input', 0)}/1K tokens, "
                  f"Output ${provider['pricing'].get('output', 0)}/1K tokens")
        if provider.get("latency_ms") is not None:
            print(f"Average Latency: {provider['latency_ms']:.2f}ms")
        print(f"Supports Tools: {provider.get('supports_tools', True)}")
        print()
    
    # Get default provider
    default_provider = run_async(ProviderConfig.get_default_provider_for_model(model_id))
    if default_provider:
        print(f"Default provider for {model_id}: {default_provider}")

if __name__ == "__main__":
    main()
