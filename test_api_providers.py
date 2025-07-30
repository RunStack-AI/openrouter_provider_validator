#!/usr/bin/env python

"""Test script to verify the OpenRouter API provider retrieval.

This script tests the new functionality to fetch provider information directly from the OpenRouter API
instead of relying on static configuration in data/providers.json.
"""

import asyncio
from openrouter_client import OpenRouterClient
from provider_config import ProviderConfig

async def main():
    # Test models
    models = [
        "moonshot/kimi-k2",
        "anthropic/claude-3.7-sonnet",
        "mistralai/mistral-large"
    ]
    
    print("\n=== Testing Direct API Client ===\n")
    
    # Test the OpenRouterClient directly
    client = OpenRouterClient()
    for model in models:
        print(f"\nProviders for {model} via direct API:")
        try:
            providers = await client.get_providers_for_model(model)
            if providers:
                for provider in providers:
                    print(f"  - {provider['name']} (ID: {provider['id']})")
                    print(f"    Endpoint ID: {provider['endpoint_id']}")
                    print(f"    Context Length: {provider['context_length']}")
                    print(f"    Supports Tools: {provider.get('supports_tools', False)}")
                    if provider.get('latency_ms') is not None:
                        print(f"    Average Latency: {provider['latency_ms']:.2f}ms")
                    if provider.get('pricing'):
                        in_price = provider['pricing'].get('input', 0)
                        out_price = provider['pricing'].get('output', 0)
                        print(f"    Pricing: Input ${in_price}/1K tokens, Output ${out_price}/1K tokens")
                    print()
            else:
                print("  No providers returned from API")
        except Exception as e:
            print(f"  Error fetching providers: {e}")
            
    print("\n=== Testing ProviderConfig Interface ===\n")
    
    # Test the ProviderConfig interface
    for model in models:
        print(f"\nProviders for {model} via ProviderConfig:")
        try:
            providers = await ProviderConfig.find_providers_for_model(model, enabled_only=False)
            if providers:
                for provider in providers:
                    print(f"  - {provider['name']} (ID: {provider['id']})")
                    if "endpoint_id" in provider:
                        print(f"    Endpoint ID: {provider['endpoint_id']}")
                    print(f"    Description: {provider.get('description', 'No description')}")
                    if "context_length" in provider:
                        print(f"    Context Length: {provider['context_length']}")
                    print(f"    Supports Tools: {provider.get('supports_tools', True)}")
                    
                    if provider.get('latency_ms') is not None:
                        print(f"    Average Latency: {provider['latency_ms']:.2f}ms")
                    
                    if 'pricing' in provider and isinstance(provider['pricing'], dict):
                        in_price = provider['pricing'].get('input', 0)
                        out_price = provider['pricing'].get('output', 0)
                        print(f"    Pricing: Input ${in_price}/1K tokens, Output ${out_price}/1K tokens")
                    print()
            else:
                print("  No providers found")
                
            # Get default provider
            default_provider = await ProviderConfig.get_default_provider_for_model(model)
            if default_provider:
                print(f"  Default provider for {model}: {default_provider}")
            else:
                print(f"  No default provider found for {model}")
        except Exception as e:
            print(f"  Error in ProviderConfig: {e}")

if __name__ == "__main__":
    asyncio.run(main())
