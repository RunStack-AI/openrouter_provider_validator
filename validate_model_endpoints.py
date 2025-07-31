#!/usr/bin/env python
"""
Validation script for the OpenRouter model endpoints response handling.

This script tests the updated ModelEndpointsResponse model against the sample response.
"""

import json
import asyncio
from pathlib import Path
from openrouter_client import ModelEndpointsResponse, OpenRouterClient

async def test_sample_json():
    """Test parsing the sample JSON response."""
    sample_file = Path("sample_model_endpoints_response.json")
    if not sample_file.exists():
        print(f"Error: Sample file {sample_file} not found")
        return
        
    print(f"Testing with sample data from {sample_file}")
    
    try:
        with open(sample_file, "r") as f:
            sample_data = json.load(f)
            
        # Parse with the updated model
        response = ModelEndpointsResponse(**sample_data)
        
        print("\nSuccessfully parsed the sample response!")
        print(f"Model ID: {response.data.id}")
        print(f"Name: {response.data.name}")
        print(f"Number of endpoints: {len(response.data.endpoints)}\n")
        
        # Print details about each endpoint
        for i, endpoint in enumerate(response.data.endpoints):
            print(f"Endpoint {i+1}: {endpoint.name}")
            print(f"  Provider: {endpoint.provider_name}")
            print(f"  Tag: {endpoint.tag}")
            print(f"  Context Length: {endpoint.context_length}")
            print(f"  Supported Parameters: {', '.join(endpoint.supported_parameters[:5])}{'...' if len(endpoint.supported_parameters) > 5 else ''}")
            print(f"  Status: {endpoint.status}")
            print(f"  Uptime: {endpoint.uptime_last_30m}%")
            print()
            
        # Test the provider extraction function
        client = OpenRouterClient()
        providers = await client.get_providers_for_model(response.data.id, tools_support_only=False)
        
        print("Extracted provider information:")
        for provider in providers:
            print(f"Provider: {provider['name']} (ID: {provider['id']})")
            print(f"  Tag: {provider['tag']}")
            print(f"  Supports Tools: {provider['supports_tools']}")
            print()
            
    except Exception as e:
        print(f"Error parsing sample data: {e}")

async def main():
    await test_sample_json()
    
    # If API key is available, test with live API
    import os
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print("\nAPI key found! Testing with live API...")
        try:
            client = OpenRouterClient()
            model_id = "moonshot/kimi-k2"  # Use one of the models known to have multiple providers
            print(f"Fetching endpoints for {model_id}...")
            endpoints = await client.get_model_endpoints(model_id)
            
            print(f"Got {len(endpoints.data.endpoints)} endpoints!")
            
            # Test provider extraction
            providers = await client.get_providers_for_model(model_id, tools_support_only=False)
            print(f"\nExtracted {len(providers)} providers from API:")
            for provider in providers:
                print(f"Provider: {provider['name']} (ID: {provider['id']})")
                print(f"  Tag: {provider['tag']}")
                print(f"  Supports Tools: {provider['supports_tools']}")
                print()
                
        except Exception as e:
            print(f"Error testing with live API: {e}")
    else:
        print("\nNo API key found. Skipping live API test.")

if __name__ == "__main__":
    asyncio.run(main())
