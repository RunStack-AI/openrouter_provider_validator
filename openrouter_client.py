#!/usr/bin/env python
"""
OpenRouter API Client for fetching model and provider information.
"""

import os
import json
from typing import Dict, List, Any, Optional
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class EndpointInfo(BaseModel):
    """Information about a specific provider endpoint for a model."""
    name: str = Field(..., description="Display name of the endpoint")
    context_length: int = Field(..., description="Maximum context length")
    pricing: Dict[str, Any] = Field(..., description="Pricing information")
    provider_name: str = Field(..., description="Name of the provider")
    tag: str = Field(..., description="Provider tag used in API requests")
    quantization: Optional[str] = Field(None, description="Model quantization info")
    max_completion_tokens: Optional[int] = Field(None, description="Maximum completion tokens")
    max_prompt_tokens: Optional[int] = Field(None, description="Maximum prompt tokens")
    supported_parameters: List[str] = Field(..., description="Parameters supported by this endpoint")
    status: int = Field(..., description="Provider status")
    uptime_last_30m: float = Field(..., description="Uptime percentage in last 30 minutes")

class ModelInfo(BaseModel):
    """Information about a model."""
    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model display name")
    created: int = Field(..., description="Creation timestamp")
    description: str = Field(..., description="Model description")
    architecture: Dict[str, Any] = Field(..., description="Architecture information")
    endpoints: List[EndpointInfo] = Field(..., description="List of endpoints for this model")

class ModelEndpointsResponse(BaseModel):
    """Response from the model endpoints API."""
    data: ModelInfo = Field(..., description="Model info with endpoints")

class OpenRouterClient:
    """Client for OpenRouter API to fetch model and provider information."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self):
        """Initialize with API key from environment."""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    async def get_model_endpoints(self, model_id: str) -> ModelEndpointsResponse:
        """Get all available endpoints for a specific model.
        
        Args:
            model_id: The model ID in format "vendor/model"
            
        Returns:
            ModelEndpointsResponse with endpoint details
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/models/{model_id}/endpoints",
                headers=self.headers
            )
            
            if response.status_code != 200:
                error_message = f"Error fetching model endpoints: {response.status_code}"
                try:
                    error_json = response.json()
                    error_message = f"{error_message} - {error_json.get('error', 'Unknown error')}"
                except Exception:
                    pass
                raise Exception(error_message)
                
            # Parse the response using the updated model structure
            return ModelEndpointsResponse(**response.json())
    
    async def get_providers_for_model(self, model_id: str, tools_support_only: bool = True) -> List[Dict[str, Any]]:
        """Get all available providers for a specific model, optionally filtering for those that support tools.
        
        Args:
            model_id: The model ID in format "vendor/model"
            tools_support_only: If True, only return providers that support tools
            
        Returns:
            List of provider information dictionaries
        """
        endpoints_response = await self.get_model_endpoints(model_id)
        model_info = endpoints_response.data
        
        providers = []
        for endpoint in model_info.endpoints:
            # Check if endpoint supports tools if filter is enabled
            supports_tools = "tools" in endpoint.supported_parameters
            if tools_support_only and not supports_tools:
                continue
            
            # Extract provider tag
            provider_tag = endpoint.tag
                
            # Calculate metrics like latency if available (would need to modify model to include benchmark data)
            latency_ms = None
            
            # Format pricing for input/output to maintain compatibility
            pricing = {
                "input": float(endpoint.pricing.get("prompt", 0)),
                "output": float(endpoint.pricing.get("completion", 0))
            }
            
            provider = {
                "id": provider_tag,  # Use the tag as the provider ID
                "endpoint_id": provider_tag,
                "name": f"{endpoint.provider_name}",
                "description": f"Provider for {model_id}",
                "context_length": endpoint.context_length,
                "pricing": pricing,
                "supports_tools": supports_tools,
                "latency_ms": latency_ms,
                "tag": provider_tag  # Add tag explicitly for clarity
            }
            
            providers.append(provider)
            
        return providers

# Main function for testing
async def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python openrouter_client.py <model_id>")
        return
        
    model_id = sys.argv[1]
    client = OpenRouterClient()
    
    providers = await client.get_providers_for_model(model_id)
    
    print(f"\nProviders supporting {model_id}:\n")
    for provider in providers:
        print(f"Provider: {provider['name']} (ID: {provider['id']})")
        print(f"Endpoint ID: {provider['endpoint_id']}")
        print(f"Tag: {provider['tag']}")
        print(f"Description: {provider['description']}")
        print(f"Context Length: {provider['context_length']}")
        print(f"Pricing: Input ${provider['pricing'].get('input', 0)}/1K tokens, Output ${provider['pricing'].get('output', 0)}/1K tokens")
        if provider['latency_ms']:
            print(f"Average Latency: {provider['latency_ms']:.2f}ms")
        print(f"Supports Tools: {provider['supports_tools']}")
        print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
