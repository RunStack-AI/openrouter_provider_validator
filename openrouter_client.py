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

class EndpointParameter(BaseModel):
    """Parameter configuration for an endpoint."""
    type: str = Field(..., description="Parameter type")
    description: Optional[str] = Field(None, description="Parameter description")
    enum: Optional[List[str]] = Field(None, description="Possible values for the parameter")
    default: Optional[Any] = Field(None, description="Default value for the parameter")

class EndpointInfo(BaseModel):
    """Information about a specific provider endpoint for a model."""
    id: str = Field(..., description="Provider+model ID")
    name: str = Field(..., description="Display name")
    organization: str = Field(..., description="Organization name")
    description: Optional[str] = Field(None, description="Description of the endpoint")
    pricing: Dict[str, float] = Field(..., description="Pricing per token, input/output")
    context_length: int = Field(..., description="Maximum context length")
    supported_parameters: Dict[str, EndpointParameter] = Field(..., description="Parameters supported by this endpoint")
    per_request_limits: Optional[Dict[str, Any]] = Field(None, description="Limits per request")
    benchmark: Optional[Dict[str, Any]] = Field(None, description="Benchmark data")

class ModelEndpointsResponse(BaseModel):
    """Response from the model endpoints API."""
    data: List[EndpointInfo] = Field(..., description="List of endpoint details")

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
                f"{self.BASE_URL}/model/{model_id}/endpoints",
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
                
            return ModelEndpointsResponse(**response.json())
    
    async def get_providers_for_model(self, model_id: str, tools_support_only: bool = True) -> List[Dict[str, Any]]:
        """Get all available providers for a specific model, optionally filtering for those that support tools.
        
        Args:
            model_id: The model ID in format "vendor/model"
            tools_support_only: If True, only return providers that support tools
            
        Returns:
            List of provider information dictionaries
        """
        endpoints = await self.get_model_endpoints(model_id)
        
        providers = []
        for endpoint in endpoints.data:
            # Check if endpoint supports tools if filter is enabled
            supports_tools = "tools" in endpoint.supported_parameters
            if tools_support_only and not supports_tools:
                continue
                
            # Extract provider ID from endpoint ID
            # Example: fireworks/moonshot/kimi-k2-fp8 -> fireworks
            provider_id = endpoint.id.split("/")[0] if "/" in endpoint.id else endpoint.id
            
            # Calculate metrics like latency if available
            latency_ms = None
            if endpoint.benchmark and "latencies" in endpoint.benchmark:
                latencies = endpoint.benchmark["latencies"]
                if latencies and isinstance(latencies, list) and len(latencies) > 0:
                    latency_ms = sum(latencies) / len(latencies)
            
            provider = {
                "id": provider_id,
                "endpoint_id": endpoint.id,
                "name": f"{endpoint.organization}",
                "description": endpoint.description or f"Provider for {model_id}",
                "context_length": endpoint.context_length,
                "pricing": endpoint.pricing,
                "supports_tools": supports_tools,
                "latency_ms": latency_ms
            }
            
            providers.append(provider)
            
        return providers

# Main function for testing
async def main():
    import asyncio
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
        print(f"Description: {provider['description']}")
        print(f"Context Length: {provider['context_length']}")
        print(f"Pricing: Input ${provider['pricing'].get('input', 0)}/1K tokens, Output ${provider['pricing'].get('output', 0)}/1K tokens")
        if provider['latency_ms']:
            print(f"Average Latency: {provider['latency_ms']:.2f}ms")
        print(f"Supports Tools: {provider['supports_tools']}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
