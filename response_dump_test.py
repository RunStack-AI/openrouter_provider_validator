#!/usr/bin/env python
"""Simple script to dump the raw JSON response from the OpenRouter API for debugging."""

import asyncio
import httpx
import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()

async def get_raw_response(model_id):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is required")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    print(f"HTTP Request: GET {url}")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        
        print(f"Status: {response.status_code} {response.reason_phrase}")
        print("Headers:")
        for k, v in response.headers.items():
            print(f"  {k}: {v}")
        
        if response.status_code == 200:
            data = response.json()
            print("\nResponse data (raw):")
            print(json.dumps(data, indent=2))
            
            # Check for specifically null uptime_last_30m values
            if "data" in data and "endpoints" in data["data"]:
                for i, endpoint in enumerate(data["data"]["endpoints"]):
                    if "uptime_last_30m" in endpoint and endpoint["uptime_last_30m"] is None:
                        print(f"\nFOUND NULL uptime_last_30m in endpoint {i}:")
                        print(json.dumps(endpoint, indent=2))
        else:
            print(f"Error: {response.text}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python response_dump_test.py <model_id>")
        return
        
    model_id = sys.argv[1]
    asyncio.run(get_raw_response(model_id))

if __name__ == "__main__":
    main()