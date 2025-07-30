#!/usr/bin/env python
"""OpenRouter Provider Validator - Test Agent

CLI tool for running tests against the toy filesystem.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx
from dotenv import load_dotenv

from client import FileSystemClient
from mcp_server import MCPServer

# Load environment variables
load_dotenv()

# Configure logging
logging_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("agent")

class OpenRouterClient:
    """Client for interacting with the OpenRouter API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY environment variable)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        self.base_url = "https://openrouter.ai/api/v1"
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        model: str,
        provider: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a chat completion via OpenRouter.
        
        Args:
            messages: List of chat messages
            model: Model identifier (e.g., anthropic/claude-3-opus)
            provider: Optional provider routing override
            tools: Optional tool definitions
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            OpenRouter API response
        """
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/example/openrouter-validator",
            "X-Title": "OpenRouter Provider Validator",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tool_choice": "auto"
        }
        
        if provider:
            data["route"] = provider
            
        if tools:
            data["tools"] = tools
            
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        async with httpx.AsyncClient(timeout=120) as client:
            logger.info(f"Sending request to OpenRouter with model {model}")
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            return response_data

class ProviderTester:
    """Test runner for OpenRouter providers using the toy filesystem."""
    
    def __init__(self, model: str, provider: Optional[str] = None):
        """Initialize the tester.
        
        Args:
            model: Model identifier to test
            provider: Optional provider to route to
        """
        self.model = model
        self.provider = provider
        self.openrouter_client = OpenRouterClient()
        self.filesystem_client = FileSystemClient()
        self.mcp_server = MCPServer(self.filesystem_client)
    
    def _setup_test_files(self):
        """Setup toy filesystem for testing."""
        # Create test directories
        self.filesystem_client.create_folders(["data/test_files", "data/test_files/nested"])
        
        # Create sample files for testing
        sample_files = [
            ("data/test_files/sample1.txt", "This is sample file 1\nIt has multiple lines\nFor testing file reading operations."),
            ("data/test_files/sample2.txt", "Sample file 2 contains different content\nUseful for testing searching functionality."),
            ("data/test_files/nested/sample3.txt", "This is a nested file\nLocated in a subdirectory\nFor testing nested path operations.")
        ]
        
        for filepath, content in sample_files:
            self.filesystem_client.write_file(filepath, content)
        
        logger.info("Toy filesystem setup complete")
    
    async def run_test(self, prompt_id: str) -> Dict[str, Any]:
        """Run a single test with the specified prompt.
        
        Args:
            prompt_id: ID of the prompt to test with
            
        Returns:
            Test result dictionary
        """
        # Ensure test files exist
        self._setup_test_files()
        
        # Load prompt
        prompts = self.filesystem_client.load_prompts()
        prompt = next((p for p in prompts if p.get("id") == prompt_id), None)
        
        if not prompt:
            logger.error(f"Prompt {prompt_id} not found")
            return {"success": False, "error": f"Prompt {prompt_id} not found"}
        
        # Load system prompt from agent profile
        with open("agents/openrouter_validator.md", "r") as f:
            system_prompt = f.read()
        
        # Setup messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt["content"]}
        ]
        
        # Get tool definitions
        tools = self.mcp_server.get_tools()
        
        start_time = datetime.now()
        try:
            # Send request to OpenRouter
            response = await self.openrouter_client.chat_completion(
                messages=messages,
                model=self.model,
                provider=self.provider,
                tools=tools,
                temperature=0.2
            )
            
            # Check if tools were used
            tool_calls = []
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "tool_calls" in choice["message"]:
                    tool_calls = choice["message"]["tool_calls"]
            
            success = len(tool_calls) > 0
            
            # Create test result
            result = {
                "provider": self.provider or self.model.split("/")[0],
                "model": self.model,
                "prompt_id": prompt_id,
                "success": success,
                "response_data": response,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                    "tool_calls": len(tool_calls)
                }
            }
            
            # Add token usage if available 
            if "usage" in response:
                result["token_usage"] = response["usage"]
            
            if not success:
                result["error_message"] = "No tool calls in response"
                result["error_category"] = "tool_usage_error"
            
            return result
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            return {
                "provider": self.provider or self.model.split("/")[0],
                "model": self.model,
                "prompt_id": prompt_id,
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "error_category": "api_error"
            }
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all prompts as tests.
        
        Returns:
            List of test results
        """
        prompts = self.filesystem_client.load_prompts()
        results = []
        
        for prompt in prompts:
            logger.info(f"Running test with prompt {prompt['id']}")
            result = await self.run_test(prompt["id"])
            results.append(result)
        
        return results

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OpenRouter Provider Validator Test Agent")
    parser.add_argument("--model", default="anthropic/claude-3.7-sonnet", help="Model to test")
    parser.add_argument("--provider", help="Provider to route to (optional)")
    parser.add_argument("--prompt", help="Specific prompt ID to test (optional)")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    # Create tester
    tester = ProviderTester(model=args.model, provider=args.provider)
    
    # Run tests
    if args.prompt:
        logger.info(f"Running single test with prompt {args.prompt}")
        result = await tester.run_test(args.prompt)
        print(json.dumps(result, indent=2))
    elif args.all:
        logger.info("Running all tests")
        results = await tester.run_all_tests()
        
        # Save results
        tester.filesystem_client.save_test_results(args.model, results)
        
        # Print summary
        success_count = sum(1 for r in results if r["success"])
        print(f"Tests completed: {len(results)} total, {success_count} successful, {len(results) - success_count} failed")
    else:
        print("Please specify --prompt ID to run a single test or --all to run all tests")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
