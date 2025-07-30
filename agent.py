#!/usr/bin/env python
"""OpenRouter Provider Validator Test Agent.

This agent runs tests against various OpenRouter providers to assess their functionality
with file system operations.
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypedDict

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from client import FileSystemClient
from filesystem_test_helper import FileSystemTestHelper
from provider_config import ProviderConfig

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY must be set in environment or .env file")

# Constants for OpenRouter API
ROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ROUTER_SITE_URL = "https://provider-validator.example.com"
ROUTER_APP_TITLE = "OpenRouter Provider Validator"

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validator")

class ModelRequestParameters(BaseModel):
    """Parameters for a model request."""
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="List of tools to make available")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice strategy")

class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str = Field(..., description="Role of the message sender: system, user, assistant or tool")
    content: Optional[str] = Field(None, description="Content of the message")
    name: Optional[str] = Field(None, description="Name of the tool")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls made by the assistant")
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call this message is responding to")

class TestResults(TypedDict, total=False):
    """Results of a test run."""
    model: str
    provider: str  
    prompt_id: str
    success: bool
    total_steps: int
    successful_steps: int
    messages: List[Dict[str, Any]]
    metrics: Dict[str, Any]

class ProviderTester:
    """Test agent for evaluating OpenRouter providers."""
    
    def __init__(self, model="anthropic/claude-3.7-sonnet", provider=None):
        """Initialize the tester.
        
        Args:
            model: The model to test
            provider: Specific provider to use (optional)
        """
        self.model = model
        self.provider = provider
        self.test_helper = FileSystemTestHelper()
        self.test_helper.initialize_test_files()
        
        self.messages = []
        self.conversation = []
        self.tool_calls_count = 0
        self.send_count = 0
        self.total_latency = 0
        
    async def load_system_prompt(self) -> str:
        """Load the system prompt for the agent.
        
        Returns:
            System prompt string
        """
        system_prompt_file = Path("agents/openrouter_validator.md")
        if not system_prompt_file.exists():
            # Create default system prompt if file doesn't exist
            system_prompt = (
                "# OpenRouter Provider Validator Test Agent\n\n"
                "You are a test agent evaluating file system operations through tools. "  
                "Your task is to follow instructions exactly, making use of the available tools."
                "\n\nPlease make sure to carry out each step of the instructions completely and accurately."
                "\n\n"
                "DO NOT make assumptions about file contents, paths, or structures unless explicitly specified."
                "Always use the appropriate tools to verify information or make changes.\n"
            )
            os.makedirs(system_prompt_file.parent, exist_ok=True)
            with open(system_prompt_file, "w") as f:
                f.write(system_prompt)
        else:
            with open(system_prompt_file, "r") as f:
                system_prompt = f.read()
                
        return system_prompt
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the tool definitions for the model.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory to list files from"
                            }
                        },
                        "required": ["directory"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read content from a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write to"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "append_file",
                    "description": "Append content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to append to"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to append to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "description": "Create a new directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Path of the directory to create"
                            }
                        },
                        "required": ["directory"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "copy_file",
                    "description": "Copy a file to a new location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Source file path"
                            },
                            "destination": {
                                "type": "string",
                                "description": "Destination file path"
                            }
                        },
                        "required": ["source", "destination"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_file",
                    "description": "Move a file to a new location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Source file path"
                            },
                            "destination": {
                                "type": "string",
                                "description": "Destination file path"
                            }
                        },
                        "required": ["source", "destination"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for content in files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory to search in"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Pattern to search for"
                            }
                        },
                        "required": ["directory", "pattern"]
                    }
                }
            }
        ]
    
    async def process_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Process a tool call and return the result.
        
        Args:
            tool_call: Tool call from the assistant
            
        Returns:
            Tool response message
        """
        function = tool_call.get("function", {})
        name = function.get("name")
        arguments = function.get("arguments", "{}")
        
        # Ensure arguments is a dictionary
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": name,
                    "content": "Error: Invalid arguments JSON"
                }
        
        self.tool_calls_count += 1
        result = ""
        
        try:
            if name == "list_files":
                directory = arguments.get("directory", ".")
                result = self.test_helper.list_files(directory)
            elif name == "read_file":
                file_path = arguments.get("file_path")
                if not file_path:
                    raise ValueError("file_path is required")
                result = self.test_helper.read_file(file_path)
            elif name == "write_file":
                file_path = arguments.get("file_path")
                content = arguments.get("content", "")
                if not file_path:
                    raise ValueError("file_path is required")
                self.test_helper.write_file(file_path, content)
                result = f"File written successfully to {file_path}"
            elif name == "append_file":
                file_path = arguments.get("file_path")
                content = arguments.get("content", "")
                if not file_path:
                    raise ValueError("file_path is required")
                self.test_helper.append_file(file_path, content)
                result = f"Content appended successfully to {file_path}"
            elif name == "create_directory":
                directory = arguments.get("directory")
                if not directory:
                    raise ValueError("directory is required")
                self.test_helper.create_directory(directory)
                result = f"Directory created successfully at {directory}"
            elif name == "copy_file":
                source = arguments.get("source")
                destination = arguments.get("destination")
                if not source or not destination:
                    raise ValueError("source and destination are required")
                self.test_helper.copy_file(source, destination)
                result = f"File copied successfully from {source} to {destination}"
            elif name == "move_file":
                source = arguments.get("source")
                destination = arguments.get("destination")
                if not source or not destination:
                    raise ValueError("source and destination are required")
                self.test_helper.move_file(source, destination)
                result = f"File moved successfully from {source} to {destination}"
            elif name == "search_files":
                directory = arguments.get("directory", ".")
                pattern = arguments.get("pattern", "")
                if not pattern:
                    raise ValueError("pattern is required")
                result = self.test_helper.search_files(directory, pattern)
            else:
                result = f"Unknown tool: {name}"
        except Exception as e:
            result = f"Error: {str(e)}"
        
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "name": name,
            "content": result
        }
    
    async def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the model and process the response.
        
        Args:
            message: User message content
            
        Returns:
            Model response
        """
        user_message = {"role": "user", "content": message}
        self.conversation.append(user_message)
        self.messages.append(user_message)
        
        # Prepare request parameters
        params = ModelRequestParameters(
            tools=self.get_tools(),
            tool_choice="auto"
        )
        
        # Construct the request
        request_data = {
            "messages": self.conversation,
            "model": self.model,
            "response_format": {"type": "json_object"},
        }
        
        # Add tools to request
        if params.tools:
            request_data["tools"] = params.tools
        if params.tool_choice:
            request_data["tool_choice"] = params.tool_choice
            
        # Add provider if specified
        if self.provider:
            request_data["transforms"] = [{"type": "provider_filter", "providers": [self.provider]}]
        
        # Additional HTTP headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": ROUTER_SITE_URL,
            "X-Title": ROUTER_APP_TITLE
        }
        
        # Send the request and measure latency
        start_time = time.time()
        self.send_count += 1
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=request_data,
                timeout=120.0
            )
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        self.total_latency += latency
        
        # Process the response
        response_data = response.json()
        
        # Extract provider info and update provider attribute if not explicitly set
        if "provider" in response_data and not self.provider:
            self.provider = response_data["provider"]
        
        # Extract the assistant's message
        assistant_message = response_data.get("choices", [{}])[0].get("message", {})
        self.conversation.append(assistant_message)
        self.messages.append(assistant_message)
        
        # Check if the assistant wants to use tools
        tool_calls = assistant_message.get("tool_calls", [])
        if tool_calls:
            # Process each tool call
            for tool_call in tool_calls:
                tool_response = await self.process_tool_call(tool_call)
                self.conversation.append(tool_response)
                self.messages.append(tool_response)
        
        return response_data
    
    async def run_test(self, prompt_id: str = "file_operations_sequence") -> TestResults:
        """Run a specific test sequence.
        
        Args:
            prompt_id: ID of the prompt sequence to run
            
        Returns:
            TestResults dictionary
        """
        # Reset conversation for this test
        self.conversation = []
        self.messages = []
        self.tool_calls_count = 0
        self.send_count = 0
        self.total_latency = 0
        
        # Re-initialize the test environment
        self.test_helper.initialize_test_files()
        
        # Load the system prompt
        system_prompt = await self.load_system_prompt()
        system_message = {"role": "system", "content": system_prompt}
        self.conversation.append(system_message)
        
        # Load the prompt sequence
        prompt_sequence = self.test_helper.load_prompt_sequence(prompt_id)
        if not prompt_sequence:
            return {
                "model": self.model,
                "provider": self.provider or "unknown",
                "prompt_id": prompt_id,
                "success": False,
                "total_steps": 0,
                "successful_steps": 0,
                "messages": [],
                "metrics": {
                    "total_tool_calls": 0,
                    "total_send_count": 0,
                    "latency_ms": 0
                }
            }
        
        logger.info(f"Running test {prompt_id} with {len(prompt_sequence['sequence'])} steps")
        
        # Run each step in the sequence
        successful_steps = 0
        for i, step in enumerate(prompt_sequence["sequence"]):
            logger.info(f"Running step {i+1}/{len(prompt_sequence['sequence'])}")
            try:
                await self.send_message(step)
                successful_steps += 1
            except Exception as e:
                logger.error(f"Error in step {i+1}: {str(e)}")
                break
        
        # Calculate metrics
        avg_latency = self.total_latency / self.send_count if self.send_count > 0 else 0
        
        # Save test results
        result = {
            "model": self.model,
            "provider": self.provider or "unknown",
            "prompt_id": prompt_id,
            "success": successful_steps == len(prompt_sequence["sequence"]),
            "total_steps": len(prompt_sequence["sequence"]),
            "successful_steps": successful_steps,
            "messages": self.messages,
            "metrics": {
                "total_tool_calls": self.tool_calls_count,
                "total_send_count": self.send_count,
                "latency_ms": avg_latency
            }
        }
        
        # Save result to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path("results")
        result_dir.mkdir(exist_ok=True)
        
        provider_suffix = f"_{self.provider}" if self.provider else ""
        result_file = result_dir / f"{self.model.replace('/', '_')}_{prompt_id}{provider_suffix}_{timestamp}.json"
        
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Test results saved to {result_file}")
        return result

async def list_providers_for_model(model: str):
    """List all available providers for a model.
    
    Args:
        model: Model identifier
    """
    providers = await ProviderConfig.find_providers_for_model(model, enabled_only=False)
    
    print(f"\nAvailable providers for {model}:\n")
    if not providers:
        print("No specific providers configured for this model. Default routing will be used.")
    else:
        for provider in providers:
            status = "Enabled" if provider.get("enabled", True) else "Disabled"
            print(f"Provider: {provider['name']} (ID: {provider['id']}) - {status}")
            if "endpoint_id" in provider:
                print(f"  Endpoint ID: {provider['endpoint_id']}")
            print(f"  {provider.get('description', '')}")
            
            if "context_length" in provider:
                print(f"  Context Length: {provider['context_length']}")
            if "pricing" in provider and isinstance(provider['pricing'], dict):
                print(f"  Pricing: Input ${provider['pricing'].get('input', 0):.2f}/1K tokens, "
                      f"Output ${provider['pricing'].get('output', 0):.2f}/1K tokens")
            if provider.get("latency_ms") is not None:
                print(f"  Average Latency: {provider['latency_ms']:.2f}ms")
            print()

async def main():
    parser = argparse.ArgumentParser(description="OpenRouter Provider Validator Test Agent")
    parser.add_argument("--model", default="anthropic/claude-3.7-sonnet", help="Model to test")
    parser.add_argument("--provider", help="Specific provider to use")
    parser.add_argument("--prompt", default="file_operations_sequence", help="Prompt sequence ID to test")
    parser.add_argument("--all", action="store_true", help="Run all prompt sequences")
    parser.add_argument("--list-providers", action="store_true", help="List available providers for the model")
    args = parser.parse_args()
    
    # Special case: list providers
    if args.list_providers:
        await list_providers_for_model(args.model)
        return
    
    tester = ProviderTester(model=args.model, provider=args.provider)
    
    # Get provider info to display
    if args.provider:
        provider_name = args.provider
        provider_info = await ProviderConfig.get_provider(args.provider)
        if provider_info:
            provider_name = provider_info.get("name", args.provider)
    elif args.model:
        # Try to auto-detect provider
        default_provider_id = await ProviderConfig.get_default_provider_for_model(args.model)
        if default_provider_id:
            provider_info = await ProviderConfig.get_provider(default_provider_id)
            provider_name = provider_info.get("name", default_provider_id) if provider_info else default_provider_id
            print(f"Using auto-detected provider: {provider_name} (ID: {default_provider_id})")
        else:
            provider_name = "OpenRouter (default routing)"
            print("No specific provider detected, using default routing")
    else:
        provider_name = "OpenRouter (default routing)"
    
    print(f"Testing {args.model} using {provider_name}")
    
    if args.all:
        # Run all prompt sequences
        test_helper = FileSystemTestHelper()
        prompts = test_helper.load_prompts()
        for prompt in prompts:
            print(f"\nRunning test: {prompt['id']}")
            await tester.run_test(prompt["id"])
    else:
        # Run specified prompt sequence
        await tester.run_test(args.prompt)

if __name__ == "__main__":
    asyncio.run(main())
