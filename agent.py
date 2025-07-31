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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypedDict

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import PydanticAI components for Agent framework
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessage, SystemPromptPart, UserPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import AgentRunResult

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

# Try to import logfire for metrics if available
try:
    import logfire
    logfire.configure(token=os.getenv("LOGFIRE_API_KEY"))
    logfire.instrument_openai()
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logger.warning("logfire not installed, skipping instrumentation")

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
        
        # Message history tracking
        self.messages = []
        self.conversation = []
        
        # Metrics
        self.tool_calls_count = 0
        self.send_count = 0
        self.total_latency = 0
        
        # Set up OpenRouter based model
        self.openai_model = None
        
        # Set up MCP Server environment variables
        self.mcp_env = {
            # Add any environment variables needed by the MCP server
        }
        
        # We'll initialize the actual server in the run_test method
        self.mcp_servers = None
        self.agent = None
        
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
                
        # Replace any time variables
        time_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = system_prompt.replace('{time_now}', time_now)
                
        return system_prompt
    
    def initialize_agent(self, system_prompt):
        """Initialize the Agent with MCP server.
        
        Args:
            system_prompt: System prompt for the agent
        """
        # Set up the MCP server for tools
        self.mcp_servers = [
            MCPServerStdio('python', ['./mcp_server.py'], env=self.mcp_env)
        ]

        from pydantic_ai.settings import ModelSettings
        model_settings = ModelSettings(provider=self.provider)
        self.openai_model = OpenAIModel(
            self.model,
            provider=OpenAIProvider(
                base_url='https://openrouter.ai/api/v1',
                api_key=api_key
            ),
            settings=model_settings
        )
        
        # If provider is specified, add it to the transforms
        # provider_transforms = None
        # if self.provider:
        #     provider_transforms = [{"type": "provider_filter", "providers": [self.provider]}]
        #     # Add this to the model config
        #     self.openai_model.default_options["transforms"] = provider_transforms
            
        # Create the agent
        self.agent = Agent(self.openai_model, mcp_servers=self.mcp_servers, system_prompt=system_prompt)
        
    # Function to filter message history for the agent
    def filtered_message_history(self, result: Optional[AgentRunResult], limit: Optional[int] = None):
        """Filter and limit the message history from an AgentRunResult.
        
        Args:
            result: The AgentRunResult object with message history
            limit: Optional int, if provided returns only system message + last N messages
            
        Returns:
            Filtered list of messages in the format expected by the agent
        """
        if result is None:
            return None
            
        # Get all messages
        messages = result.all_messages()
        
        # Extract system message
        system_message = next((msg for msg in messages if type(msg.parts[0]) == SystemPromptPart), None)
        
        # Filter non-system messages
        non_system_messages = [msg for msg in messages if type(msg.parts[0]) != SystemPromptPart]
        
        # Apply limit if specified
        if limit is not None and limit > 0 and len(non_system_messages) > limit:
            non_system_messages = non_system_messages[-limit:]
        
        # Combine system message with other messages
        result_messages = []
        if system_message:
            result_messages.append(system_message)
        result_messages.extend(non_system_messages)
        
        return result_messages
    
    async def send_message(self, message: str, result: Optional[AgentRunResult] = None) -> Dict[str, Any]:
        """Send a message to the agent and process the response.
        
        Args:
            message: User message content
            result: Previous AgentRunResult for conversation history
            
        Returns:
            Response data including messages and metrics
        """
        start_time = time.time()
        self.send_count += 1
        
        # Run the agent
        try:
            agent_result = await self.agent.run(
                message,
                message_history=self.filtered_message_history(result, limit=24)
            )
            
            # Extract all messages including tool calls
            all_messages = agent_result.all_messages()
            
            # Count tool calls
            for msg in all_messages:
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        self.tool_calls_count += 1
            
            # Add to conversation history
            self.messages.extend(all_messages)
            
            # Calculate latency
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            self.total_latency += latency
            
            # Extract provider info from response if available
            response_json = agent_result.new_messages()[-1] if agent_result.new_messages() else {}
            if isinstance(response_json, dict) and "provider" in response_json and not self.provider:
                self.provider = response_json.get("provider")
            
            return {
                "result": agent_result,
                "output": agent_result.output,
                "messages": all_messages,
                "latency_ms": latency
            }
            
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return {
                "error": str(e),
                "messages": [],
                "latency_ms": 0
            }
    
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
        
        # Initialize agent with system prompt
        self.initialize_agent(system_prompt)
        
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
        
        # Run each step in the sequence through the agent
        successful_steps = 0
        previous_result = None
        
        # Start the MCP servers
        async with self.agent.run_mcp_servers():
            for i, step in enumerate(prompt_sequence["sequence"]):
                logger.info(f"Running step {i+1}/{len(prompt_sequence['sequence'])}")
                try:
                    response = await self.send_message(step, previous_result)
                    if "error" in response:
                        logger.error(f"Error in step {i+1}: {response['error']}")
                        break
                    
                    previous_result = response.get("result")
                    successful_steps += 1
                except Exception as e:
                    logger.error(f"Error in step {i+1}: {str(e)}")
                    break
        
        # Calculate metrics
        avg_latency = self.total_latency / self.send_count if self.send_count > 0 else 0
        
        # Extract messages in serializable format
        serialized_messages = []
        for msg in self.messages:
            if hasattr(msg, 'dict'):
                serialized_messages.append(msg.dict())
            elif hasattr(msg, 'model_dump'):
                serialized_messages.append(msg.model_dump())
            else:
                # Attempt to serialize based on role/content pattern
                serialized_msg = {}
                if hasattr(msg, 'role'):
                    serialized_msg['role'] = msg.role
                if hasattr(msg, 'content'):
                    serialized_msg['content'] = msg.content
                serialized_messages.append(serialized_msg)
        
        # Save test results
        result = {
            "model": self.model,
            "provider": self.provider or "unknown",
            "prompt_id": prompt_id,
            "success": successful_steps == len(prompt_sequence["sequence"]),
            "total_steps": len(prompt_sequence["sequence"]),
            "successful_steps": successful_steps,
            "messages": serialized_messages,
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
        
        # Create a filename-safe version of model and provider for the path
        model_safe = self.model.replace('/', '_')
        provider_suffix = f"_{self.provider}" if self.provider else ""
        
        # Determine file path; handle both nested and flat formats
        if '/' in self.model or '\\' in self.model or result_dir.is_dir():
            # If we need a subdirectory approach (error suggests this)
            subdir = result_dir / f"{model_safe}_{prompt_id}{provider_suffix}"
            subdir.mkdir(exist_ok=True, parents=True)
            result_file = subdir / f"{self.provider or 'default'}_{timestamp}.json"
            logger.info(f"Saving results to subdirectory: {subdir}")
        else:
            # Flat approach for backward compatibility
            result_file = result_dir / f"{model_safe}_{prompt_id}{provider_suffix}_{timestamp}.json"
        
        # Ensure all parent directories exist
        result_file.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Test results saved to {result_file}")
        except Exception as e:
            logger.error(f"Failed to save results file: {e}")
        
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
