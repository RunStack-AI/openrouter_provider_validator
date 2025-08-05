#!/usr/bin/env python
"""OpenRouter Provider Validator Test Agent.

This agent runs tests against various OpenRouter providers to assess their functionality
with file system operations.
"""

import argparse
import asyncio
import json
import os
import shutil
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
from pydantic_ai.messages import ModelResponsePart

from client import FileSystemClient, TestResult
from filesystem_test_helper import FileSystemTestHelper
from provider_config import ProviderConfig
from serialization_helper import json_serializable

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

from tracelight import log_exception_state

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
    
    def __init__(self, model="anthropic/claude-3.7-sonnet", provider=None, test_files_dir=None):
        """Initialize the tester.
        
        Args:
            model: The model to test
            provider: Specific provider to use (optional)
            test_files_dir: Directory for test files (optional, provider-specific)
        """
        self.model = model
        self.provider = provider
        
        # If test_files_dir is provided, use it; otherwise use default
        self.test_files_dir = test_files_dir
        self.test_helper = FileSystemTestHelper(test_files_dir=self.test_files_dir)
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
        
        # Set up MCP Server environment variables with custom test files path if available
        self.mcp_env = {}
        if self.test_files_dir:
            self.mcp_env["TEST_FILES_DIR"] = str(self.test_files_dir)
        
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
            
            # Look for validation errors in the response
            validation_errors = []
            for msg in all_messages:
                for part in msg.parts:
                    # Check for validation errors in any part that contains error messages
                    # regardless of the specific part type
                    has_error = False
                    error_content = ""
                    
                    # Check if it's a ToolReturnPart with error content
                    if isinstance(part, ToolReturnPart) and hasattr(part, 'content') and "Error executing tool" in str(part.content):
                        has_error = True
                        error_content = str(part.content).lower()
                    
                    # Check if it's a retry-prompt part with error content 
                    elif (hasattr(part, 'part_kind') and part.part_kind == "retry-prompt" and 
                          hasattr(part, 'content') and "Error executing tool" in str(part.content)):
                        has_error = True
                        error_content = str(part.content).lower()
                    
                    # Process the error if we found one
                    if has_error and error_content and any(pattern in error_content for pattern in [
                            "validation error", 
                            "errors.pydantic.dev", 
                            "type=model_type", 
                            "type=value_error", 
                            "type=type_error", 
                            "type=missing",
                            "field required", 
                            "input should be",
                            "not valid"
                        ]):
                        # Extract the tool name from either attribute or JSON content
                        tool_name = "unknown"
                        if hasattr(part, 'tool_name'):
                            tool_name = part.tool_name
                        elif hasattr(part, 'content') and isinstance(part.content, dict) and 'tool_name' in part.content:
                            tool_name = part.content['tool_name']
                        
                        validation_errors.append({
                            "message": part.content if hasattr(part, 'content') else str(part),
                            "tool": tool_name,
                            "timestamp": datetime.now().isoformat()
                        })
                        logger.info(f"Detected validation error in tool: {tool_name}")
            
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
                "latency_ms": latency,
                "validation_errors": validation_errors
            }
            
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            trace = log_exception_state(e, logger)
            return {
                "error": str(e),
                "messages": [],
                "latency_ms": 0,
                "validation_errors": [],
                "debug": trace
            }
    
    async def run_test(self, prompt_id: str = "file_operations_sequence") -> TestResult:
        """Run a specific test sequence.
        
        Args:
            prompt_id: ID of the prompt sequence to run
            
        Returns:
            TestResult object
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
            # Return a TestResult object even for failure
            return TestResult(
                model=self.model,
                provider=self.provider or "unknown",
                prompt_id=prompt_id,
                success=False,
                perfect_success=False,
                timestamp=datetime.now().isoformat(),
                metrics={
                    "total_tool_calls": 0,
                    "total_send_count": 0,
                    "latency_ms": 0,
                    "total_steps": 0,
                    "successful_steps": 0,
                }
            )
        
        logger.info(f"Running test {prompt_id} with {len(prompt_sequence['sequence'])} steps")
        
        # Run each step in the sequence through the agent
        successful_steps = 0
        previous_result = None
        all_validation_errors = []
        
        # Start the MCP servers
        async with self.agent.run_mcp_servers():
            for i, step in enumerate(prompt_sequence["sequence"]):
                logger.info(f"Running step {i+1}/{len(prompt_sequence['sequence'])}")
                try:
                    response = await self.send_message(step, previous_result)
                    if "error" in response:
                        logger.error(f"Error in step {i+1}: {response['error']}")
                        break
                    
                    # Accumulate validation errors
                    if "validation_errors" in response and response["validation_errors"]:
                        all_validation_errors.extend(response["validation_errors"])
                        logger.warning(f"Step {i+1} had {len(response['validation_errors'])} validation errors")
                    
                    previous_result = response.get("result")
                    successful_steps += 1
                except Exception as e:
                    logger.error(f"Error in step {i+1}: {str(e)}")
                    break
        
        # Calculate metrics
        avg_latency = self.total_latency / self.send_count if self.send_count > 0 else 0
        
        # Extract messages in serializable format using our improved serialization
        serialized_messages = []
        
        for msg in self.messages:
            try:
                # Create a serializable message object with properly handled parts
                serialized_msg = {}
                
                if hasattr(msg, 'parts') and isinstance(msg.parts, list):
                    # Handle message parts separately using the helper
                    parts = []
                    for part in msg.parts:
                        # Convert each part to a JSON-serializable dict
                        part_dict = {"content": json_serializable(part)}
                        parts.append(part_dict)
                    serialized_msg['parts'] = parts
                    serialized_messages.append(serialized_msg)
                else:
                    # Fallback for messages without parts
                    serialized_messages.append({"content": str(msg)})
            except Exception as e:
                # If serialization fails, include a placeholder with error info
                logger.warning(f"Error serializing message: {e}")
                trace_data = log_exception_state(e, logger)
                serialized_messages.append({"error": f"Failed to serialize message: {str(e)}", "debug": trace_data})
        
        # Create metrics dictionary including extra fields that don't fit in TestResult
        metrics = {
            "total_tool_calls": self.tool_calls_count,
            "total_send_count": self.send_count,
            "latency_ms": avg_latency,
            "total_steps": len(prompt_sequence["sequence"]),
            "successful_steps": successful_steps,
            "messages": serialized_messages
        }
        
        # Create TestResult object
        result = TestResult(
            model=self.model,
            provider=self.provider or "unknown",
            prompt_id=prompt_id,
            success=successful_steps == len(prompt_sequence["sequence"]),
            perfect_success=successful_steps == len(prompt_sequence["sequence"]) and len(all_validation_errors) == 0,
            validation_errors=all_validation_errors,
            validation_error_count=len(all_validation_errors),
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
        
        # Save result to file with proper directory structure
        # Format: results/model/prompt_id/provider_variant/timestamp.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean up model, provider, and prompt_id for use in paths
        model_safe = self.model.replace('/', '_')
        prompt_id_safe = prompt_id  # Usually already safe but included for clarity
        provider_variant = "default" if not self.provider else self.provider.replace('/', '_')
        
        # Extract provider name and variant (if any)
        if "_" in provider_variant and not provider_variant.startswith("_"):
            # If provider has a variant like "fireworks_fp8"
            provider_parts = provider_variant.split('_', 1)
            if len(provider_parts) == 2:
                provider_variant = f"{provider_parts[0]}_{provider_parts[1]}"
        
        # Set up directory structure
        result_dir = Path("results")
        model_dir = result_dir / model_safe
        prompt_dir = model_dir / prompt_id_safe
        provider_dir = prompt_dir / provider_variant
        
        # Ensure all directories exist
        provider_dir.mkdir(exist_ok=True, parents=True)
        
        # Create the result file
        result_file = provider_dir / f"{timestamp}.json"
        
        try:
            with open(result_file, "w") as f:
                # Convert result to dict for saving with proper serialization
                result_dict = json.loads(result.json())
                json.dump(result_dict, f, indent=2, default=str)
            logger.info(f"Test results saved to {result_file}")
            
            # Also save a copy to the client's format for reporting
            client = FileSystemClient()
            client.save_test_result(result)
            
            # Log validation error info if present
            if all_validation_errors:
                logger.warning(f"Test completed with {len(all_validation_errors)} validation errors")
                for i, err in enumerate(all_validation_errors[:5]):  # Log first 5 errors
                    if isinstance(err, dict) and "message" in err:
                        logger.warning(f"Validation error {i+1}: {str(err['message'])[:100]}...")
            
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
    parser.add_argument("--test-dir", help="Custom directory for test files")
    args = parser.parse_args()
    
    # Special case: list providers
    if args.list_providers:
        await list_providers_for_model(args.model)
        return
    
    # Create a test directory if specified
    test_files_dir = None
    if args.test_dir:
        test_files_dir = Path(args.test_dir)
        test_files_dir.mkdir(exist_ok=True, parents=True)
    
    tester = ProviderTester(model=args.model, provider=args.provider, test_files_dir=test_files_dir)
    
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
        test_helper = FileSystemTestHelper(test_files_dir=test_files_dir)
        prompts = test_helper.load_prompts()
        for prompt in prompts:
            print(f"\nRunning test: {prompt['id']}")
            await tester.run_test(prompt["id"])
    else:
        # Run specified prompt sequence
        await tester.run_test(args.prompt)

if __name__ == "__main__":
    asyncio.run(main())