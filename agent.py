# PydanticAI Agent with MCP for OpenRouter Provider Validator

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessage, SystemPromptPart, UserPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import AgentRunResult

from dotenv import load_dotenv
import os
import argparse
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import asyncio
import traceback

load_dotenv()

# Configure logging if logfire is available
try:
    import logfire
    logfire_token = os.getenv("LOGFIRE_API_KEY")
    if logfire_token:
        logfire.configure(token=logfire_token)
        logfire.instrument_openai()
        print("Logfire configured for logging")
    else:
        print("Logfire API key not found, skipping configuration")
except ImportError:
    print("Logfire not installed, skipping logging configuration")

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="OpenRouter Provider Validator Agent")
    parser.add_argument(
        "--model", 
        type=str, 
        default="anthropic/claude-3.7-sonnet",
        help="Model identifier to use with OpenRouter (default: anthropic/claude-3.7-sonnet)"
    )
    parser.add_argument(
        "--provider", 
        type=str, 
        default=None,
        help="Provider name to test (default: None, will not route to specific provider)"
    )
    return parser.parse_args()

# Get command line arguments
args = parse_args()

# Set up OpenRouter based model with optional provider routing
API_KEY = os.getenv('OPENROUTER_API_KEY')
if API_KEY is None:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

# Configure model settings with provider routing if specified
model_settings = {}
if args.provider:
    model_settings["extra_body"] = {
        "provider": args.provider
    }
    print(f"Routing to provider: {args.provider}")

model = OpenAIModel(
    args.model,  # Use the model from command line arguments
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1', 
        api_key=API_KEY
    ),
    settings=model_settings
)

# Set up MCP Server for the Agent
mcp_servers = [
    MCPServerStdio('python', ['./mcp_server.py']),
]

# Function to filter message history
def filtered_message_history(
    result: Optional[AgentRunResult], 
    limit: Optional[int] = None, 
    include_tool_messages: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    Filter and limit the message history from an AgentRunResult.
    
    Args:
        result: The AgentRunResult object with message history
        limit: Optional int, if provided returns only system message + last N messages
        include_tool_messages: Whether to include tool messages in the history
        
    Returns:
        Filtered list of messages in the format expected by the agent
    """
    if result is None:
        return None
        
    # Get all messages
    messages: list[ModelMessage] = result.all_messages()
    
    # Extract system message (always the first one with role="system")
    system_message = next((msg for msg in messages if type(msg.parts[0]) == SystemPromptPart), None)
    
    # Filter non-system messages
    non_system_messages = [msg for msg in messages if type(msg.parts[0]) != SystemPromptPart]
    
    # Apply tool message filtering if requested
    if not include_tool_messages:
        non_system_messages = [msg for msg in non_system_messages if not any(isinstance(part, ToolCallPart) or isinstance(part, ToolReturnPart) for part in msg.parts)]
    
    # Find the most recent UserPromptPart before applying limit
    latest_user_prompt_part = None
    latest_user_prompt_index = -1
    for i, msg in enumerate(non_system_messages):
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                latest_user_prompt_part = part
                latest_user_prompt_index = i
    
    # Apply limit if specified, but ensure paired tool calls and returns stay together
    if limit is not None and limit > 0:
        # Identify tool call IDs and their corresponding return parts
        tool_call_ids = {}
        tool_return_ids = set()
        
        for i, msg in enumerate(non_system_messages):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_call_ids[part.tool_call_id] = i
                elif isinstance(part, ToolReturnPart):
                    tool_return_ids.add(part.tool_call_id)
        
        # Take the last 'limit' messages but ensure we include paired messages
        if len(non_system_messages) > limit:
            included_indices = set(range(len(non_system_messages) - limit, len(non_system_messages)))
            
            # Include any missing tool call messages for tool returns that are included
            for i, msg in enumerate(non_system_messages):
                if i in included_indices:
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart) and part.tool_call_id in tool_call_ids:
                            included_indices.add(tool_call_ids[part.tool_call_id])
            
            # Check if the latest UserPromptPart would be excluded by the limit
            if (latest_user_prompt_index >= 0 and 
                latest_user_prompt_index not in included_indices and 
                latest_user_prompt_part is not None and 
                system_message is not None):
                # Find if system_message already has a UserPromptPart
                user_prompt_index = next((i for i, part in enumerate(system_message.parts) 
                                       if isinstance(part, UserPromptPart)), None)
                
                if user_prompt_index is not None:
                    # Replace existing UserPromptPart
                    system_message.parts[user_prompt_index] = latest_user_prompt_part
                else:
                    # Add new UserPromptPart to system message
                    system_message.parts.append(latest_user_prompt_part)
            
            # Create a new list with only the included messages
            non_system_messages = [msg for i, msg in enumerate(non_system_messages) if i in included_indices]
    
    # Combine system message with other messages
    result_messages = []
    if system_message:
        result_messages.append(system_message)
    result_messages.extend(non_system_messages)
    
    return result_messages

# Set up Agent with Server
agent_name = "openrouter_validator"
def load_agent_prompt(agent: str):
    """Loads given agent replacing `time_now` var with current time"""
    print(f"Loading {agent} agent prompt")
    time_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if the agents folder exists
    agents_dir = os.path.join(os.getcwd(), "agents")
    if not os.path.exists(agents_dir):
        os.makedirs(agents_dir)
    
    # Check if the agent prompt file exists
    agent_prompt_path = os.path.join(agents_dir, f"{agent}.md")
    if not os.path.exists(agent_prompt_path):
        # Create a basic prompt if it doesn't exist
        basic_prompt = f"""# OpenRouter Provider Validator Agent

## Identity
You are the OpenRouter Provider Validator Agent, designed to test and evaluate various OpenRouter.ai providers using predefined prompts with a focus on tool use capabilities.

## Capabilities
- Configure and manage providers for testing
- Manage test prompts focused on tool use
- Execute tests and collect results
- Generate reports and statistics
- Analyze provider performance

## Current Time
{{time_now}}
"""
        with open(agent_prompt_path, "w") as f:
            f.write(basic_prompt)
        print(f"Created basic agent prompt at {agent_prompt_path}")
    
    with open(agent_prompt_path, "r") as f:
        agent_prompt = f.read()
    
    agent_prompt = agent_prompt.replace('{time_now}', time_now)
    return agent_prompt

# Load up the agent system prompt
agent_prompt = load_agent_prompt(agent_name)

# Display the selected model
print(f"Using model: {args.model}")

# Initialize the agent
agent = Agent(model, mcp_servers=mcp_servers, system_prompt=agent_prompt)

async def main():
    """CLI testing in a conversation with the agent"""
    async with agent.run_mcp_servers(): 
        result: AgentRunResult = None

        # Chat Loop
        while True:
            if result:
                print(f"\n{result.output}")
            user_input = input("\n> ")
            err = None
            for i in range(0, 2):
                try:
                    # Use the filtered message history
                    result = await agent.run(
                        user_input, 
                        message_history=filtered_message_history(
                            result,
                            limit=24,                  # Last 24 non-system messages
                            include_tool_messages=True # Include tool messages
                        )
                    )
                    break
                except Exception as e:
                    err = e
                    traceback.print_exc()
                    await asyncio.sleep(2)
            if result is None:
                print(f"\nError {err}. Try again...\n")
                continue
            elif len(result.output) == 0:
                continue
                
if __name__ == "__main__":
    asyncio.run(main())
