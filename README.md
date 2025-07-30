# OpenRouter Provider Validator

A tool for systematically testing and evaluating various OpenRouter.ai providers using predefined prompt sequences with a focus on tool use capabilities.

## Overview

This project helps you assess the reliability and performance of different OpenRouter.ai providers by testing their ability to interact with a toy filesystem through tools. The tests use sequences of related prompts to evaluate the model's ability to maintain context and perform multi-step operations.

## Features

- Test models with sequences of related prompts
- Evaluate multi-step task completion capability
- Automatically set up toy filesystem for testing
- Track success rates and tool usage metrics
- Generate comparative reports across models
- **Auto-detect available providers for specific models via API**
- Test the same model across multiple providers automatically
- Save detailed test results for analysis

## Architecture

The system consists of these core components:

1. **Filesystem Client** (`client.py`) - Manages data storage and retrieval
2. **Filesystem Test Helper** (`filesystem_test_helper.py`) - Initializes test environments
3. **MCP Server** (`mcp_server.py`) - Exposes filesystem operations as tools through FastMCP
4. **Provider Config** (`provider_config.py`) - Manages provider configurations and model routing
5. **Test Agent** (`agent.py`) - Executes prompt sequences and interacts with OpenRouter
6. **Test Runner** (`test_runner.py`) - Orchestrates automated test execution
7. **Prompt Definitions** (`data/prompts.json`) - Defines test scenarios with prompt sequences

## Technical Implementation

The validator uses the PydanticAI framework to create a robust testing system:

- **Agent Framework**: Uses the `pydantic_ai.Agent` class to manage interactions and tool calling
- **MCP Server**: Implements a FastMCP server that exposes filesystem operations as tools
- **Model Interface**: Connects to OpenRouter through the `OpenAIModel` and `OpenAIProvider` classes
- **Test Orchestration**: Manages testing across providers and models, collecting metrics and results

The test agent creates instances of the Agent class to run tests while tracking performance metrics.

## Test Methodology

The validator tests providers using a sequence of steps:

1. A toy filesystem is initialized with sample files
2. The agent sends a sequence of prompts for each test
3. Each prompt builds on previous steps in a coherent workflow
4. The system evaluates tool use and success rate for each step
5. Results are stored and analyzed across models

## Requirements

- Python 3.9 or higher
- An OpenRouter API key
- Required packages: `pydantic`, `httpx`, `python-dotenv`, `pydantic-ai`

## Setup

1. Clone this repository
2. Create a `.env` file with your API key:
   ```
   OPENROUTER_API_KEY=your-api-key-here
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Listing Available Providers

List all available providers for a specific model:

```bash
python agent.py --model moonshot/kimi-k2 --list-providers
```

Or list providers for multiple models:

```bash
python test_runner.py --list-providers --models anthropic/claude-3.7-sonnet moonshot/kimi-k2
```

### Running Individual Tests

Test a single prompt sequence with a specific model:

```bash
python agent.py --model anthropic/claude-3.7-sonnet --prompt file_operations_sequence
```

Test with a specific provider for a model (overriding auto-detection):

```bash
python agent.py --model moonshot/kimi-k2 --provider fireworks --prompt file_operations_sequence
```

### Running All Tests

Run all prompt sequences against a specific model (auto-detects provider):

```bash
python agent.py --model moonshot/kimi-k2 --all
```

### Testing With All Providers

Test a model with all its enabled providers automatically:

```bash
python test_runner.py --models moonshot/kimi-k2 --all-providers
```

This will automatically run all tests for each provider configured for the moonshot/kimi-k2 model, generating a comprehensive comparison report.

### Automated Testing Across Models

Run same tests on multiple models for comparison:

```bash
python test_runner.py --models anthropic/claude-3.7-sonnet moonshot/kimi-k2
```

With specific provider mappings:

```bash
python test_runner.py --models moonshot/kimi-k2 anthropic/claude-3.7-sonnet --providers "moonshot/kimi-k2:fireworks" "anthropic/claude-3.7-sonnet:anthropic"
```

## Provider Configuration

The system automatically discovers providers for models directly from the OpenRouter API using the `/model/{model_id}/endpoints` endpoint. This ensures that:

1. You always have the most up-to-date provider information
2. You can see accurate pricing and latency metrics 
3. You only test with providers that actually support the tools feature

The API-based approach means you don't need to maintain manual provider configurations in most cases. However, for backward compatibility and fallback purposes, the system also supports loading provider configurations from `data/providers.json`.

## Prompt Sequences

Tests are organized as sequences of related prompts that build on each other. Examples include:

### File Operations Sequence
1. Read a file and describe contents
2. Create a summary in a new file
3. Read another file
4. Append content to that file
5. Create a combined file in a new directory

### Search and Report
1. Search files for specific content
2. Create a report of search results
3. Move the report to a different location

### Error Handling
1. Attempt to access non-existent files
2. Document error handling approach
3. Test error recovery capabilities

The full set of test sequences is defined in `data/prompts.json` and can be customized.

## Test Results

Results include detailed metrics:

- Overall success (pass/fail)
- Success rate for individual steps
- Number of tool calls per step
- Latency measurements
- Token usage statistics

A summary report is generated with comparative statistics across models and providers. When testing with multiple providers, the system generates provider comparison tables showing which provider performs best for each model.

## Extending the System

### Adding Custom Provider Configurations

While the system can automatically detect providers from the OpenRouter API, you can add custom provider configurations to `data/providers.json` to override or supplement the API data:

```json
{
  "id": "custom_provider_id",
  "name": "Custom Provider Name (via OpenRouter)",
  "enabled": true,
  "supported_models": [
    "vendorid/modelname"
  ],
  "description": "Description of the provider and model"
}
```

You can also disable specific providers by setting `"enabled": false` in their configuration.

### Adding New Prompt Sequences

Add new test scenarios to `data/prompts.json` following this format:

```json
{
  "id": "new_test_scenario",
  "name": "Description of Test",
  "description": "Detailed explanation of what this tests",
  "sequence": [
    "First prompt in sequence",
    "Second prompt building on first",
    "Third prompt continuing the task"  
  ]
}
```

### Customizing the Agent Behavior

Edit `agents/openrouter_validator.md` to modify the system prompt and agent behavior.

## License

MIT
