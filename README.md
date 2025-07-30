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
- Auto-detect and use appropriate providers for specific models
- Test the same model across multiple providers automatically
- Save detailed test results for analysis

## Architecture

The system consists of these core components:

1. **Filesystem Client** (`client.py`) - Manages data storage and retrieval
2. **MCP Server** (`mcp_server.py`) - Exposes filesystem operations as tools
3. **Provider Config** (`provider_config.py`) - Manages provider configurations and model routing
4. **Test Agent** (`agent.py`) - Executes prompt sequences and interacts with OpenRouter
5. **Test Runner** (`test_runner.py`) - Orchestrates automated test execution
6. **Prompt Definitions** (`data/prompts.json`) - Defines test scenarios with prompt sequences
7. **Provider Definitions** (`data/providers.json`) - Configures available providers and their supported models

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
- Required packages: `pydantic`, `httpx`, `python-dotenv`

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
python agent.py --model moonshot/kimi-k2 --provider deepinfra --prompt file_operations_sequence
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

This will automatically run all tests for each enabled provider configured for the moonshot/kimi-k2 model, generating a comprehensive comparison report.

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

Providers are configured in `data/providers.json` with the following structure:

```json
{
  "id": "provider_id",
  "name": "Provider Display Name",
  "enabled": true,
  "supported_models": ["model/id1", "model/id2"],
  "description": "Description of the provider"
}
```

The system automatically matches models to appropriate providers based on the `supported_models` field. This allows testing different provider implementations of the same model (e.g., multiple providers for Moonshot/Kimi-K2).

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

### Adding New Models and Providers

Add new provider configurations to `data/providers.json` to test additional models:

```json
{
  "id": "new_provider_id",
  "name": "New Provider Name (via OpenRouter)",
  "enabled": true,
  "supported_models": [
    "vendorid/modelname"
  ],
  "description": "Description of the provider and model"
}
```

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
