# OpenRouter Provider Validator

A tool for systematically testing and evaluating various OpenRouter.ai providers using predefined prompts with a focus on tool use capabilities.

## Overview

This project helps you assess the reliability and performance of different OpenRouter.ai providers across multiple test runs and generates comprehensive reports for analysis.

## Features

- Configure and test multiple OpenRouter providers
- Create and manage test prompts focused on tool use capabilities
- Execute parallel tests with comprehensive logging
- Analyze response data and errors
- Generate detailed performance reports
- Track token usage and provider-specific metrics

## Architecture

The system consists of these core components:

1. **Filesystem Client** (`client.py`) - Manages data storage and retrieval
2. **MCP Server** (`mcp_server.py`) - Exposes client operations as tools
3. **Test Agent** (`agent.py`) - Provides interactive CLI testing
4. **Test Runner** (`test_runner.py`) - Orchestrates parallel test execution
5. **Provider Configuration** (`provider_config.py`) - Manages provider settings
6. **Metrics Extraction** (`metrics_extractor.py`) - Extracts data from responses and logs
7. **Error Classification** (`error_classifier.py`) - Categorizes errors by pattern
8. **Report Generation** (`report_generator.py`) - Creates human-readable reports

## Requirements

- Python 3.9 or higher
- An OpenRouter API key
- Required packages: `pydantic-ai`, `httpx`, `python-dotenv`

## Setup

1. Clone this repository
2. Create a `.env` file with your API key:
   ```
   OPENROUTER_API_KEY=your-api-key-here
   ```
3. Install dependencies:
   ```
   pip install pydantic-ai httpx python-dotenv
   ```

## Usage

### Interactive Mode

Use the agent to interactively test providers:

```bash
python agent.py --model anthropic/claude-3.7-sonnet
```

Specify a provider to test:

```bash
python agent.py --model anthropic/claude-3.7-sonnet --provider anthropic
```

### Automated Testing

Run automated tests across all enabled providers:

```bash
python test_runner.py
```

## Configuration

Edit these JSON files to configure the validator:

- `data/providers.json` - Provider configurations
- `data/prompts.json` - Test prompts

## Output

The system generates:

- Test results in the `results/` directory (organized by model)
- Analysis reports in the `reports/` directory
- Logs in the `logs/` directory

## License

MIT
