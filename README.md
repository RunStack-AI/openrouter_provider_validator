# OpenRouter Provider Validator

A tool for systematically testing and evaluating various OpenRouter.ai providers using predefined prompts with a focus on tool use capabilities.

## Overview

This project helps you assess the reliability and performance of different OpenRouter.ai providers by testing their ability to interact with a toy filesystem through tools. The tests are automated and results are stored for analysis and reporting.

## Features

- Configure and test multiple OpenRouter providers
- Create and manage test prompts focused on tool use capabilities
- Execute tests against a toy filesystem to evaluate performance
- Analyze response data and errors
- Generate detailed performance reports
- Track token usage and provider-specific metrics

## Architecture

The system consists of these core components:

1. **Filesystem Client** (`client.py`) - Manages data storage and retrieval
2. **MCP Server** (`mcp_server.py`) - Exposes filesystem operations as tools
3. **Test Agent** (`agent.py`) - Executes tests and interacts with OpenRouter
4. **Test Runner** (`test_runner.py`) - Orchestrates parallel test execution
5. **Provider Configuration** (`provider_config.py`) - Manages provider settings
6. **Metrics Extraction** (`metrics_extractor.py`) - Extracts metrics from responses and logs
7. **Error Classification** (`error_classifier.py`) - Categorizes errors by pattern
8. **Report Generation** (`report_generator.py`) - Creates human-readable reports

## Test Methodology

The validator tests providers using a sequence of steps:

1. A toy filesystem is created with sample files and directories
2. Test prompts instruct the model to perform various file operations
3. The system evaluates whether the model correctly uses tools to complete tasks
4. Results are stored, analyzed, and summarized in reports

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

### Running Individual Tests

Test a single prompt with a specific model and provider:

```bash
python agent.py --model anthropic/claude-3.7-sonnet --prompt read_simple_file
```

Or use the simplified test runner:

```bash
python run_test.py read_simple_file
```

### Running All Tests

Run all prompts against a specific model/provider:

```bash
python agent.py --model anthropic/claude-3.7-sonnet --all
```

### Automated Testing Across Providers

Run tests on all enabled providers with multiple models:

```bash
python test_runner.py
```

## Toy Filesystem Structure

The toy filesystem used for testing includes:

- `data/test_files/` - Base directory for test files
  - `sample1.txt` - Simple text file for reading
  - `sample2.txt` - Another sample file with different content
  - `nested/` - Subdirectory for testing nested path operations
    - `sample3.txt` - File in nested directory

Tests will create, modify, and manipulate these files to evaluate tool use capabilities.

## Test Prompts

The system includes various test prompts focused on different file operations:

- Reading files
- Writing and appending to files
- Creating directories
- Searching for content
- Copying and moving files
- Handling errors
- Executing sequences of operations

These prompts are stored in `data/prompts.json` and can be customized.

## Report Generation

After running tests, you can generate reports with:

```bash
python -c "from report_generator import generate_and_save_all_reports; generate_and_save_all_reports()"
```

Reports are saved to the `reports/` directory with timestamps.

## Extending the System

### Adding New Providers

Edit `data/providers.json` to add new provider configurations.

### Creating New Test Prompts

Add new test cases to `data/prompts.json` following the existing format.

### Adding New Tools

Extend the MCP Server in `mcp_server.py` to add new filesystem operations.

## License

MIT
