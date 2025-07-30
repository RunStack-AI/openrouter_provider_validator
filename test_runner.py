"""OpenRouter Provider Validator - Test Runner

Orchestrates test execution across multiple providers with parallel processing.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from client import FileSystemClient, ProviderConfig, TestPrompt, TestResult
from error_classifier import classify_error

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/test_runner.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger("test_runner")

# Initialize the filesystem client
client = FileSystemClient()

# Load API key from environment variable
API_KEY = os.getenv('OPENROUTER_API_KEY')
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

async def test_provider_with_prompt(provider_name: str, model_name: str, prompt: TestPrompt) -> TestResult:
    """Test a single provider with a specific prompt.
    
    Args:
        provider_name: Name of the provider to test
        model_name: Model identifier
        prompt: Test prompt to use
        
    Returns:
        Test result object
    """
    logger.info(f"Testing provider {provider_name} with model {model_name} using prompt {prompt.id}")
    
    # Set up the model with provider routing
    model_settings = {
        "extra_body": {
            "provider": provider_name
        }
    }
    
    model = OpenAIModel(
        model_name,
        provider=OpenAIProvider(
            base_url='https://openrouter.ai/api/v1',
            api_key=API_KEY
        ),
        settings=model_settings
    )
    
    # Set up MCP Server
    mcp_servers = [MCPServerStdio('python', ['./mcp_server.py'])]
    
    # Load agent prompt
    with open("agents/openrouter_validator.md", "r") as f:
        agent_prompt = f.read()
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    agent_prompt = agent_prompt.replace('{time_now}', time_now)
    
    # Create agent
    agent = Agent(model, mcp_servers=mcp_servers, system_prompt=agent_prompt)
    
    # Prepare result object with defaults
    result = TestResult(
        provider=provider_name,
        model=model_name,
        prompt_id=prompt.id,
        success=False,  # Default to False, will be updated if test succeeds
        timestamp=datetime.now()
    )
    
    try:
        async with agent.run_mcp_servers():
            # Execute test
            agent_result = await agent.run(prompt.prompt)
            
            # Extract response data
            response_id = agent_result.id if hasattr(agent_result, 'id') else None
            token_usage = agent_result.usage if hasattr(agent_result, 'usage') else None
            
            # Extract full response data
            response_data = {}
            if hasattr(agent_result, 'response'):
                response_data = agent_result.response
            
            # Update result
            result.success = True
            result.response_id = response_id
            result.token_usage = token_usage
            result.response_data = response_data
            
            logger.info(f"Test succeeded for {provider_name}/{model_name}/{prompt.id}")
            
    except Exception as e:
        error_message = str(e)
        error_category = classify_error(error_message)
        
        # Update result with error details
        result.error_message = error_message
        result.error_category = error_category
        
        logger.error(f"Test failed for {provider_name}/{model_name}/{prompt.id}: {error_message} (Category: {error_category})")
    
    # Save the result
    client.save_test_result(result)
    
    return result

async def test_provider(provider_name: str, model_name: str, prompts: List[TestPrompt], attempts: int = 3) -> Dict[str, Any]:
    """Test a provider with all prompts, with multiple attempts.
    
    Args:
        provider_name: Name of the provider to test
        model_name: Model identifier
        prompts: List of prompts to test
        attempts: Number of attempts to make for each prompt
        
    Returns:
        Dictionary with test results summary
    """
    logger.info(f"Starting tests for provider {provider_name} with model {model_name}")
    
    results = []
    failed_request_ids = []
    error_categories = {}
    
    for prompt in prompts:
        for attempt in range(attempts):
            logger.info(f"Attempt {attempt+1}/{attempts} for {provider_name}/{model_name}/{prompt.id}")
            
            result = await test_provider_with_prompt(provider_name, model_name, prompt)
            results.append(result)
            
            # Track failures
            if not result.success:
                if result.response_id:
                    failed_request_ids.append(result.response_id)
                
                # Track error categories
                category = result.error_category or 'unknown'
                error_categories[category] = error_categories.get(category, 0) + 1
    
    # Calculate summary statistics
    total_attempts = len(results)
    failures = sum(1 for r in results if not r.success)
    
    return {
        "provider": provider_name,
        "model": model_name,
        "total_attempts": total_attempts,
        "failures": failures,
        "failed_request_ids": failed_request_ids,
        "error_categories": error_categories
    }

async def run_tests(max_workers: int = 5):
    """Run tests for all enabled providers in parallel.
    
    Args:
        max_workers: Maximum number of concurrent tests
    """
    # Load providers and prompts
    providers = client.load_providers()
    enabled_providers = [p for p in providers if p.enabled]
    prompts = client.load_prompts()
    
    if not enabled_providers:
        logger.warning("No enabled providers found. Please add providers to data/providers.json")
        return
    
    if not prompts:
        logger.warning("No test prompts found. Please add prompts to data/prompts.json")
        return
    
    logger.info(f"Starting tests with {len(enabled_providers)} providers and {len(prompts)} prompts")
    
    # Group providers by model for result organization
    model_results = {}
    
    # Create tasks for each provider
    tasks = []
    for provider in enabled_providers:
        task = test_provider(provider.name, provider.model, prompts)
        tasks.append(task)
    
    # Run tests with concurrency limits
    results = await asyncio.gather(*tasks)
    
    # Organize results by model
    for result in results:
        model_name = result["model"]
        if model_name not in model_results:
            model_results[model_name] = {
                "providers": {},
                "total_attempts": 0,
                "total_failures": 0
            }
        
        model_results[model_name]["providers"][result["provider"]] = {
            "attempts": result["total_attempts"],
            "failures": result["failures"],
            "failed_request_ids": result["failed_request_ids"],
            "error_categories": result["error_categories"]
        }
        
        model_results[model_name]["total_attempts"] += result["total_attempts"]
        model_results[model_name]["total_failures"] += result["failures"]
    
    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path("data") / "aggregated_results.json"
    
    with open(results_path, 'w') as f:
        json.dump(model_results, f, indent=2)
    
    logger.info(f"Tests completed. Results saved to {results_path}")
    
    return model_results

async def main():
    """Main entry point for the test runner."""
    try:
        # Ensure directories exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Run the tests
        results = await run_tests()
        
        # Generate summary report
        if results:
            report = ["# OpenRouter Provider Test Results\n"]
            report.append(f"## Test Summary\n")
            report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            for model_name, model_data in results.items():
                report.append(f"## Model: {model_name}\n")
                report.append(f"Total Attempts: {model_data['total_attempts']}\n")
                report.append(f"Total Failures: {model_data['total_failures']}\n")
                report.append(f"Overall Success Rate: {((model_data['total_attempts'] - model_data['total_failures']) / model_data['total_attempts'] * 100):.2f}%\n")
                
                report.append("### Provider Results\n")
                for provider_name, provider_data in model_data["providers"].items():
                    report.append(f"#### Provider: {provider_name}\n")
                    report.append(f"Attempts: {provider_data['attempts']}\n")
                    report.append(f"Failures: {provider_data['failures']}\n")
                    
                    success_rate = ((provider_data['attempts'] - provider_data['failures']) / provider_data['attempts'] * 100)
                    report.append(f"Success Rate: {success_rate:.2f}%\n")
                    
                    if provider_data['error_categories']:
                        report.append("\nError Categories:\n")
                        for category, count in provider_data['error_categories'].items():
                            report.append(f"- {category}: {count} occurrences\n")
                            
                    report.append("\n")
            
            # Save the report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"test_report_{timestamp}"
            report_content = "".join(report)
            
            client.save_report(report_name, report_content)
            logger.info(f"Test report generated: {report_name}.md")
            
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
