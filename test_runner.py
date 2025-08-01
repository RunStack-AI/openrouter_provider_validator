#!/usr/bin/env python
"""Test runner for automating OpenRouter Provider Validator tests."""

import argparse
import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ProviderTester
from filesystem_test_helper import FileSystemTestHelper
from provider_config import ProviderConfig
from client import TestResult

# Configure logging
logging_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"test_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("test_runner")

async def load_prompts(prompts_dir="prompts", prompt_ids=None):
    """
    Load all prompts from the prompts directory.
    
    Args:
        prompts_dir: Directory containing prompt files
        prompt_ids: Optional list of specific prompt IDs to load
    
    Returns:
        List of prompt configurations
    """
    prompts_path = Path(prompts_dir)
    prompts = []
    
    if not prompts_path.exists() or not prompts_path.is_dir():
        logger.error(f"Prompts directory not found: {prompts_path}")
        return []
    
    # Load all JSON files in the prompts directory
    for prompt_file in prompts_path.glob("*.json"):
        try:
            with open(prompt_file, "r") as f:
                prompt = json.load(f)
                prompt["id"] = prompt_file.stem
                
                if prompt_ids is None or prompt["id"] in prompt_ids:
                    prompts.append(prompt)
        except Exception as e:
            logger.error(f"Error loading prompt file {prompt_file}: {e}")
    
    return prompts


def get_provider_id(provider_config):
    return provider_config.get("id") if provider_config else None


async def get_provider_display_name(provider_id, results=None):
    """
    Get a user-friendly display name for a provider.
    
    Args:
        provider_id: Provider ID
        results: Optional list of test results
        
    Returns:
        User-friendly provider name
    """
    if results and len(results) > 0:
        # Check if we have a TestResult object (has model attribute) or dict
        if hasattr(results[0], 'provider'):
            return results[0].provider
        elif isinstance(results[0], dict) and "provider" in results[0]:
            return results[0]["provider"]
        
    try:
        provider_config = await ProviderConfig.get_provider(provider_id) if provider_id else None
        if provider_config and "name" in provider_config:
            return provider_config["name"]
    except Exception:
        pass
        
    return provider_id or "Default Routing"


async def run_provider_tests(model: str, provider_config: Dict, all_prompts: List[Dict], output_dir: str, batch_timestamp: str):
    """
    Run tests for a specific provider and model.
    
    Args:
        model: Model identifier
        provider_config: Provider configuration
        all_prompts: List of prompts to test
        output_dir: Directory to save results
        batch_timestamp: Timestamp for this batch of tests
        
    Returns:
        Test results for this provider
    """
    provider_id = get_provider_id(provider_config)
    provider_name = provider_config.get("name", await get_provider_display_name(provider_id))
    
    logger.info(f"Running tests with provider: {provider_name}")
    
    # Create provider-specific test directory
    model_safe = model.replace("/", "_")
    provider_safe = "default" if provider_id is None else provider_id.replace("/", "_")
    provider_test_dir = Path("data/test_files") / f"{model_safe}_{provider_safe}"
    
    # Create directory and initialize with template files
    provider_test_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy template files if a templates directory exists
    templates_dir = Path("data/test_files/templates")
    if templates_dir.exists() and templates_dir.is_dir():
        # Clear any previous test files
        for item in provider_test_dir.iterdir():
            if item.is_dir() and item.name != "templates":
                shutil.rmtree(item)
            elif item.is_file():
                item.unlink()
                
        # Copy template files to provider test directory
        for template_file in templates_dir.glob("**/*"):
            if template_file.is_file():
                # Construct the relative path from templates_dir
                rel_path = template_file.relative_to(templates_dir)
                # Construct destination path in provider_test_dir
                dest_file = provider_test_dir / rel_path
                # Create parent directories if needed
                dest_file.parent.mkdir(exist_ok=True, parents=True)
                # Copy the template file
                shutil.copy2(template_file, dest_file)
    
    # Initialize tester with the provider ID and test directory
    provider_results = []
    tester = ProviderTester(
        model=model, 
        provider=provider_id,
        test_files_dir=provider_test_dir
    )
    
    for prompt in all_prompts:
        logger.info(f"Running test: {prompt['id']} with provider {provider_name}")
        try:
            result = await tester.run_test(prompt["id"])
            provider_results.append(result)
        except Exception as e:
            logger.error(f"Error running test {prompt['id']} with provider {provider_name}: {e}")
            # Create a failed result record as a TestResult object
            failed_result = TestResult(
                model=model,
                provider=provider_id or "unknown",
                prompt_id=prompt["id"],
                success=False,
                error_message=str(e),
                error_category="execution_error",
                timestamp=datetime.now().isoformat(),
                metrics={
                    "total_steps": 0,
                    "successful_steps": 0,
                    "total_tool_calls": 0,
                    "total_send_count": 0,
                    "latency_ms": 0
                }
            )
            provider_results.append(failed_result)
    
    result_key = "default"
    if provider_id is not None and provider_id != "":
        result_key = provider_id
    
    return result_key, provider_results


async def run_tests(
    models: Optional[List[str]] = None,
    prompts: Optional[List[str]] = None,
    providers: Optional[Dict[str, str]] = None, 
    output_dir: Optional[str] = None,
    test_all_providers: bool = False,
    parallel: bool = True
):
    """Run tests for specified models and prompts.
    
    Args:
        models: List of models to test (defaults to claude models)
        prompts: List of prompt IDs to test (defaults to all)
        providers: Dictionary mapping models to specific providers (optional)
        output_dir: Directory to save results (defaults to "results")
        test_all_providers: Whether to test all enabled providers for each model
        parallel: Whether to run provider tests in parallel
    """
    output_dir = output_dir or "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Default models
    models = models or [
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.7-haiku"
    ]
    
    # Load all prompts
    all_prompts = await load_prompts(prompt_ids=prompts)
    if not all_prompts:
        logger.error("No prompts found")
        return
    
    if prompts:
        logger.info(f"Loaded {len(all_prompts)} prompts: {', '.join(p['id'] for p in all_prompts)}")
    else:
        logger.info(f"Loaded {len(all_prompts)} prompts")
    
    # Create a single timestamp for this batch of tests
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run tests for each model
    all_results = {}
    for model in models:
        logger.info(f"Running tests for model: {model}")
        
        model_providers = []
        
        # If we have a specific provider for this model, use it
        if providers and model in providers:
            provider_id = providers[model]
            provider_config = await ProviderConfig.get_provider(provider_id)
            if provider_config:
                model_providers.append(provider_config)
            else:
                logger.warning(f"Provider {provider_id} not found for model {model}")
        elif test_all_providers:
            # Test all enabled providers for this model
            model_providers = await ProviderConfig.find_providers_for_model(model)
        else:
            # Use the default provider for this model (or None for default routing)
            default_provider_id = await ProviderConfig.get_default_provider_for_model(model)
            if default_provider_id:
                provider_config = await ProviderConfig.get_provider(default_provider_id)
                if provider_config:
                    model_providers.append(provider_config)
        
        # If no providers specified or found, add None for default routing
        if not model_providers:
            model_providers.append(None)
        
        # Run tests for each provider
        model_results = {}
        
        if parallel and len(model_providers) > 1:
            # Run provider tests in parallel
            tasks = []
            for provider_config in model_providers:
                task = asyncio.create_task(
                    run_provider_tests(model, provider_config, all_prompts, output_dir, batch_timestamp)
                )
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error running provider tests: {result}")
                else:
                    provider_key, provider_results = result
                    model_results[provider_key] = provider_results
        else:
            # Run provider tests sequentially
            for provider_config in model_providers:
                provider_key, provider_results = await run_provider_tests(
                    model, provider_config, all_prompts, output_dir, batch_timestamp
                )
                model_results[provider_key] = provider_results
        
        all_results[model] = model_results
    
    # Save summary report
    try:
        report_path = Path(output_dir) / f"summary_report_{batch_timestamp}.json"
        with open(report_path, "w") as f:
            # Convert TestResult objects to dicts for JSON serialization
            json_results = {}
            for model, model_results in all_results.items():
                json_results[model] = {}
                for provider_key, provider_results in model_results.items():
                    json_results[model][provider_key] = [
                        json.loads(res.json()) if hasattr(res, 'json') else res 
                        for res in provider_results
                    ]
                    
            json.dump(json_results, f, indent=2, default=str)
        logger.info(f"Summary report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error saving summary report: {e}")
    
    # Return the overall results
    return all_results


async def main():
    parser = argparse.ArgumentParser(description="OpenRouter Provider Validator Test Runner")
    parser.add_argument("--models", nargs="+", help="Models to test")
    parser.add_argument("--prompts", nargs="+", help="Specific prompt IDs to test")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--all-providers", action="store_true", help="Test all providers for each model")
    parser.add_argument("--sequential", action="store_true", help="Run provider tests sequentially")
    args = parser.parse_args()
    
    await run_tests(
        models=args.models,
        prompts=args.prompts,
        output_dir=args.output_dir,
        test_all_providers=args.all_providers,
        parallel=not args.sequential
    )


if __name__ == "__main__":
    asyncio.run(main())