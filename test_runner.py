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
from error_classifier import get_error_description

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

async def load_prompts(prompts_path="data/prompts.json", prompt_ids=None):
    """
    Load all prompts from the prompts JSON file.
    
    Args:
        prompts_path: Path to prompts JSON file
        prompt_ids: Optional list of specific prompt IDs to load
    
    Returns:
        List of prompt configurations
    """
    prompts = []
    
    try:
        prompts_file = Path(prompts_path)
        if not prompts_file.exists() or not prompts_file.is_file():
            logger.error(f"Prompts file not found: {prompts_file}")
            return []
        
        # Load all prompts from the JSON file
        with open(prompts_file, "r") as f:
            all_prompts = json.load(f)
            
        # Filter by prompt_ids if provided
        for prompt in all_prompts:
            if prompt_ids is None or prompt["id"] in prompt_ids:
                prompts.append(prompt)
                
    except Exception as e:
        logger.error(f"Error loading prompts file {prompts_path}: {e}")
    
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


def aggregate_errors(results):
    """
    Aggregate error categories from test results.
    
    Args:
        results: List of test results
        
    Returns:
        Dictionary of error categories and their counts
    """
    error_counts = {}
    for result in results:
        if not result.success and result.error_category:
            error_counts[result.error_category] = error_counts.get(result.error_category, 0) + 1
    return error_counts


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
    
    # Add error summary to results
    return result_key, {
        "results": provider_results,
        "error_summary": aggregate_errors(provider_results)
    }


async def generate_markdown_report(all_results, output_dir, batch_timestamp):
    """
    Generate a markdown summary report with error statistics.
    
    Args:
        all_results: Test results organized by model and provider
        output_dir: Directory to save the report
        batch_timestamp: Timestamp for this report
    """
    report_path = Path(output_dir) / f"summary_report_{batch_timestamp}.md"
    
    with open(report_path, "w") as f:
        # Write report header
        f.write(f"# OpenRouter Provider Validator Test Report\n\n")
        f.write(f"## Summary - {batch_timestamp}\n\n")
        
        # Write overall statistics
        total_tests = 0
        total_success = 0
        total_failure = 0
        model_stats = {}
        
        for model, model_results in all_results.items():
            model_total = 0
            model_success = 0
            model_failure = 0
            
            for provider_key, provider_data in model_results.items():
                if provider_key == "_aggregate":
                    continue
                    
                for result in provider_data["results"]:
                    model_total += 1
                    if result.success:
                        model_success += 1
                    else:
                        model_failure += 1
            
            total_tests += model_total
            total_success += model_success
            total_failure += model_failure
            
            model_stats[model] = {
                "total": model_total,
                "success": model_success,
                "failure": model_failure,
                "success_rate": f"{model_success / model_total * 100:.1f}%" if model_total > 0 else "N/A"
            }
        
        # Write overall statistics
        f.write(f"**Total Tests**: {total_tests}\n\n")
        f.write(f"**Success Rate**: {total_success / total_tests * 100:.1f}% ({total_success}/{total_tests})\n\n")
        
        # Write per-model statistics
        f.write("### Model Success Rates\n\n")
        f.write("| Model | Success | Failure | Success Rate |\n")
        f.write("| ----- | ------- | ------- | ------------ |\n")
        
        for model, stats in model_stats.items():
            f.write(f"| {model} | {stats['success']} | {stats['failure']} | {stats['success_rate']} |\n")
        
        f.write("\n")
        
        # Add error statistics section
        f.write("## Error Statistics\n\n")
        
        # Write per-model error statistics
        for model, model_results in all_results.items():
            f.write(f"### Model: {model}\n\n")
            
            # Aggregate error statistics across all providers for this model
            if "_aggregate" in model_results:
                f.write("#### Overall Error Distribution\n\n")
                f.write("| Error Category | Count | Description |\n")
                f.write("| -------------- | ----- | ----------- |\n")
                
                for category, count in sorted(
                    model_results["_aggregate"]["error_summary"].items(),
                    key=lambda x: x[1],  # Sort by count
                    reverse=True  # Highest count first
                ):
                    description = get_error_description(category)
                    f.write(f"| {category} | {count} | {description} |\n")
                
                f.write("\n")
            
            # Per-provider error statistics
            for provider_key, provider_data in model_results.items():
                if provider_key == "_aggregate":
                    continue
                    
                provider_display = await get_provider_display_name(provider_key)
                if "error_summary" in provider_data and provider_data["error_summary"]:
                    f.write(f"#### Provider: {provider_display}\n\n")
                    f.write("| Error Category | Count | Description |\n")
                    f.write("| -------------- | ----- | ----------- |\n")
                    
                    for category, count in sorted(
                        provider_data["error_summary"].items(),
                        key=lambda x: x[1],  # Sort by count
                        reverse=True  # Highest count first
                    ):
                        description = get_error_description(category)
                        f.write(f"| {category} | {count} | {description} |\n")
                    
                    f.write("\n")
    
    logger.info(f"Markdown summary report saved to {report_path}")


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
                    provider_key, provider_data = result
                    model_results[provider_key] = provider_data
        else:
            # Run provider tests sequentially
            for provider_config in model_providers:
                provider_key, provider_data = await run_provider_tests(
                    model, provider_config, all_prompts, output_dir, batch_timestamp
                )
                model_results[provider_key] = provider_data
        
        # Add aggregate error statistics across all providers for this model
        all_provider_errors = {}
        for provider_key, provider_data in model_results.items():
            for error_cat, count in provider_data["error_summary"].items():
                all_provider_errors[error_cat] = all_provider_errors.get(error_cat, 0) + count
        
        model_results["_aggregate"] = {"error_summary": all_provider_errors}
        all_results[model] = model_results
    
    # Save JSON summary report
    try:
        json_report_path = Path(output_dir) / f"summary_report_{batch_timestamp}.json"
        with open(json_report_path, "w") as f:
            # Convert test results to dicts for JSON serialization
            json_results = {}
            for model, model_results in all_results.items():
                json_results[model] = {}
                for provider_key, provider_data in model_results.items():
                    if provider_key == "_aggregate":
                        # Just include the error summary for aggregate data
                        json_results[model][provider_key] = {
                            "error_summary": provider_data["error_summary"]
                        }
                    else:
                        # Include both results and error summary for providers
                        json_results[model][provider_key] = {
                            "results": [
                                json.loads(res.model_dump_json()) if hasattr(res, 'model_dump_json') else res 
                                for res in provider_data["results"]
                            ],
                            "error_summary": provider_data["error_summary"]
                        }
            
            json.dump(json_results, f, indent=2, default=str)
        logger.info(f"JSON summary report saved to {json_report_path}")
    except Exception as e:
        logger.error(f"Error saving JSON summary report: {e}")
    
    # Generate markdown summary report
    try:
        await generate_markdown_report(all_results, output_dir, batch_timestamp)
    except Exception as e:
        logger.error(f"Error generating markdown summary report: {e}")
    
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