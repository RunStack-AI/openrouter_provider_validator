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


def get_provider_id(provider_config: Union[Dict, str, None]) -> Optional[str]:
    """
    Extract the provider ID from a provider configuration.
    Returns None for default routing.
    
    Args:
        provider_config: Provider configuration dict or direct ID string
        
    Returns:
        Provider ID string or None
    """
    if provider_config is None or provider_config == "":
        return None
    if isinstance(provider_config, dict) and "id" in provider_config:
        return provider_config["id"]
    return provider_config  # If it's already an ID


async def get_provider_display_name(provider_id: Optional[str], results: Optional[List[Dict]] = None) -> str:
    """
    Get a user-friendly provider name.
    Try to get from results first, then provider config, then ID.
    
    Args:
        provider_id: Provider ID
        results: Optional list of test results
        
    Returns:
        User-friendly provider name
    """
    if results and len(results) > 0 and "provider" in results[0]:
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
            # Create a failed result record
            provider_results.append({
                "model": model,
                "provider": provider_id or "unknown",
                "prompt_id": prompt["id"],
                "success": False,
                "error_message": str(e),
                "error_category": "execution_error",
                "timestamp": datetime.now().isoformat(),
                "total_steps": 0,
                "successful_steps": 0,
                "messages": [],
                "metrics": {}
            })
    
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
    test_helper = FileSystemTestHelper()
    all_prompts = test_helper.load_prompts()
    
    # Filter to specified prompts if provided
    if prompts:
        all_prompts = [p for p in all_prompts if p["id"] in prompts]
    
    combined_results = {}
    
    # Create templates directory if it doesn't exist
    templates_dir = Path("data/test_files/templates")
    if not templates_dir.exists():
        templates_dir.mkdir(exist_ok=True, parents=True)
        
        # Create sample template files if none exist
        with open(templates_dir / "sample1.txt", "w") as f:
            f.write("This is sample file 1\nIt has multiple lines\nFor testing file reading operations.")
        
        # Create a nested directory in templates
        nested_dir = templates_dir / "nested"
        nested_dir.mkdir(exist_ok=True, parents=True)
        
        with open(nested_dir / "sample3.txt", "w") as f:
            f.write("This is a nested file\nLocated in a subdirectory\nFor testing nested path operations.")
    
    # Run tests for each model
    for model in models:
        logger.info(f"Testing model: {model}")
        combined_results[model] = {}
        
        if test_all_providers:
            # Get all enabled providers for this model via API
            model_providers = await ProviderConfig.find_providers_for_model(model, enabled_only=True)
            if not model_providers:
                logger.warning(f"No enabled providers found for model {model}, using default routing")
                model_providers = [{"id": None, "name": "Default Routing"}]
        else:
            # Check if a specific provider was requested for this model
            provider_id = None
            if providers and model in providers:
                provider_id = providers[model]
                logger.info(f"Using specified provider: {provider_id}")
            else:
                # Check if we can auto-detect a provider
                provider_id = await ProviderConfig.get_default_provider_for_model(model)
                if provider_id:
                    logger.info(f"Auto-detected provider: {provider_id}")
                else:
                    logger.info(f"No specific provider found for model {model}, using default routing")
                    
            provider_name = await get_provider_display_name(provider_id)
            model_providers = [{"id": provider_id, "name": provider_name}]
        
        logger.info(f"Running tests for model {model} with {len(model_providers)} provider(s)")
        
        # Generate a timestamp for this batch of tests
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if parallel and len(model_providers) > 1:
            # Run provider tests in parallel using asyncio.gather
            logger.info(f"Running provider tests in parallel for model {model}")
            provider_tasks = [
                run_provider_tests(model, provider_config, all_prompts, output_dir, batch_timestamp)
                for provider_config in model_providers
            ]
            
            provider_results = await asyncio.gather(*provider_tasks)
            
            # Process results from parallel tasks
            for result_key, results in provider_results:
                combined_results[model][result_key] = results
        else:
            # Run provider tests sequentially
            logger.info(f"Running provider tests sequentially for model {model}")
            for provider_config in model_providers:
                result_key, provider_results = await run_provider_tests(
                    model, provider_config, all_prompts, output_dir, batch_timestamp
                )
                combined_results[model][result_key] = provider_results
    
    # Save combined results to the root results folder
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S') 
    combined_results_path = Path(output_dir) / f"combined_results_{timestamp_str}.json"
    
    with open(combined_results_path, "w") as f:
        json.dump(combined_results, f, indent=2, default=str)
    
    # Generate summary report
    report = generate_summary_report(combined_results)
    report_path = Path(output_dir) / f"summary_report_{timestamp_str}.md"
    
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Tests completed. Results saved to {output_dir}")
    logger.info(f"Summary report: {report_path}")
    
    return combined_results


def generate_summary_report(results):
    """Generate a markdown summary report from test results.
    
    Args:
        results: Nested dictionary of test results by model and provider
        
    Returns:
        Markdown formatted report
    """
    report = ["# OpenRouter Provider Validator Test Report"]
    report.append(f"\n## Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Count total tests
    total_tests = 0
    total_successful = 0
    
    for model_results in results.values():
        for provider_results in model_results.values():
            total_tests += len(provider_results)
            total_successful += sum(1 for r in provider_results if r.get("success", False))
    
    overall_success_rate = total_successful / total_tests if total_tests > 0 else 0
    
    report.append(f"Total tests: {total_tests}")
    report.append(f"Successful tests: {total_successful} ({overall_success_rate:.1%})\n")
    
    # Add model-specific sections
    report.append("## Results by Model and Provider\n")
    
    for model, model_results in results.items():
        report.append(f"### Model: {model}\n")
        
        # Provider comparison table
        report.append("| Provider | Tests | Success Rate | Avg Tool Calls | Avg Steps | Avg Latency |")
        report.append("| -------- | ----- | ----------- | -------------- | --------- | ----------- |")
        
        for provider_id, provider_results in model_results.items():
            if not provider_results:
                continue
                
            # Get provider name from the first result or use the ID
            provider_name = provider_results[0].get("provider", provider_id) if provider_results else provider_id
            
            # Calculate statistics
            successful = sum(1 for r in provider_results if r.get("success", False))
            success_rate = successful / len(provider_results) if provider_results else 0
            
            # Use defensive accessor pattern for metrics
            avg_tool_calls = sum(r.get("metrics", {}).get("total_tool_calls", 0) for r in provider_results) / len(provider_results) if provider_results else 0
            
            # Handle the case where total_steps might be 0
            avg_steps_values = []
            for r in provider_results:
                total_steps = r.get("total_steps", 0)
                successful_steps = r.get("successful_steps", 0) 
                if total_steps > 0:
                    avg_steps_values.append(successful_steps / total_steps)
            
            avg_steps = sum(avg_steps_values) / len(avg_steps_values) if avg_steps_values else 0
            avg_latency = sum(r.get("metrics", {}).get("latency_ms", 0) for r in provider_results) / len(provider_results) if provider_results else 0
            
            report.append(f"| {provider_name} | {len(provider_results)} | {success_rate:.1%} | {avg_tool_calls:.1f} | {avg_steps:.1%} | {avg_latency:.0f}ms |")
        
        report.append("\n")
        
        # Detailed results for each provider
        for provider_id, provider_results in model_results.items():
            if not provider_results:
                continue
                
            provider_name = provider_results[0].get("provider", provider_id) if provider_results else provider_id
            report.append(f"#### Provider: {provider_name}\n")
            
            report.append("| Prompt | Success | Steps | Tool Calls | Latency |")
            report.append("| ------ | ------- | ----- | ---------- | ------- |")
            
            for r in provider_results:
                # Handle potential missing keys with defaults
                success = "✓" if r.get("success", False) else "✗"
                successful_steps = r.get("successful_steps", 0)
                total_steps = r.get("total_steps", 0)
                tool_calls = r.get("metrics", {}).get("total_tool_calls", 0)
                latency = r.get("metrics", {}).get("latency_ms", 0)
                prompt_id = r.get("prompt_id", "unknown")
                
                report.append(f"| {prompt_id} | {success} | {successful_steps}/{total_steps} | {tool_calls} | {latency}ms |")
            
            report.append("\n")
    
    return "\n".join(report)


async def main():
    parser = argparse.ArgumentParser(description="Run automated tests with OpenRouter Provider Validator")
    parser.add_argument("--models", nargs="+", help="Models to test (space separated)")
    parser.add_argument("--prompts", nargs="+", help="Specific prompt IDs to test (space separated)")
    parser.add_argument("--providers", nargs="+", help="Providers to use in format 'model:provider' (space separated)")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--list-providers", action="store_true", help="List available providers for each model")
    parser.add_argument("--all-providers", action="store_true", help="Test all enabled providers for each model")
    parser.add_argument("--sequential", action="store_true", help="Run provider tests sequentially instead of in parallel")
    args = parser.parse_args()
    
    # Special case: list available providers for specified models
    if args.list_providers:
        models = args.models or [
            "anthropic/claude-3.7-sonnet",
            "anthropic/claude-3.7-haiku",
            "moonshot/kimi-k2"  # Add the Moonshot model
        ]
        
        print("\nAvailable providers by model:\n")
        for model in models:
            providers = await ProviderConfig.find_providers_for_model(model, enabled_only=False)
            print(f"Model: {model}")
            if providers:
                for provider in providers:
                    status = "Enabled" if provider.get("enabled", True) else "Disabled"
                    print(f"  - {provider['name']} (ID: {provider['id']}) - {status}")
                    print(f"    {provider.get('description', '')}")
            else:
                print("  No providers configured for this model")
            print("")
        return
    
    # Parse provider mappings if provided
    provider_dict = {}
    if args.providers:
        for mapping in args.providers:
            parts = mapping.split(":")
            if len(parts) == 2:
                provider_dict[parts[0]] = parts[1]
    
    # Run the tests
    await run_tests(
        models=args.models, 
        prompts=args.prompts, 
        providers=provider_dict, 
        output_dir=args.output_dir,
        test_all_providers=args.all_providers,
        parallel=not args.sequential  # Use parallel by default unless --sequential flag is set
    )


if __name__ == "__main__":
    asyncio.run(main())