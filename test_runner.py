#!/usr/bin/env python
"""Test runner for automating OpenRouter Provider Validator tests."""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

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

async def run_tests(
    models: Optional[List[str]] = None,
    prompts: Optional[List[str]] = None,
    providers: Optional[Dict[str, str]] = None, 
    output_dir: Optional[str] = None,
    test_all_providers: bool = False
):
    """Run tests for specified models and prompts.
    
    Args:
        models: List of models to test (defaults to claude models)
        prompts: List of prompt IDs to test (defaults to all)
        providers: Dictionary mapping models to specific providers (optional)
        output_dir: Directory to save results (defaults to "results")
        test_all_providers: Whether to test all enabled providers for each model
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
                    
            provider_name = "Default Routing"
            provider_config = None
            try:
                provider_config = await ProviderConfig.get_provider(provider_id) if provider_id else None
            except Exception as e:
                logger.warning(f"Error getting provider config: {e}")
                
            if provider_config:
                provider_name = provider_config.get("name", provider_id)
                
            model_providers = [{"id": provider_id, "name": provider_name}]
        
        logger.info(f"Running tests for model {model} with {len(model_providers)} provider(s)")
        
        # Run tests for each provider of this model
        for provider_config in model_providers:
            provider_id = provider_config.get("id")
            provider_name = provider_config.get("name", provider_id if provider_id else "Default Routing")
            
            logger.info(f"Running tests with provider: {provider_name}")
            
            provider_results = []
            tester = ProviderTester(model=model, provider=provider_id)
            
            for prompt in all_prompts:
                logger.info(f"Running test: {prompt['id']}")
                result = await tester.run_test(prompt["id"])
                provider_results.append(result)
                
                # Save individual result
                provider_suffix = f"_{provider_id}" if provider_id else ""
                result_file = os.path.join(output_dir, f"{model.replace('/', '_')}_{prompt['id']}{provider_suffix}.json")
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)
            
            # Save results for this provider
            combined_results[model][provider_id or "default"] = provider_results
    
    # Save combined results
    combined_results_file = os.path.join(output_dir, f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(combined_results_file, "w") as f:
        json.dump(combined_results, f, indent=2)
    
    # Generate summary report
    report = generate_summary_report(combined_results)
    report_file = os.path.join(output_dir, f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_file, "w") as f:
        f.write(report)
    
    logger.info(f"Tests completed. Results saved to {output_dir}")
    logger.info(f"Summary report: {report_file}")
    
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
            total_successful += sum(1 for r in provider_results if r["success"])
    
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
            successful = sum(1 for r in provider_results if r["success"])
            success_rate = successful / len(provider_results) if provider_results else 0
            
            avg_tool_calls = sum(r["metrics"]["total_tool_calls"] for r in provider_results) / len(provider_results) if provider_results else 0
            avg_steps = sum(r["successful_steps"] / r["total_steps"] for r in provider_results) / len(provider_results) if provider_results else 0
            avg_latency = sum(r["metrics"]["latency_ms"] for r in provider_results) / len(provider_results) if provider_results else 0
            
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
                report.append(f"| {r['prompt_id']} | {'✓' if r['success'] else '✗'} | {r['successful_steps']}/{r['total_steps']} | {r['metrics']['total_tool_calls']} | {r['metrics']['latency_ms']}ms |")
            
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
        test_all_providers=args.all_providers
    )

if __name__ == "__main__":
    asyncio.run(main())
