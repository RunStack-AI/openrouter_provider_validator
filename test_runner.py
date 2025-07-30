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
    output_dir: Optional[str] = None
):
    """Run tests for specified models and prompts.
    
    Args:
        models: List of models to test (defaults to claude models)
        prompts: List of prompt IDs to test (defaults to all)
        providers: Dictionary mapping models to specific providers (optional)
        output_dir: Directory to save results (defaults to "results")
    """
    output_dir = output_dir or "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Default models
    models = models or [
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.7-haiku"
    ]
    
    # Load all prompts
    from client import FileSystemClient
    filesystem_client = FileSystemClient()
    all_prompts = filesystem_client.load_prompts()
    
    # Filter to specified prompts if provided
    if prompts:
        all_prompts = [p for p in all_prompts if p["id"] in prompts]
    
    results = {}
    
    # Run tests for each model
    for model in models:
        logger.info(f"Testing model: {model}")
        
        # Check if a specific provider was requested for this model
        provider = None
        if providers and model in providers:
            provider = providers[model]
            logger.info(f"Using specified provider: {provider}")
        else:
            # Check if we can auto-detect a provider
            provider = ProviderConfig.get_default_provider_for_model(model)
            if provider:
                logger.info(f"Auto-detected provider: {provider}")
            else:
                logger.info(f"No specific provider found for model {model}, using default routing")
        
        model_results = []
        tester = ProviderTester(model=model, provider=provider)
        
        for prompt in all_prompts:
            logger.info(f"Running test: {prompt['id']}")
            result = await tester.run_test(prompt["id"])
            model_results.append(result)
            
            # Save individual result
            result_file = os.path.join(output_dir, f"{model.replace('/', '_')}_{prompt['id']}.json")
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
                
        results[model] = model_results
    
    # Save combined results
    combined_results_file = os.path.join(output_dir, f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(combined_results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    report = generate_summary_report(results)
    report_file = os.path.join(output_dir, f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_file, "w") as f:
        f.write(report)
    
    logger.info(f"Tests completed. Results saved to {output_dir}")
    logger.info(f"Summary report: {report_file}")
    
    return results

def generate_summary_report(results):
    """Generate a markdown summary report from test results.
    
    Args:
        results: Dictionary of test results by model
        
    Returns:
        Markdown formatted report
    """
    report = ["# OpenRouter Provider Validator Test Report"]
    report.append(f"\n## Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Add overall statistics
    total_tests = sum(len(model_results) for model_results in results.values())
    total_successful = sum(sum(1 for r in model_results if r["success"]) for model_results in results.values())
    overall_success_rate = total_successful / total_tests if total_tests > 0 else 0
    
    report.append(f"Total tests: {total_tests}")
    report.append(f"Successful tests: {total_successful} ({overall_success_rate:.1%})\n")
    
    # Add model-specific sections
    report.append("## Results by Model\n")
    
    for model, model_results in results.items():
        # Get the provider name from the first result
        provider_name = model_results[0]["provider"] if model_results else "Unknown"
        
        successful = sum(1 for r in model_results if r["success"])
        success_rate = successful / len(model_results) if model_results else 0
        
        report.append(f"### {model} (Provider: {provider_name})")
        report.append(f"Tests: {len(model_results)}")
        report.append(f"Successful: {successful} ({success_rate:.1%})")
        
        # Calculate average metrics
        avg_tool_calls = sum(r["metrics"]["total_tool_calls"] for r in model_results) / len(model_results) if model_results else 0
        avg_step_success = sum(r["successful_steps"] / r["total_steps"] for r in model_results) / len(model_results) if model_results else 0
        avg_latency = sum(r["metrics"]["latency_ms"] for r in model_results) / len(model_results) if model_results else 0
        
        report.append(f"Average tool calls per test: {avg_tool_calls:.2f}")
        report.append(f"Average step success rate: {avg_step_success:.1%}")
        report.append(f"Average latency: {avg_latency:.0f}ms\n")
        
        # Add prompt-specific details
        report.append("| Prompt | Success | Steps | Tool Calls |")
        report.append("| ------ | ------- | ----- | ---------- |")
        
        for r in model_results:
            report.append(f"| {r['prompt_id']} | {'✓' if r['success'] else '✗'} | {r['successful_steps']}/{r['total_steps']} | {r['metrics']['total_tool_calls']} |")
        
        report.append("\n")
    
    return "\n".join(report)

async def main():
    parser = argparse.ArgumentParser(description="Run automated tests with OpenRouter Provider Validator")
    parser.add_argument("--models", nargs="+", help="Models to test (space separated)")
    parser.add_argument("--prompts", nargs="+", help="Specific prompt IDs to test (space separated)")
    parser.add_argument("--providers", nargs="+", help="Providers to use in format 'model:provider' (space separated)")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--list-providers", action="store_true", help="List available providers for each model")
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
            providers = ProviderConfig.find_providers_for_model(model, enabled_only=False)
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
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    asyncio.run(main())
