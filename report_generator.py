"""OpenRouter Provider Validator - Report Generator

Creates human-readable reports from test results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from client import FileSystemClient, ProviderSummary, TestResult
from metrics_extractor import extract_metrics_from_test_results

# Initialize client
client = FileSystemClient()

def generate_provider_report(provider_name: str, model_name: str) -> str:
    """Generate a report for a specific provider and model combination.
    
    Args:
        provider_name: Name of the provider to report on
        model_name: Model identifier
        
    Returns:
        Markdown formatted report content
    """
    # Load test results for this model
    results = client.load_test_results(model_name)
    
    # Filter to only this provider
    provider_results = [r for r in results if r.provider == provider_name]
    
    if not provider_results:
        return f"# {provider_name} Provider Report for {model_name}\n\nNo test results found for this provider/model combination.\n"
    
    # Extract metrics
    metrics = extract_metrics_from_test_results(provider_results)
    
    # Build the report
    report = []
    report.append(f"# {provider_name} Provider Report for {model_name}\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary section
    report.append("## Summary\n")
    report.append(f"- Total Tests: {metrics['total_tests']}\n")
    report.append(f"- Successful Tests: {metrics['successful_tests']} ({metrics['success_rate']:.2f}%)\n")
    report.append(f"- Failed Tests: {metrics['failed_tests']}\n")
    report.append(f"- Total Tokens Used: {metrics['total_tokens_used']}\n")
    
    # Token usage section
    report.append("## Token Usage\n")
    report.append(f"- Average Total Tokens: {metrics['avg_total_tokens']:.2f}\n")
    report.append(f"- Average Prompt Tokens: {metrics['avg_prompt_tokens']:.2f}\n")
    report.append(f"- Average Completion Tokens: {metrics['avg_completion_tokens']:.2f}\n")
    
    # Error analysis section
    if metrics['failed_tests'] > 0:
        report.append("## Error Analysis\n")
        report.append("### Error Categories\n")
        for category, count in metrics['error_categories'].items():
            report.append(f"- {category}: {count} ({(count / metrics['failed_tests'] * 100):.2f}%)\n")
        
        # Sample error messages
        report.append("\n### Sample Error Messages\n")
        error_samples = [r for r in provider_results if not r.success][:5]  # Up to 5 samples
        for sample in error_samples:
            report.append(f"- **Prompt ID**: {sample.prompt_id}\n")
            report.append(f"  **Error**: {sample.error_message}\n")
            report.append(f"  **Category**: {sample.error_category or 'unknown'}\n\n")
    
    # Tool usage section
    if metrics['tool_usage']:
        report.append("## Tool Usage\n")
        for tool, count in sorted(metrics['tool_usage'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {tool}: {count} calls\n")
    
    return "".join(report)

def generate_model_comparison_report(model_name: str) -> str:
    """Generate a comparison report across all providers for a specific model.
    
    Args:
        model_name: Model identifier to compare providers for
        
    Returns:
        Markdown formatted report content
    """
    # Get summaries for all providers of this model
    summaries = client.generate_provider_summary(model_name)
    
    if not summaries:
        return f"# Provider Comparison for {model_name}\n\nNo test results found for this model.\n"
    
    # Sort by success rate
    summaries.sort(key=lambda x: x.failure_rate)
    
    # Build the report
    report = []
    report.append(f"# Provider Comparison for {model_name}\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary table
    report.append("## Provider Performance\n")
    report.append("| Provider | Attempts | Success | Failure Rate | Main Error Types |\n")
    report.append("| --- | --- | --- | --- | --- |\n")
    
    for summary in summaries:
        # Format error categories
        error_types = ", ".join([f"{k} ({v})" for k, v in summary.error_categories.items()])
        error_types = error_types if error_types else "None"
        
        report.append(f"| {summary.provider} | {summary.total_attempts} | {summary.successful_attempts} | "
                   f"{summary.failure_rate:.2f}% | {error_types} |\n")
    
    return "".join(report)

def generate_summary_report() -> str:
    """Generate an overall summary report for all test results.
    
    Returns:
        Markdown formatted report content
    """
    # Get system stats
    stats = client.get_stats()
    
    # Get all model names
    models = client.list_models()
    
    # Build the report
    report = []
    report.append(f"# OpenRouter Provider Validator Summary Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # System stats section
    report.append("## System Statistics\n")
    report.append(f"- Total Providers: {stats['total_providers']}\n")
    report.append(f"- Enabled Providers: {stats['enabled_providers']}\n")
    report.append(f"- Total Test Prompts: {stats['total_prompts']}\n")
    report.append(f"- Models Tested: {stats['models_tested']}\n")
    report.append(f"- Total Test Results: {stats['total_test_results']}\n")
    report.append(f"- Successful Tests: {stats['successful_tests']}\n")
    report.append(f"- Failed Tests: {stats['failed_tests']}\n\n")
    
    # Model success rates
    if models:
        report.append("## Model Performance Summary\n")
        report.append("| Model | Providers | Attempts | Success Rate |\n")
        report.append("| --- | --- | --- | --- |\n")
        
        for model in models:
            # Get summaries for each model
            summaries = client.generate_provider_summary(model)
            
            if summaries:
                total_attempts = sum(s.total_attempts for s in summaries)
                total_success = sum(s.successful_attempts for s in summaries)
                success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0
                
                report.append(f"| {model} | {len(summaries)} | {total_attempts} | {success_rate:.2f}% |\n")
    
    return "".join(report)

def generate_error_pattern_report() -> str:
    """Generate a report focused on error patterns across all tests.
    
    Returns:
        Markdown formatted report content
    """
    # Get all results
    all_results = client.load_test_results()
    failed_results = [r for r in all_results if not r.success]
    
    if not failed_results:
        return "# Error Pattern Analysis\n\nNo failed tests found to analyze.\n"
    
    # Categorize errors
    error_categories = {}
    for result in failed_results:
        category = result.error_category or "unknown"
        if category not in error_categories:
            error_categories[category] = []
        error_categories[category].append(result)
    
    # Build the report
    report = []
    report.append(f"# Error Pattern Analysis\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"Total Failed Tests: {len(failed_results)}\n\n")
    
    # Error distribution section
    report.append("## Error Distribution\n")
    report.append("| Category | Count | Percentage |\n")
    report.append("| --- | --- | --- |\n")
    
    for category, errors in sorted(error_categories.items(), key=lambda x: len(x[1]), reverse=True):
        percentage = (len(errors) / len(failed_results) * 100)
        report.append(f"| {category} | {len(errors)} | {percentage:.2f}% |\n")
    
    # Detailed analysis of each category
    report.append("\n## Detailed Error Analysis\n")
    
    for category, errors in sorted(error_categories.items(), key=lambda x: len(x[1]), reverse=True):
        report.append(f"### {category} Errors\n")
        
        # Provider distribution
        provider_counts = {}
        for error in errors:
            provider_counts[error.provider] = provider_counts.get(error.provider, 0) + 1
        
        report.append("#### Provider Distribution\n")
        for provider, count in sorted(provider_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {provider}: {count} errors ({count / len(errors) * 100:.2f}%)\n")
        
        # Sample errors
        report.append("\n#### Sample Error Messages\n")
        samples = errors[:3]  # Take up to 3 samples
        for sample in samples:
            report.append(f"- **Provider**: {sample.provider}\n")
            report.append(f"  **Model**: {sample.model}\n")
            report.append(f"  **Prompt ID**: {sample.prompt_id}\n")
            report.append(f"  **Error**: {sample.error_message}\n\n")
    
    return "".join(report)

def save_report(report_name: str, content: str) -> bool:
    """Save a report to files.
    
    Args:
        report_name: Name of the report file
        content: Report content
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client.save_report(report_name, content)
        return True
    except Exception:
        return False

def generate_and_save_all_reports() -> Dict[str, bool]:
    """Generate and save all standard reports.
    
    Returns:
        Dictionary mapping report names to success status
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    
    # Summary report
    summary_report = generate_summary_report()
    results["summary"] = save_report(f"summary_{timestamp}", summary_report)
    
    # Error analysis report
    error_report = generate_error_pattern_report()
    results["error_analysis"] = save_report(f"error_analysis_{timestamp}", error_report)
    
    # Model comparison reports
    for model in client.list_models():
        model_name = model.replace("/", "_")
        comparison_report = generate_model_comparison_report(model)
        results[f"model_{model_name}"] = save_report(f"model_comparison_{model_name}_{timestamp}", comparison_report)
    
    return results
