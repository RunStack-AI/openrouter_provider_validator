"""OpenRouter Provider Validator - Validation Error Report Generator

Generates specialized reports focused on validation errors.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from client import FileSystemClient

def generate_validation_error_report(client, models=None, batch_timestamp=None):
    """Generate a report focused on validation errors across models and providers.
    
    Args:
        client: FileSystemClient instance
        models: Optional list of models to include
        batch_timestamp: Optional timestamp for the report filename
        
    Returns:
        Report content as string
    """
    if not models:
        models = client.list_models()
    
    if not batch_timestamp:
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = []
    report.append("# Validation Error Analysis Report\n")
    report.append("This report analyzes validation errors that occurred during test executions, including those in otherwise successful tests.\n\n")
    
    total_tests = 0
    tests_with_validation_errors = 0
    perfect_success_tests = 0
    total_validation_errors = 0
    
    # Common validation error patterns
    error_patterns = {}
    model_validation_stats = {}
    
    for model in models:
        results = client.load_test_results(model)
        model_tests = len(results)
        model_errors = 0
        model_perfect = 0
        provider_stats = {}
        
        for result in results:
            total_tests += 1
            provider = result.provider
            
            # Initialize provider stats if needed
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "total": 0,
                    "success": 0,
                    "perfect_success": 0,
                    "validation_errors": 0
                }
            
            provider_stats[provider]["total"] += 1
            
            if result.success:
                provider_stats[provider]["success"] += 1
            
            # Check for perfect_success
            perfect_success = getattr(result, "perfect_success", result.success)
            if perfect_success:
                provider_stats[provider]["perfect_success"] += 1
                model_perfect += 1
                perfect_success_tests += 1
            
            # Track validation errors
            validation_error_count = getattr(result, "validation_error_count", 0)
            if validation_error_count > 0:
                provider_stats[provider]["validation_errors"] += validation_error_count
                model_errors += validation_error_count
                total_validation_errors += validation_error_count
                tests_with_validation_errors += 1
                
                # Analyze error patterns
                validation_errors = getattr(result, "validation_errors", [])
                for error in validation_errors:
                    error_msg = error.get("message", "").lower()
                    
                    # Extract error type
                    error_type = "other_validation_error"
                    
                    if "type=model_type" in error_msg:
                        error_type = "model_type_error"
                    elif "type=type_error" in error_msg:
                        error_type = "type_error"
                    elif "type=value_error" in error_msg:
                        error_type = "value_error"
                    elif "type=missing" in error_msg or "field required" in error_msg:
                        error_type = "missing_field_error"
                    elif "url" in error_msg or "uri" in error_msg:
                        error_type = "url_format_error"
                    elif "json" in error_msg:
                        error_type = "json_format_error"
                    
                    error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        # Store model validation stats
        model_validation_stats[model] = {
            "total_tests": model_tests,
            "validation_errors": model_errors,
            "perfect_success": model_perfect,
            "provider_stats": provider_stats
        }
    
    # Overall statistics
    if total_tests > 0:
        error_rate = (tests_with_validation_errors / total_tests) * 100
        perfect_rate = (perfect_success_tests / total_tests) * 100
        avg_errors = total_validation_errors / total_tests
        
        report.append("## Overall Statistics\n")
        report.append(f"- **Total Tests**: {total_tests}\n")
        report.append(f"- **Tests with Validation Errors**: {tests_with_validation_errors} ({error_rate:.2f}%)\n")
        report.append(f"- **Perfect Success Rate**: {perfect_rate:.2f}%\n")
        report.append(f"- **Total Validation Errors**: {total_validation_errors}\n")
        report.append(f"- **Average Validation Errors per Test**: {avg_errors:.2f}\n\n")
        
        # Error pattern summary
        report.append("## Validation Error Patterns\n")
        report.append("| Error Type | Count | Percentage |\n")
        report.append("| ---------- | ----- | ---------- |\n")
        
        for error_type, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_validation_errors) * 100 if total_validation_errors > 0 else 0
            report.append(f"| {error_type} | {count} | {percentage:.2f}% |\n")
        
        report.append("\n")
        
        # Per-model statistics
        report.append("## Validation Errors by Model\n")
        report.append("| Model | Total Tests | Success Rate | Perfect Success Rate | Validation Errors | Avg Errors/Test |\n")
        report.append("| ----- | ----------- | ----------- | -------------------- | ----------------- | --------------- |\n")
        
        for model, stats in model_validation_stats.items():
            total = stats["total_tests"]
            if total == 0:
                continue
                
            perfect_rate = (stats["perfect_success"] / total) * 100
            error_count = stats["validation_errors"]
            avg_errors = error_count / total
            
            report.append(f"| {model} | {total} | {perfect_rate:.2f}% | {error_count} | {avg_errors:.2f} |\n")
        
        report.append("\n")
        
        # Per-provider statistics
        for model, stats in model_validation_stats.items():
            report.append(f"### Model: {model}\n\n")
            report.append("| Provider | Tests | Success Rate | Perfect Success Rate | Validation Errors | Avg Errors/Test |\n")
            report.append("| -------- | ----- | ----------- | -------------------- | ----------------- | --------------- |\n")
            
            for provider, provider_stats in stats["provider_stats"].items():
                total = provider_stats["total"]
                if total == 0:
                    continue
                    
                success_rate = (provider_stats["success"] / total) * 100
                perfect_rate = (provider_stats["perfect_success"] / total) * 100
                error_count = provider_stats["validation_errors"]
                avg_errors = error_count / total
                
                report.append(f"| {provider} | {total} | {success_rate:.2f}% | {perfect_rate:.2f}% | {error_count} | {avg_errors:.2f} |\n")
            
            report.append("\n")
    
    # Save report
    report_content = "".join(report)
    client.save_report(f"validation_errors_{batch_timestamp}", report_content)
    
    return report_content


def main():
    client = FileSystemClient()
    generate_validation_error_report(client)


if __name__ == "__main__":
    main()
