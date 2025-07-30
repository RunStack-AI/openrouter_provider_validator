"""OpenRouter Provider Validator - Metrics Extractor

Extracts metrics from responses and logs for analysis.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple

from client import TestResult

# Set up logging
logger = logging.getLogger("metrics_extractor")

def extract_response_metrics(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from an OpenAI-compatible response.
    
    Args:
        response_data: Raw response data from the API
        
    Returns:
        Dictionary of extracted metrics
    """
    metrics = {
        "id": None,              # Response ID
        "model": None,          # Actual model used
        "created": None,        # Timestamp
        "completion_tokens": None,  # Tokens in completion
        "prompt_tokens": None,     # Tokens in prompt
        "total_tokens": None,      # Total tokens used
        "latency_ms": None,     # Response time in ms
        "finish_reason": None   # Why the generation stopped
    }
    
    # Extract standard OpenAI fields
    if "id" in response_data:
        metrics["id"] = response_data["id"]
    
    if "model" in response_data:
        metrics["model"] = response_data["model"]
    
    if "created" in response_data:
        metrics["created"] = response_data["created"]
        metrics["created_iso"] = datetime.fromtimestamp(response_data["created"]).isoformat()
    
    if "usage" in response_data:
        usage = response_data["usage"]
        if "completion_tokens" in usage:
            metrics["completion_tokens"] = usage["completion_tokens"]
        if "prompt_tokens" in usage:
            metrics["prompt_tokens"] = usage["prompt_tokens"]
        if "total_tokens" in usage:
            metrics["total_tokens"] = usage["total_tokens"]
    
    # Extract finish reason from choices if available
    if "choices" in response_data and len(response_data["choices"]) > 0:
        if "finish_reason" in response_data["choices"][0]:
            metrics["finish_reason"] = response_data["choices"][0]["finish_reason"]
    
    # Extract response latency if available
    if "response_ms" in response_data:
        metrics["latency_ms"] = response_data["response_ms"]
    
    return metrics

def extract_tool_usage(response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract information about tools used in the response.
    
    Args:
        response_data: Raw response data from the API
        
    Returns:
        List of tools used in the response
    """
    tools_used = []
    
    # Extract tool calls from choices if available
    if "choices" in response_data and len(response_data["choices"]) > 0:
        choice = response_data["choices"][0]
        
        # Check for tool calls in message
        if "message" in choice and "tool_calls" in choice["message"]:
            for tool_call in choice["message"]["tool_calls"]:
                tools_used.append({
                    "id": tool_call.get("id"),
                    "type": tool_call.get("type"),
                    "function": {
                        "name": tool_call.get("function", {}).get("name"),
                        "arguments": tool_call.get("function", {}).get("arguments")
                    }
                })
    
    return tools_used

def extract_metrics_from_test_results(results: List[TestResult]) -> Dict[str, Any]:
    """Extract and aggregate metrics from a list of test results.
    
    Args:
        results: List of test results to analyze
        
    Returns:
        Dictionary of aggregated metrics
    """
    metrics = {
        "total_tests": len(results),
        "successful_tests": sum(1 for r in results if r.success),
        "failed_tests": sum(1 for r in results if not r.success),
        "success_rate": 0.0,
        "avg_total_tokens": 0,
        "avg_prompt_tokens": 0,
        "avg_completion_tokens": 0,
        "total_tokens_used": 0,
        "error_categories": {},
        "tool_usage": {}
    }
    
    if metrics["total_tests"] > 0:
        metrics["success_rate"] = (metrics["successful_tests"] / metrics["total_tests"]) * 100
    
    # Collect token usage statistics
    token_counts = []
    prompt_token_counts = []
    completion_token_counts = []
    
    for result in results:
        # Process success/failure metrics
        if not result.success:
            error_cat = result.error_category or "unknown"
            metrics["error_categories"][error_cat] = metrics["error_categories"].get(error_cat, 0) + 1
        
        # Process token usage
        if result.token_usage:
            if "total_tokens" in result.token_usage:
                token_counts.append(result.token_usage["total_tokens"])
                metrics["total_tokens_used"] += result.token_usage["total_tokens"]
            
            if "prompt_tokens" in result.token_usage:
                prompt_token_counts.append(result.token_usage["prompt_tokens"])
            
            if "completion_tokens" in result.token_usage:
                completion_token_counts.append(result.token_usage["completion_tokens"])
        
        # Process tool usage
        if result.response_data and result.success:
            tools = extract_tool_usage(result.response_data)
            for tool in tools:
                tool_name = tool.get("function", {}).get("name")
                if tool_name:
                    metrics["tool_usage"][tool_name] = metrics["tool_usage"].get(tool_name, 0) + 1
    
    # Calculate averages
    if token_counts:
        metrics["avg_total_tokens"] = sum(token_counts) / len(token_counts)
    if prompt_token_counts:
        metrics["avg_prompt_tokens"] = sum(prompt_token_counts) / len(prompt_token_counts)
    if completion_token_counts:
        metrics["avg_completion_tokens"] = sum(completion_token_counts) / len(completion_token_counts)
    
    return metrics

def extract_log_metrics(log_content: str) -> Dict[str, Any]:
    """Extract metrics from log files.
    
    Args:
        log_content: Raw log file content
        
    Returns:
        Dictionary of metrics extracted from logs
    """
    metrics = {
        "request_count": 0,
        "error_count": 0,
        "rate_limit_count": 0,
        "timeout_count": 0,
        "providers": {}
    }
    
    # Count requests
    request_matches = re.findall(r"Sending request to OpenRouter", log_content, re.IGNORECASE)
    metrics["request_count"] = len(request_matches)
    
    # Count errors
    error_matches = re.findall(r"Error|Exception|Failed", log_content, re.IGNORECASE)
    metrics["error_count"] = len(error_matches)
    
    # Count rate limits
    rate_limit_matches = re.findall(r"rate limit|too many requests|429", log_content, re.IGNORECASE)
    metrics["rate_limit_count"] = len(rate_limit_matches)
    
    # Count timeouts
    timeout_matches = re.findall(r"timeout|connection|504", log_content, re.IGNORECASE)
    metrics["timeout_count"] = len(timeout_matches)
    
    # Extract provider information
    provider_matches = re.findall(r"Testing provider ([\w-]+) with model ([\w\-\/\.]+)", log_content)
    for provider, model in provider_matches:
        if provider not in metrics["providers"]:
            metrics["providers"][provider] = {
                "models": {},
                "request_count": 0
            }
        
        if model not in metrics["providers"][provider]["models"]:
            metrics["providers"][provider]["models"][model] = 0
        
        metrics["providers"][provider]["models"][model] += 1
        metrics["providers"][provider]["request_count"] += 1
    
    return metrics
