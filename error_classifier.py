"""OpenRouter Provider Validator - Error Classifier

Classifies errors encountered during testing into standardized categories.
"""

import re
from typing import Dict, Optional, Any, List, Tuple

def classify_error(status_code: Optional[int], error_message: str) -> str:
    """Classify an error into a standardized category.
    
    Args:
        status_code: HTTP status code (if available)
        error_message: Error message text
        
    Returns:
        Standardized error category
    """
    error_message = str(error_message).lower()
    
    # Check for rate limit errors
    if status_code == 429 or any(pattern in error_message for pattern in [
        "rate limit", 
        "too many requests", 
        "ratelimit", 
        "quota exceeded",
        "request limit"
    ]):
        return "rate_limit_error"
    
    # Check for authentication errors
    if status_code in (401, 403) or any(pattern in error_message for pattern in [
        "unauthorized", 
        "authentication", 
        "auth", 
        "invalid key", 
        "invalid api key",
        "permission",
        "forbidden"
    ]):
        return "authentication_error"
    
    # Check for server errors
    if (status_code and status_code >= 500) or any(pattern in error_message for pattern in [
        "server error", 
        "internal error", 
        "service unavailable"
    ]):
        return "server_error"
    
    # Check for timeout errors
    if any(pattern in error_message for pattern in [
        "timeout", 
        "timed out", 
        "deadline exceeded"
    ]):
        return "timeout_error"
    
    # Check for content filtering/moderation errors
    if any(pattern in error_message for pattern in [
        "content policy", 
        "content filter", 
        "moderation", 
        "inappropriate", 
        "harmful content",
        "violates policy"
    ]):
        return "content_filter_error"
    
    # Check for input validation errors (enhanced to catch Pydantic validation errors)
    if status_code == 400 or any(pattern in error_message for pattern in [
        "invalid request", 
        "validation error", 
        "invalid parameter", 
        "bad request",
        "pydantic",  # Added pattern for Pydantic errors
        "errors.pydantic.dev",  # Added pattern for Pydantic error URLs
        "validation errors",  # Added pattern for multiple validation errors
        "not valid",  # Added common validation phrase
        "field required",  # Added common validation phrase
        "type error",  # Added common validation error message
        "input should be"  # Added common validation error message
    ]):
        return "input_validation_error"
    
    # Check for tool-related errors
    if any(pattern in error_message for pattern in [
        "tool", 
        "function", 
        "not found", 
        "invalid tool", 
        "invalid function",
        "not supported"
    ]):
        return "tool_usage_error"
    
    # Check for token/context length errors
    if any(pattern in error_message for pattern in [
        "token", 
        "context length", 
        "too long", 
        "maximum context"
    ]):
        return "token_limit_error"
    
    # Check for request formatting errors
    if any(pattern in error_message for pattern in [
        "json", 
        "format", 
        "malformed", 
        "syntax", 
        "parsing"
    ]):
        return "request_format_error"
    
    # Check for provider-specific errors
    if any(pattern in error_message for pattern in [
        "provider", 
        "model", 
        "routing", 
        "not available"
    ]):
        return "provider_error"
    
    # Default to unknown/other
    return "unknown_error"

def get_error_description(category: str) -> str:
    """Get a human-readable description of an error category.
    
    Args:
        category: The error category
        
    Returns:
        Human-readable description
    """
    descriptions = {
        "rate_limit_error": "Rate limits exceeded or too many requests in a time period",
        "authentication_error": "Authentication or authorization problems",
        "server_error": "Server-side errors or service unavailability",
        "timeout_error": "Request timed out or took too long to complete",
        "content_filter_error": "Content was flagged or filtered by safety measures",
        "input_validation_error": "Invalid input parameters, validation failures or schema mismatch",
        "tool_usage_error": "Issues with function or tool usage/format",
        "token_limit_error": "Exceeded token or context length limits",
        "request_format_error": "Malformed request or formatting problems",
        "provider_error": "Provider-specific errors or routing issues",
        "unknown_error": "Unclassified or unknown error type",
        "configuration_error": "Local configuration issues or missing settings",
        "max_retries_exceeded": "Maximum retry attempts reached without success",
        "execution_error": "General error during test execution"
    }
    
    return descriptions.get(category, "No description available")

def analyze_error_patterns(error_messages: List[str]) -> Dict[str, int]:
    """Analyze patterns across multiple errors to find common issues.
    
    Args:
        error_messages: List of error message strings
        
    Returns:
        Dictionary of error patterns and their frequencies
    """
    patterns = {}
    
    for message in error_messages:
        category = classify_error(None, message)
        patterns[category] = patterns.get(category, 0) + 1
    
    return patterns
