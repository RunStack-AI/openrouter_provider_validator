"""OpenRouter Provider Validator - Error Classifier

Analyzes error messages to categorize them for reporting and analysis.
"""

import re
from typing import Dict, List, Optional

# Define error categories and patterns to identify them
ERROR_CATEGORIES = {
    "rate_limit": ["rate limit", "too many requests", "429"],
    "authentication": ["unauthorized", "invalid api key", "401"],
    "provider_error": ["provider error", "upstream error", "502"],
    "tool_format": ["tool", "function", "invalid format", "invalid json"],
    "timeout": ["timeout", "connection", "504", "request timed out"],
    "model_unavailable": ["model not available", "503", "model is overloaded"],
    "context_length": ["context length", "too long", "token limit"],
    "content_filter": ["content filter", "content policy", "violates", "inappropriate"]
}

def classify_error(error_message: str) -> str:
    """Classify an error message into one of the predefined categories.
    
    Args:
        error_message: The error message text to classify
        
    Returns:
        Error category name, or 'unknown' if no match is found
    """
    if not error_message:
        return "unknown"
    
    # Normalize the error message for matching
    normalized_error = error_message.lower()
    
    # Check each category for matching patterns
    for category, patterns in ERROR_CATEGORIES.items():
        for pattern in patterns:
            if pattern.lower() in normalized_error:
                return category
    
    # Default category if no match is found
    return "unknown"

def get_error_categories() -> Dict[str, List[str]]:
    """Get the defined error categories and their patterns.
    
    Returns:
        Dictionary mapping category names to lists of patterns
    """
    return ERROR_CATEGORIES

def analyze_error_distribution(error_counts: Dict[str, int]) -> Dict[str, float]:
    """Analyze the distribution of errors across categories.
    
    Args:
        error_counts: Dictionary mapping category names to error counts
        
    Returns:
        Dictionary mapping category names to percentage of total errors
    """
    total_errors = sum(error_counts.values())
    if total_errors == 0:
        return {}
    
    distribution = {}
    for category, count in error_counts.items():
        distribution[category] = (count / total_errors) * 100
    
    return distribution

def extract_error_details(error_message: str) -> Dict[str, Optional[str]]:
    """Extract specific details from error messages using regex patterns.
    
    Args:
        error_message: The error message text to analyze
        
    Returns:
        Dictionary containing extracted error details
    """
    details = {
        "status_code": None,
        "request_id": None,
        "error_type": None,
        "error_code": None
    }
    
    # Extract HTTP status code
    status_code_match = re.search(r'status[\s_-]code[:\s]*(\d+)', error_message, re.IGNORECASE)
    if status_code_match:
        details["status_code"] = status_code_match.group(1)
    
    # Extract request ID if present
    request_id_match = re.search(r'req[\s_-]?(?:uest)?[\s_-]?(?:id)?[:\s]*([a-zA-Z0-9-]+)', error_message, re.IGNORECASE)
    if request_id_match:
        details["request_id"] = request_id_match.group(1)
    
    # Extract error type if present
    error_type_match = re.search(r'error[\s_-](?:type)?[:\s]*([a-zA-Z0-9_]+)', error_message, re.IGNORECASE)
    if error_type_match:
        details["error_type"] = error_type_match.group(1)
    
    # Extract error code if present
    error_code_match = re.search(r'error[\s_-]code[:\s]*([a-zA-Z0-9_]+)', error_message, re.IGNORECASE)
    if error_code_match:
        details["error_code"] = error_code_match.group(1)
    
    return details
