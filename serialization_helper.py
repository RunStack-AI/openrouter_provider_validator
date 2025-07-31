#!/usr/bin/env python
"""Helper utilities for serializing complex objects to JSON."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import json

def serialize_datetime(dt):
    """Convert datetime objects to ISO format strings."""
    if isinstance(dt, datetime):
        return dt.isoformat()
    return dt

def serialize_message_part(part: Any) -> Dict[str, Any]:
    """Safely serialize a message part to a dictionary.
    
    Args:
        part: Message part object (TextPart, ToolCallPart, etc.)
        
    Returns:
        Dictionary with serializable values
    """
    if hasattr(part, 'model_dump'):
        # Pydantic v2 approach
        return part.model_dump()
    try:
        # Convert to dict and handle datetime objects
        part_dict = {k: serialize_datetime(v) for k, v in part.__dict__.items()}
        return part_dict
    except Exception:
        # Fallback to string representation if all else fails
        return {"content": str(part)}

def serialize_message(message: Any) -> Dict[str, Any]:
    """Serialize a message object with parts to a dictionary.
    
    Args:
        message: Message object with parts attribute
        
    Returns:
        Dictionary with serializable parts
    """
    try:
        # If message has parts, process them individually
        if hasattr(message, 'parts') and isinstance(message.parts, list):
            parts = []
            for part in message.parts:
                part_dict = serialize_message_part(part)
                parts.append({"content": part_dict})
            return {"parts": parts}
        elif hasattr(message, '__dict__'):
            # For other objects with __dict__
            return {k: serialize_datetime(v) for k, v in message.__dict__.items()} 
        else:
            # Fallback for unknown objects
            return {"content": str(message)}
    except Exception as e:
        # Last resort fallback
        return {"error": f"Failed to serialize message: {str(e)}"}

def json_serializable(obj: Any) -> Dict[str, Any]:
    """Convert any object to a JSON serializable form.
    
    Args:
        obj: Object to make serializable
        
    Returns:
        JSON serializable representation
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    if hasattr(obj, 'model_dump_json'):
        # Handle Pydantic v2 models with direct JSON serialization
        return json.loads(obj.model_dump_json())
    
    if hasattr(obj, 'model_dump'):
        # Handle Pydantic v2 models
        return obj.model_dump()
    
    if hasattr(obj, 'dict'):
        # Handle Pydantic v1 models
        return obj.dict()
    
    if hasattr(obj, '__dict__'):
        # Handle objects with __dict__
        result = {}
        for k, v in obj.__dict__.items():
            if not k.startswith('_'):  # Skip private attributes
                if isinstance(v, (str, int, float, bool, type(None))):
                    result[k] = v
                else:
                    result[k] = json_serializable(v)
        return result
    
    if isinstance(obj, (list, tuple)):
        # Handle lists and tuples
        return [json_serializable(x) for x in obj]
    
    if isinstance(obj, dict):
        # Handle dictionaries
        return {k: json_serializable(v) for k, v in obj.items()}
    
    # Fallback to string representation
    return str(obj)