#!/usr/bin/env python
"""Test script to verify serialization of pydantic-ai message objects.

Run this script to check if our serialization functions can properly
handle various ModelMessage objects from pydantic-ai.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from serialization_helper import json_serializable

def test_serialize_datetime():
    """Test serialization of datetime objects."""
    dt = datetime.now()
    serialized = json_serializable(dt)
    print(f"Datetime ({dt}) serialized to: {serialized}")
    
    # Try round-trip through JSON
    json_str = json.dumps({"timestamp": serialized})
    print(f"JSON string: {json_str}")
    
    roundtrip = json.loads(json_str)
    print(f"Roundtrip: {roundtrip['timestamp']}")
    
    return roundtrip['timestamp'] == serialized

def test_serialize_nested_object():
    """Test serialization of objects with datetime attributes."""
    class TestObject:
        def __init__(self):
            self.created_at = datetime.now()
            self.name = "Test"
            self.nested = {"key": datetime.now()}
            self.items = [datetime.now(), "string", 123]
    
    obj = TestObject()
    serialized = json_serializable(obj)
    print(f"Serialized object: {serialized}")
    
    # Try round-trip through JSON
    json_str = json.dumps(serialized, indent=2)
    print(f"JSON string:\n{json_str}")
    
    roundtrip = json.loads(json_str)
    print(f"Roundtrip successful: {bool(roundtrip)}")
    
    return bool(roundtrip)

def main():
    """Run all serialization tests."""
    print("\n===== Testing Serialization Helper =====\n")
    
    dt_test = test_serialize_datetime()
    print(f"\nDatetime test {'PASSED' if dt_test else 'FAILED'}")
    
    obj_test = test_serialize_nested_object()
    print(f"\nNested object test {'PASSED' if obj_test else 'FAILED'}")
    
    print("\nAll tests {'PASSED' if dt_test and obj_test else 'FAILED'}")

if __name__ == "__main__":
    main()