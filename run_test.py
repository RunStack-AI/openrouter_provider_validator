#!/usr/bin/env python
"""Helper script to run a single test with the OpenRouter Provider Validator."""

import asyncio
import json
import sys

from agent import ProviderTester

async def run_example_test():
    # Create a tester with Claude
    tester = ProviderTester(model="anthropic/claude-3.7-sonnet")
    
    # Check for command line argument for prompt ID
    prompt_id = sys.argv[1] if len(sys.argv) > 1 else "file_operations_sequence"
    
    # Run a test with the specified prompt
    print(f"Running test with prompt: {prompt_id}")
    result = await tester.run_test(prompt_id)
    
    # Print result summary
    print(f"\nTest Result:")
    print(f"Success: {result['success']}")
    
    if not result['success']:
        print(f"Error: {result.get('error_message', 'Unknown error')}")
        print(f"Category: {result.get('error_category', 'uncategorized')}")
    else:
        print(f"Model: {result['model']}")
        print(f"Provider: {result['provider']}")
        print(f"Tool calls: {result['metrics']['tool_calls']}")
        
        # Show token usage if available
        if 'token_usage' in result:
            print(f"Token usage: {result['token_usage']}")
    
    # Save the result to a file for inspection
    with open("last_test_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Full result saved to 'last_test_result.json'")

if __name__ == "__main__":
    asyncio.run(run_example_test())
