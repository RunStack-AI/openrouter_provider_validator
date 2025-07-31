# Directory Structure Fix

## Issue Description
The validator has been experiencing problems with inconsistent directory structures for test results which caused the error:

```
FileNotFoundError: [Errno 2] No such file or directory: 'results/moonshotai_kimi-k2_file_operations_sequence_fireworks/fp8_20250731_221421.json'
```

The directory structure was convoluted and inconsistently organized, making it difficult to find and manage test results.

## Current Structure vs. Desired Structure

### Current (Problematic) Structure
```
results/
├── moonshotai_kimi-k2_file_operations_sequence_fireworks/
│   └── fp8_20250731_221421.json
```

The current structure combines model, prompt ID, and provider information into a single directory name, leading to very long directory names and confusion about which part is which.

### Desired Structure
```
results/
├── moonshotai_kimi-k2/               # Model
│   └── file_operations_sequence/      # Prompt ID
│       └── fireworks_fp8/             # Provider + Variant
│           └── 20250731_221421.json   # Timestamp-based file
```

This structure is more logical, with a clear hierarchy that separates model, prompt type, and provider information.

## Changes Made

### 1. Fixed `agent.py`
Updated the directory structure creation in `agent.py` to use a more logical hierarchy:

```python
# Before
subdir = result_dir / f"{model_safe}_{prompt_id}_{provider_safe}"
provider_dir = subdir / provider_safe
result_file = provider_dir / f"{timestamp}.json"

# After
model_dir = result_dir / model_safe
prompt_dir = model_dir / prompt_id_safe
provider_dir = prompt_dir / provider_variant
result_file = provider_dir / f"{timestamp}.json"
```

### 2. Fixed `client.py`
Updated the result loading and saving functions in `client.py` to match the new hierarchy:

```python
# New loading pattern
model_dir = results_dir / model.replace("/", "_")
for prompt_dir in model_dir.iterdir():
    for provider_dir in prompt_dir.iterdir():
        for result_file in provider_dir.glob("*.json"):
            # Process file here
```

The save function was also updated with more reliable path handling.

## How to Apply the Fixes

We've provided the following files to apply these changes:

1. **Updated `agent.py`** - Already modified with the new directory structure

2. **`client_directory_fix.py`** - Run this script to update the client.py file with the new directory structure:
   ```bash
   python client_directory_fix.py
   ```

3. **Optional Result Migration** - If you want to migrate existing results to the new structure, uncomment the migration function in client_directory_fix.py.

## Benefits

1. **More Organized**: Clear hierarchy makes results easier to navigate
2. **Reduced Path Length**: Shorter directory names avoid potential path length issues
3. **Better Grouping**: Results are logically grouped by model, prompt, and provider
4. **Consistent with Reporting**: Directory structure now matches the reports organization
5. **Path Finding**: Consistent path structure makes it easier to find reports by pattern matching

## Verification
After applying the fixes, run a test to verify the correct structure is created:

```bash
python agent.py --model anthropic/claude-3.7-sonnet --prompt file_operations_sequence 
```

Check that the results are saved in the expected location with the new structure:
```
results/anthropic_claude-3.7-sonnet/file_operations_sequence/default/YYYYMMDD_HHMMSS.json
```