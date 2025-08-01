# Validation Error Tracking in OpenRouter Provider Validator

## Overview

The OpenRouter Provider Validator tracks validation errors encountered during model testing. These errors occur when a model attempts to use a tool but provides invalid parameters according to the Pydantic schema, resulting in errors like:

```
Error executing tool read_file: 1 validation error for read_fileArguments
field required [type=missing, input_value={'path': None}, input_type=dict]
For further information visit https://errors.pydantic.dev/2.11/v/missing
```

## How Error Tracking Works

1. **Detection**: When a model makes a tool call with invalid parameters, the MCP server returns a validation error.

2. **Capture**: The `agent.py` file detects these errors in the `send_message` method by scanning message parts for validation error patterns. When found, they are recorded in the `validation_errors` array, including:
   - The error message
   - The tool name
   - A timestamp

3. **Accumulation**: Throughout a test run, all validation errors are accumulated in the `all_validation_errors` list within the `run_test` method in `agent.py`.

4. **Result Storage**: The TestResult object includes:
   - `validation_errors`: A list containing detailed information about each validation error
   - `validation_error_count`: The total number of validation errors encountered
   - `perfect_success`: A boolean indicating if the test was successful with NO validation errors

5. **Analysis**: The `test_runner.py` file analyzes validation errors using the `extract_validation_errors` function, which categorizes them into types like:
   - `model_type_error`: Incorrect model type (e.g., expected a dict, got a string)
   - `type_error`: Incorrect data type (e.g., int vs string)
   - `value_error`: Value is not valid despite having the correct type
   - `missing_field_error`: Required field is missing
   - `url_format_error`: URL format is invalid
   - `json_format_error`: JSON format is invalid
   - `generic_validation_error`: Other validation errors

6. **Reporting**: Validation errors are included in both JSON and Markdown reports, showing:
   - Total validation error count per test
   - Aggregated validation error types across providers and models
   - Detailed breakdowns by error type

## Key Considerations

- **Successful Tests Can Have Validation Errors**: A test can succeed overall (completing all required steps) but still encounter validation errors during execution when the model retries after errors

- **Perfect Success vs Success**: A test with `success=true` means all steps completed, while `perfect_success=true` means all steps completed with zero validation errors

- **Validation Error Patterns**: The system uses patterns like "validation error", "errors.pydantic.dev", "field required", etc. to detect validation errors

## Using Validation Error Data

The validation error data is valuable for:

1. **Model Comparison**: Identify which models are better at following tool schemas correctly

2. **Provider Comparison**: Compare validation error rates across different providers for the same model

3. **Tool Design Feedback**: Identify which tools or parameters frequently cause validation errors

4. **Improve MCP Protocol**: Use validation error patterns to improve tool definitions or documentation

## Example Report Section

The summary report includes a section like this for validation errors:

```markdown
### Validation Error Details

| Error Type | Count | Description |
| ---------- | ----- | ----------- |
| missing_field_error | 12 | Required field is missing |
| type_error | 5 | Incorrect data type (e.g., int vs. string) |
| url_format_error | 3 | URL format is invalid |
```