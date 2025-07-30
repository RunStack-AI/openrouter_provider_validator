"""OpenRouter Provider Validator MCP Server

Wrap the filesystem client to expose its functionality as MCP tools.
"""

import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field, field_validator

from client import FileSystemClient, ProviderConfig, TestPrompt, TestResult, ProviderSummary

# Set up logging
logger = logging.getLogger("openrouter_validator")
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up a rotating file handler
file_handler = logging.handlers.RotatingFileHandler(
    "logs/validator.log", maxBytes=10*1024*1024, backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# Console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(console_handler)

# Initialize the FastMCP server
mcp = FastMCP("OpenRouter Provider Validator")

# Define request models for tools
class ListProvidersRequest(BaseModel):
    """Request model for listing providers."""
    pass

class GetProviderRequest(BaseModel):
    """Request model for getting a specific provider."""
    name: str = Field(description="Provider name to retrieve")

class SaveProviderRequest(BaseModel):
    """Request model for saving a provider."""
    provider: ProviderConfig = Field(description="Provider configuration to save")

class DeleteProviderRequest(BaseModel):
    """Request model for deleting a provider."""
    name: str = Field(description="Provider name to delete")

class ListPromptsRequest(BaseModel):
    """Request model for listing test prompts."""
    pass

class GetPromptRequest(BaseModel):
    """Request model for getting a specific test prompt."""
    id: str = Field(description="Prompt ID to retrieve")

class SavePromptRequest(BaseModel):
    """Request model for saving a test prompt."""
    prompt: TestPrompt = Field(description="Test prompt to save")

class DeletePromptRequest(BaseModel):
    """Request model for deleting a test prompt."""
    id: str = Field(description="Prompt ID to delete")

class SaveTestResultRequest(BaseModel):
    """Request model for saving a test result."""
    result: TestResult = Field(description="Test result to save")

class LoadTestResultsRequest(BaseModel):
    """Request model for loading test results."""
    model: Optional[str] = Field(default=None, description="Optional model name to filter by")

class GenerateProviderSummaryRequest(BaseModel):
    """Request model for generating provider summary."""
    model: str = Field(description="Model name to generate summary for")

class SaveReportRequest(BaseModel):
    """Request model for saving a report."""
    report_name: str = Field(description="Name of the report file")
    content: str = Field(description="Report content")

class ListModelsRequest(BaseModel):
    """Request model for listing models."""
    pass

class ClearResultsRequest(BaseModel):
    """Request model for clearing test results."""
    model: Optional[str] = Field(default=None, description="Optional model name to clear results for")

class GetStatsRequest(BaseModel):
    """Request model for getting validator statistics."""
    pass

# Define response models for tools
class ListProvidersResponse(BaseModel):
    """Response model for listing providers."""
    providers: List[ProviderConfig] = Field(description="List of provider configurations")

class ProvidersResponse(BaseModel):
    """Response model for provider operations."""
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Operation result message")
    provider: Optional[ProviderConfig] = Field(default=None, description="Provider configuration if available")

class ListPromptsResponse(BaseModel):
    """Response model for listing prompts."""
    prompts: List[TestPrompt] = Field(description="List of test prompts")

class PromptResponse(BaseModel):
    """Response model for prompt operations."""
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Operation result message")
    prompt: Optional[TestPrompt] = Field(default=None, description="Test prompt if available")

class TestResultResponse(BaseModel):
    """Response model for test result operations."""
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Operation result message")

class LoadTestResultsResponse(BaseModel):
    """Response model for loading test results."""
    results: List[TestResult] = Field(description="List of test results")

class ProviderSummaryResponse(BaseModel):
    """Response model for provider summary."""
    summaries: List[ProviderSummary] = Field(description="List of provider summaries")

class ListModelsResponse(BaseModel):
    """Response model for listing models."""
    models: List[str] = Field(description="List of model names")

class StatisticsResponse(BaseModel):
    """Response model for validator statistics."""
    stats: Dict[str, Any] = Field(description="Validator statistics")

# Define the app context manager to initialize the FileSystemClient
@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Initialize the FileSystemClient and provide it in the app context."""
    client = FileSystemClient()
    logger.info("FileSystemClient initialized")
    try:
        yield {"client": client}
    finally:
        logger.info("FileSystemClient shutdown")

# Set up the lifespan context
mcp = FastMCP("OpenRouter Provider Validator", lifespan=app_lifespan)

# Define tool functions
@mcp.tool()
def list_providers(request: ListProvidersRequest, ctx: Context) -> ListProvidersResponse:
    """List all configured OpenRouter providers.
    
    Returns a list of all provider configurations stored in the system.
    """
    logger.info("Listing providers")
    client = ctx.request_context.lifespan_context["client"]
    providers = client.load_providers()
    return ListProvidersResponse(providers=providers)

@mcp.tool()
def get_provider(request: GetProviderRequest, ctx: Context) -> ProvidersResponse:
    """Get a specific provider configuration by name.
    
    Args:
        name: The provider name to retrieve
        
    Returns information about the requested provider if it exists.
    """
    logger.info(f"Getting provider: {request.name}")
    client = ctx.request_context.lifespan_context["client"]
    providers = client.load_providers()
    
    for provider in providers:
        if provider.name == request.name:
            return ProvidersResponse(
                success=True,
                message=f"Provider {request.name} found",
                provider=provider
            )
    
    return ProvidersResponse(
        success=False,
        message=f"Provider {request.name} not found"
    )

@mcp.tool()
def save_provider(request: SaveProviderRequest, ctx: Context) -> ProvidersResponse:
    """Save or update a provider configuration.
    
    If a provider with the same name exists, it will be updated.
    Otherwise, a new provider will be added.
    """
    logger.info(f"Saving provider: {request.provider.name}")
    client = ctx.request_context.lifespan_context["client"]
    providers = client.load_providers()
    
    # Check if provider already exists
    found = False
    for i, provider in enumerate(providers):
        if provider.name == request.provider.name:
            providers[i] = request.provider
            found = True
            message = f"Updated provider {request.provider.name}"
            break
    
    if not found:
        providers.append(request.provider)
        message = f"Added new provider {request.provider.name}"
    
    client.save_providers(providers)
    
    return ProvidersResponse(
        success=True,
        message=message,
        provider=request.provider
    )

@mcp.tool()
def delete_provider(request: DeleteProviderRequest, ctx: Context) -> ProvidersResponse:
    """Delete a provider configuration by name.
    
    Args:
        name: The provider name to delete
    """
    logger.info(f"Deleting provider: {request.name}")
    client = ctx.request_context.lifespan_context["client"]
    providers = client.load_providers()
    
    original_count = len(providers)
    providers = [p for p in providers if p.name != request.name]
    
    if len(providers) < original_count:
        client.save_providers(providers)
        return ProvidersResponse(
            success=True,
            message=f"Provider {request.name} deleted"
        )
    else:
        return ProvidersResponse(
            success=False,
            message=f"Provider {request.name} not found"
        )

@mcp.tool()
def list_prompts(request: ListPromptsRequest, ctx: Context) -> ListPromptsResponse:
    """List all configured test prompts.
    
    Returns a list of all test prompts stored in the system.
    """
    logger.info("Listing prompts")
    client = ctx.request_context.lifespan_context["client"]
    prompts = client.load_prompts()
    return ListPromptsResponse(prompts=prompts)

@mcp.tool()
def get_prompt(request: GetPromptRequest, ctx: Context) -> PromptResponse:
    """Get a specific test prompt by ID.
    
    Args:
        id: The prompt ID to retrieve
        
    Returns information about the requested prompt if it exists.
    """
    logger.info(f"Getting prompt: {request.id}")
    client = ctx.request_context.lifespan_context["client"]
    prompts = client.load_prompts()
    
    for prompt in prompts:
        if prompt.id == request.id:
            return PromptResponse(
                success=True,
                message=f"Prompt {request.id} found",
                prompt=prompt
            )
    
    return PromptResponse(
        success=False,
        message=f"Prompt {request.id} not found"
    )

@mcp.tool()
def save_prompt(request: SavePromptRequest, ctx: Context) -> PromptResponse:
    """Save or update a test prompt.
    
    If a prompt with the same ID exists, it will be updated.
    Otherwise, a new prompt will be added.
    """
    logger.info(f"Saving prompt: {request.prompt.id}")
    client = ctx.request_context.lifespan_context["client"]
    prompts = client.load_prompts()
    
    # Check if prompt already exists
    found = False
    for i, prompt in enumerate(prompts):
        if prompt.id == request.prompt.id:
            prompts[i] = request.prompt
            found = True
            message = f"Updated prompt {request.prompt.id}"
            break
    
    if not found:
        prompts.append(request.prompt)
        message = f"Added new prompt {request.prompt.id}"
    
    client.save_prompts(prompts)
    
    return PromptResponse(
        success=True,
        message=message,
        prompt=request.prompt
    )

@mcp.tool()
def delete_prompt(request: DeletePromptRequest, ctx: Context) -> PromptResponse:
    """Delete a test prompt by ID.
    
    Args:
        id: The prompt ID to delete
    """
    logger.info(f"Deleting prompt: {request.id}")
    client = ctx.request_context.lifespan_context["client"]
    prompts = client.load_prompts()
    
    original_count = len(prompts)
    prompts = [p for p in prompts if p.id != request.id]
    
    if len(prompts) < original_count:
        client.save_prompts(prompts)
        return PromptResponse(
            success=True,
            message=f"Prompt {request.id} deleted"
        )
    else:
        return PromptResponse(
            success=False,
            message=f"Prompt {request.id} not found"
        )

@mcp.tool()
def save_test_result(request: SaveTestResultRequest, ctx: Context) -> TestResultResponse:
    """Save a test result to the filesystem.
    
    Args:
        result: The test result to save
    """
    logger.info(f"Saving test result: {request.result.provider}/{request.result.model}/{request.result.prompt_id}")
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        client.save_test_result(request.result)
        return TestResultResponse(
            success=True,
            message="Test result saved successfully"
        )
    except Exception as e:
        logger.error(f"Error saving test result: {str(e)}")
        return TestResultResponse(
            success=False,
            message=f"Error saving test result: {str(e)}"
        )

@mcp.tool()
def load_test_results(request: LoadTestResultsRequest, ctx: Context) -> LoadTestResultsResponse:
    """Load test results, optionally filtered by model.
    
    Args:
        model: Optional model name to filter results
    """
    logger.info(f"Loading test results" + (f" for model {request.model}" if request.model else ""))
    client = ctx.request_context.lifespan_context["client"]
    results = client.load_test_results(request.model)
    return LoadTestResultsResponse(results=results)

@mcp.tool()
def generate_provider_summary(request: GenerateProviderSummaryRequest, ctx: Context) -> ProviderSummaryResponse:
    """Generate summary statistics for all providers of a model.
    
    Args:
        model: Model name to generate summary for
    """
    logger.info(f"Generating provider summary for model {request.model}")
    client = ctx.request_context.lifespan_context["client"]
    summaries = client.generate_provider_summary(request.model)
    return ProviderSummaryResponse(summaries=summaries)

@mcp.tool()
def save_report(request: SaveReportRequest, ctx: Context) -> TestResultResponse:
    """Save a report to the reports directory.
    
    Args:
        report_name: Name of the report file
        content: Report content
    """
    logger.info(f"Saving report: {request.report_name}")
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        client.save_report(request.report_name, request.content)
        return TestResultResponse(
            success=True,
            message=f"Report {request.report_name} saved successfully"
        )
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        return TestResultResponse(
            success=False,
            message=f"Error saving report: {str(e)}"
        )

@mcp.tool()
def list_models(request: ListModelsRequest, ctx: Context) -> ListModelsResponse:
    """List all models that have test results.
    
    Returns a list of model names that have test results.
    """
    logger.info("Listing models")
    client = ctx.request_context.lifespan_context["client"]
    models = client.list_models()
    return ListModelsResponse(models=models)

@mcp.tool()
def clear_results(request: ClearResultsRequest, ctx: Context) -> TestResultResponse:
    """Clear test results, optionally for a specific model.
    
    Args:
        model: Optional model name to clear results for
    """
    logger.info(f"Clearing test results" + (f" for model {request.model}" if request.model else ""))
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        client.clear_results(request.model)
        message = f"Test results cleared" + (f" for model {request.model}" if request.model else "")
        return TestResultResponse(
            success=True,
            message=message
        )
    except Exception as e:
        logger.error(f"Error clearing results: {str(e)}")
        return TestResultResponse(
            success=False,
            message=f"Error clearing results: {str(e)}"
        )

@mcp.tool()
def get_stats(request: GetStatsRequest, ctx: Context) -> StatisticsResponse:
    """Get overall statistics about the validator data.
    
    Returns statistics about providers, prompts, models, and test results.
    """
    logger.info("Getting validator statistics")
    client = ctx.request_context.lifespan_context["client"]
    stats = client.get_stats()
    return StatisticsResponse(stats=stats)

def main():
    mcp.run()
    
if __name__ == "__main__":
    main()
