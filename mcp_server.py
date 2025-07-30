"""OpenRouter Provider Validator - MCP Server

Exposes OpenRouter Provider Validator operations as tools for LLM interaction.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from mcp.server.fastmcp import Context, FastMCP

# Configure logging with rotation
level = os.getenv("LOG_LEVEL", "INFO")
log_folder = Path("logs")
log_folder.mkdir(exist_ok=True)
log_file = log_folder / f"mcp_{datetime.now().strftime('%Y%m%d')}.log"

logger = logging.getLogger("mcp_server")
logger.setLevel(getattr(logging, level))
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(file_handler)

# Define BaseModels for requests and responses
class PathRequest(BaseModel):
    """Request with a file path."""
    path: str = Field(description="File path to read/write")

class ContentRequest(BaseModel):
    """Request with content to write."""
    path: str = Field(description="File path to write to")
    content: str = Field(description="Content to write")

class AppendRequest(BaseModel):
    """Request to append content."""
    path: str = Field(description="File path to append to")
    content: str = Field(description="Content to append")
    create_if_missing: bool = Field(default=True, description="Whether to create the file if it doesn't exist")

class ListFolderRequest(BaseModel):
    """Request to list folder contents."""
    path: str = Field(description="Folder path to list")
    include_details: bool = Field(default=False, description="Whether to include file details")

class CreateFoldersRequest(BaseModel):
    """Request to create folders."""
    paths: List[str] = Field(description="List of folder paths to create")

class SearchFilesRequest(BaseModel):
    """Request to search files for a pattern."""
    pattern: str = Field(description="Text pattern to search for")
    path: str = Field(default=".", description="Directory to search in")
    recursive: bool = Field(default=True, description="Whether to search in subdirectories")

class CopyEntryRequest(BaseModel):
    """Request to copy a file or directory."""
    source_path: str = Field(description="Source path to copy from")
    destination_path: str = Field(description="Destination path to copy to")
    overwrite: bool = Field(default=False, description="Whether to overwrite existing files")

class MoveEntryRequest(BaseModel):
    """Request to move a file or directory."""
    source_path: str = Field(description="Source path to move")
    destination_dir: str = Field(description="Destination directory to move to")
    overwrite: bool = Field(default=False, description="Whether to overwrite if destination exists")

class ProviderDetailsRequest(BaseModel):
    """Request for information about a specific provider."""
    provider_id: str = Field(description="ID of the provider to retrieve")

class ModelTestResultsRequest(BaseModel):
    """Request to load test results for a model."""
    model: Optional[str] = Field(default=None, description="Model to load results for (all models if None)")

class ProviderSummaryRequest(BaseModel):
    """Request to generate a summary for a provider."""
    model: str = Field(description="Model to generate provider summary for")

class SaveReportRequest(BaseModel):
    """Request to save a report."""
    name: str = Field(description="Name for the report file")
    content: str = Field(description="Report content in markdown format")

# Responses
class FileContentResponse(BaseModel):
    """Response containing file content."""
    content: str = Field(description="File content")
    success: bool = Field(default=True, description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")

class OperationResponse(BaseModel):
    """Response for a basic operation."""
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if the operation failed")

class FolderEntryType(str, Enum):
    """Type of folder entry."""
    FILE = "file"
    DIRECTORY = "directory"

class FolderEntry(BaseModel):
    """Entry in a folder listing."""
    name: str = Field(description="Name of the file or folder")
    type: FolderEntryType = Field(description="Whether it's a file or directory")
    path: Optional[str] = Field(default=None, description="Full path of entry (for detailed listings)")
    size: Optional[int] = Field(default=None, description="Size in bytes (for files)")
    modified: Optional[float] = Field(default=None, description="Last modified timestamp (for files)")

class ListFolderResponse(BaseModel):
    """Response with folder contents."""
    entries: List[FolderEntry] = Field(description="List of folder entries")
    success: bool = Field(default=True, description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")

class FileMatch(BaseModel):
    """Match information for a file search."""
    line_number: int = Field(description="Line number of the match")
    context: str = Field(description="Text context around the match")

class SearchMatch(BaseModel):
    """Search match in a file."""
    path: str = Field(description="Path to the file with matches")
    matches: List[FileMatch] = Field(description="List of matches in the file") 

class SearchFilesResponse(BaseModel):
    """Response with search results."""
    matches: List[SearchMatch] = Field(description="List of matched files with results")
    success: bool = Field(default=True, description="Whether the search was successful")
    error: Optional[str] = Field(default=None, description="Error message if search failed")

class ErrorCategory(BaseModel):
    """Error category and count."""
    category: str = Field(description="Error category name")
    count: int = Field(description="Number of errors in this category")

class ProviderSummaryResponse(BaseModel):
    """Response with provider summary."""
    provider: str = Field(description="Provider name")
    model: str = Field(description="Model identifier")
    total_attempts: int = Field(description="Total test attempts")
    successful_attempts: int = Field(description="Successful attempts")
    failure_rate: float = Field(description="Failure rate as percentage")
    error_categories: Dict[str, int] = Field(description="Error counts by category")
    avg_response_time: Optional[float] = Field(default=None, description="Average response time in ms")
    success: bool = Field(default=True, description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")

class ModelListResponse(BaseModel):
    """Response with list of models."""
    models: List[str] = Field(description="List of models with test results")
    success: bool = Field(default=True, description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")

class StatsResponse(BaseModel):
    """Response with system statistics."""
    total_providers: int = Field(description="Total number of providers")
    enabled_providers: int = Field(description="Number of enabled providers")
    total_prompts: int = Field(description="Total number of prompts")
    models_tested: int = Field(description="Number of models tested")
    total_test_results: int = Field(description="Total number of test results")
    successful_tests: int = Field(description="Number of successful tests")
    failed_tests: int = Field(description="Number of failed tests")
    success: bool = Field(default=True, description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")

# Initialize the MCP server
@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize and cleanup resources for the MCP server."""
    # Import and initialize the client
    from client import FileSystemClient
    
    client = FileSystemClient()
    logger.info("Initialized FileSystemClient for MCP Server")
    
    try:
        yield {"client": client}
    finally:
        logger.info("Shutting down MCP server")

mcp = FastMCP(
    "OpenRouter Provider Validator",
    description="MCP Server for the OpenRouter Provider Validator",
    lifespan=lifespan
)

# Define tool functions
@mcp.tool()
def read_file(request: PathRequest, ctx: Context) -> FileContentResponse:
    """Read the contents of a file.
    
    Args:
        request: Contains the path of the file to read
        ctx: Tool context with client access
        
    Returns:
        File content or error message
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Reading file: {request.path}")
        content = client.read_file(request.path)
        return FileContentResponse(content=content, success=True)
    except Exception as e:
        logger.error(f"Error reading file {request.path}: {str(e)}")
        return FileContentResponse(content="", success=False, error=str(e))

@mcp.tool()
def write_file(request: ContentRequest, ctx: Context) -> OperationResponse:
    """Write content to a file, creating it if it doesn't exist.
    
    Args:
        request: Contains the path and content for the file
        ctx: Tool context with client access
        
    Returns:
        Success status or error message
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Writing to file: {request.path}")
        client.write_file(request.path, request.content)
        return OperationResponse(success=True)
    except Exception as e:
        logger.error(f"Error writing to file {request.path}: {str(e)}")
        return OperationResponse(success=False, error=str(e))

@mcp.tool()
def append_to_file(request: AppendRequest, ctx: Context) -> OperationResponse:
    """Append content to a file.
    
    Args:
        request: Contains the path, content to append, and creation flag
        ctx: Tool context with client access
        
    Returns:
        Success status or error message
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Appending to file: {request.path}")
        client.append_to_file(request.path, request.content, request.create_if_missing)
        return OperationResponse(success=True)
    except Exception as e:
        logger.error(f"Error appending to file {request.path}: {str(e)}")
        return OperationResponse(success=False, error=str(e))

@mcp.tool()
def list_folder_contents(request: ListFolderRequest, ctx: Context) -> ListFolderResponse:
    """List the contents of a directory.
    
    Args:
        request: Contains the path to list and detail flag
        ctx: Tool context with client access
        
    Returns:
        Directory contents or error message
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Listing folder contents: {request.path}")
        entries_raw = client.list_folder_contents(request.path, request.include_details)
        
        # Convert to FolderEntry objects
        entries = []
        for entry in entries_raw:
            folder_entry = FolderEntry(
                name=entry["name"],
                type=FolderEntryType.DIRECTORY if entry["type"] == "directory" else FolderEntryType.FILE
            )
            
            # Add additional fields if details were requested
            if request.include_details:
                if "path" in entry:
                    folder_entry.path = entry["path"]
                if "size" in entry:
                    folder_entry.size = entry["size"] 
                if "modified" in entry:
                    folder_entry.modified = entry["modified"]
            
            entries.append(folder_entry)
        
        return ListFolderResponse(entries=entries, success=True)
    except Exception as e:
        logger.error(f"Error listing folder {request.path}: {str(e)}")
        return ListFolderResponse(entries=[], success=False, error=str(e))

@mcp.tool()
def create_folders(request: CreateFoldersRequest, ctx: Context) -> OperationResponse:
    """Create one or more directories.
    
    Args:
        request: Contains paths of directories to create
        ctx: Tool context with client access
        
    Returns:
        Success status or error message
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Creating folders: {', '.join(request.paths)}")
        client.create_folders(request.paths)
        return OperationResponse(success=True)
    except Exception as e:
        logger.error(f"Error creating folders: {str(e)}")
        return OperationResponse(success=False, error=str(e))

@mcp.tool()
def search_files(request: SearchFilesRequest, ctx: Context) -> SearchFilesResponse:
    """Search files for a text pattern.
    
    Args:
        request: Contains search pattern and options
        ctx: Tool context with client access
        
    Returns:
        Search results or error message
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Searching for pattern '{request.pattern}' in {request.path}")
        matches_raw = client.search_files(request.pattern, request.path, request.recursive)
        
        # Convert to structured response
        matches = []
        for match in matches_raw:
            file_matches = []
            for m in match["matches"]:
                file_matches.append(FileMatch(
                    line_number=m["line_number"],
                    context=m["context"]
                ))
                
            matches.append(SearchMatch(
                path=match["path"],
                matches=file_matches
            ))
        
        return SearchFilesResponse(matches=matches, success=True)
    except Exception as e:
        logger.error(f"Error searching files: {str(e)}")
        return SearchFilesResponse(matches=[], success=False, error=str(e))

@mcp.tool()
def copy_entry(request: CopyEntryRequest, ctx: Context) -> OperationResponse:
    """Copy a file or directory.
    
    Args:
        request: Contains source and destination paths and overwrite flag
        ctx: Tool context with client access
        
    Returns:
        Success status or error message
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Copying {request.source_path} to {request.destination_path}")
        client.copy_entry(request.source_path, request.destination_path, request.overwrite)
        return OperationResponse(success=True)
    except Exception as e:
        logger.error(f"Error copying entry: {str(e)}")
        return OperationResponse(success=False, error=str(e))

@mcp.tool()
def move_entry(request: MoveEntryRequest, ctx: Context) -> OperationResponse:
    """Move a file or directory.
    
    Args:
        request: Contains source path, destination directory, and overwrite flag
        ctx: Tool context with client access
        
    Returns:
        Success status or error message
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Moving {request.source_path} to {request.destination_dir}")
        client.move_entry(request.source_path, request.destination_dir, request.overwrite)
        return OperationResponse(success=True)
    except Exception as e:
        logger.error(f"Error moving entry: {str(e)}")
        return OperationResponse(success=False, error=str(e))

@mcp.tool()
def load_test_results(request: ModelTestResultsRequest, ctx: Context) -> Dict[str, Any]:
    """Load test results for a model.
    
    Args:
        request: Contains the model to load results for (optional)
        ctx: Tool context with client access
        
    Returns:
        Dictionary with test results or error
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Loading test results for model: {request.model or 'all'}")
        results = client.load_test_results(request.model)
        
        # Convert to serializable format
        serialized_results = []
        for result in results:
            serialized_results.append(json.loads(result.json()))
        
        return {
            "results": serialized_results,
            "count": len(serialized_results),
            "success": True
        }
    except Exception as e:
        logger.error(f"Error loading test results: {str(e)}")
        return {
            "results": [],
            "count": 0,
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def get_provider_summary(request: ProviderSummaryRequest, ctx: Context) -> List[ProviderSummaryResponse]:
    """Generate summary statistics for providers of a model.
    
    Args:
        request: Contains the model to generate summary for
        ctx: Tool context with client access
        
    Returns:
        List of provider summaries or error
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Generating provider summary for model: {request.model}")
        summaries = client.generate_provider_summary(request.model)
        
        # Convert to response objects
        responses = []
        for summary in summaries:
            responses.append(ProviderSummaryResponse(
                provider=summary.provider,
                model=summary.model,
                total_attempts=summary.total_attempts,
                successful_attempts=summary.successful_attempts,
                failure_rate=summary.failure_rate,
                error_categories=summary.error_categories,
                avg_response_time=summary.avg_response_time,
                success=True
            ))
        
        return responses
    except Exception as e:
        logger.error(f"Error generating provider summary: {str(e)}")
        return [ProviderSummaryResponse(
            provider="error",
            model=request.model,
            total_attempts=0,
            successful_attempts=0,
            failure_rate=0.0,
            error_categories={},
            success=False,
            error=str(e)
        )]

@mcp.tool()
def list_models(ctx: Context) -> ModelListResponse:
    """List all models that have test results.
    
    Args:
        ctx: Tool context with client access
        
    Returns:
        List of model names or error
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info("Listing models with test results")
        models = client.list_models()
        return ModelListResponse(models=models, success=True)
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return ModelListResponse(models=[], success=False, error=str(e))

@mcp.tool()
def get_stats(ctx: Context) -> StatsResponse:
    """Get overall statistics about the test data.
    
    Args:
        ctx: Tool context with client access
        
    Returns:
        Statistics or error message
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info("Getting system statistics")
        stats = client.get_stats()
        
        return StatsResponse(
            total_providers=stats.get("total_providers", 0),
            enabled_providers=stats.get("enabled_providers", 0),
            total_prompts=stats.get("total_prompts", 0),
            models_tested=stats.get("models_tested", 0),
            total_test_results=stats.get("total_test_results", 0),
            successful_tests=stats.get("successful_tests", 0),
            failed_tests=stats.get("failed_tests", 0),
            success=True
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return StatsResponse(
            total_providers=0,
            enabled_providers=0,
            total_prompts=0,
            models_tested=0,
            total_test_results=0,
            successful_tests=0,
            failed_tests=0,
            success=False,
            error=str(e)
        )

@mcp.tool()
def save_report(request: SaveReportRequest, ctx: Context) -> OperationResponse:
    """Save a report to the reports directory.
    
    Args:
        request: Contains report name and content
        ctx: Tool context with client access
        
    Returns:
        Success status or error message
    """
    client = ctx.request_context.lifespan_context["client"]
    
    try:
        logger.info(f"Saving report: {request.name}")
        client.save_report(request.name, request.content)
        return OperationResponse(success=True)
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        return OperationResponse(success=False, error=str(e))

# Create a class to expose server
class MCPServer:
    """OpenRouter Provider Validator MCP Server."""
    
    def __init__(self, client):
        """Initialize the server with a filesystem client.
        
        Args:
            client: FileSystemClient instance
        """
        self.client = client
        self._mcp = mcp
        
    def get_tools(self):
        """Get the tool definitions.
        
        Returns:
            List of tool definitions for OpenRouter API
        """
        # This would normally come from introspection, but manually define for now
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to read"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file, creating it if it doesn't exist",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to write to"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "append_to_file",
                    "description": "Append content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string", 
                                "description": "File path to append to"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to append"
                            },
                            "create_if_missing": {
                                "type": "boolean",
                                "description": "Whether to create the file if it doesn't exist",
                                "default": True
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_folder_contents",
                    "description": "List the contents of a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Folder path to list"
                            },
                            "include_details": {
                                "type": "boolean",
                                "description": "Whether to include file details",
                                "default": False
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_folders",
                    "description": "Create one or more directories",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of folder paths to create"
                            }
                        },
                        "required": ["paths"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search files for a text pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Text pattern to search for"
                            },
                            "path": {
                                "type": "string",
                                "description": "Directory to search in",
                                "default": "."
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Whether to search in subdirectories",
                                "default": True
                            }
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "copy_entry",
                    "description": "Copy a file or directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_path": {
                                "type": "string",
                                "description": "Source path to copy from"
                            },
                            "destination_path": {
                                "type": "string",
                                "description": "Destination path to copy to"
                            },
                            "overwrite": {
                                "type": "boolean",
                                "description": "Whether to overwrite existing files",
                                "default": False
                            }
                        },
                        "required": ["source_path", "destination_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_entry",
                    "description": "Move a file or directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_path": {
                                "type": "string",
                                "description": "Source path to move"
                            },
                            "destination_dir": {
                                "type": "string",
                                "description": "Destination directory to move to"
                            },
                            "overwrite": {
                                "type": "boolean",
                                "description": "Whether to overwrite if destination exists",
                                "default": False
                            }
                        },
                        "required": ["source_path", "destination_dir"]
                    }
                }
            }
        ]

def main():
    mcp.run()
    
if __name__ == "__main__":
    main()
