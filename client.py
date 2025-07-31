"""OpenRouter Provider Validator Client

Filesystem client for managing test data, configurations, and results.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
from pydantic import BaseModel, Field


class TestResult(BaseModel):
    """Individual test result."""
    provider: str = Field(description="Provider name")
    model: str = Field(description="Model used")
    prompt_id: str = Field(description="Prompt identifier")
    success: bool = Field(description="Whether test succeeded")
    response_data: Optional[Dict[str, Any]] = Field(default=None, description="Full response data")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    error_category: Optional[str] = Field(default=None, description="Classified error category")
    timestamp: str = Field(description="Test execution timestamp")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics")


class ProviderSummary(BaseModel):
    """Summary statistics for a provider."""
    provider: str = Field(description="Provider name")
    model: str = Field(description="Model identifier")
    total_attempts: int = Field(description="Total test attempts")
    successful_attempts: int = Field(description="Successful attempts")
    failure_rate: float = Field(description="Failure rate as percentage")
    error_categories: Dict[str, int] = Field(description="Error counts by category")
    avg_response_time: Optional[float] = Field(default=None, description="Average response time in ms")


class FileSystemClient:
    """Client for managing OpenRouter validator filesystem operations."""
    
    def __init__(self, base_path: str = "."):
        """Initialize filesystem client.
        
        Args:
            base_path: Base directory path for all operations
        """
        self.base_path = Path(base_path)
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            self.base_path / "data",
            self.base_path / "results",
            self.base_path / "reports",
            self.base_path / "logs",
            self.base_path / "data/test_files",
            self.base_path / "data/test_files/nested"
        ]
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)
    
    # File system operations for MCP Server
    
    def read_file(self, path: str) -> str:
        """Read file content.
        
        Args:
            path: Path to file
            
        Returns:
            File content
        """
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
    
    def write_file(self, path: str, content: str) -> bool:
        """Write content to a file.
        
        Args:
            path: Path to file
            content: Content to write
            
        Returns:
            True if successful
        """
        try:
            # Create parent directories if they don't exist
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            
            with open(path, "w") as f:
                f.write(content)
            return True
        except Exception as e:
            raise ValueError(f"Error writing to file: {str(e)}")
    
    def append_to_file(self, path: str, content: str, create_if_missing: bool = True) -> bool:
        """Append content to a file.
        
        Args:
            path: Path to file
            content: Content to append
            create_if_missing: Whether to create the file if it doesn't exist
            
        Returns:
            True if successful
        """
        try:
            # Create parent directories if they don't exist
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            
            if not os.path.exists(path) and not create_if_missing:
                raise ValueError(f"File {path} does not exist and create_if_missing is False")
            
            with open(path, "a") as f:
                f.write(content)
            return True
        except Exception as e:
            raise ValueError(f"Error appending to file: {str(e)}")
    
    def file_exists(self, path: str) -> bool:
        """Check if a file exists.
        
        Args:
            path: Path to file
            
        Returns:
            True if the file exists
        """
        return os.path.isfile(path)
    
    def list_folder_contents(self, path: str, include_details: bool = False) -> List[Dict[str, Any]]:
        """List contents of a directory.
        
        Args:
            path: Directory path
            include_details: Whether to include file details
            
        Returns:
            List of directory contents
        """
        try:
            entries = []
            for entry in os.listdir(path):
                entry_path = os.path.join(path, entry)
                if not include_details:
                    entries.append({
                        "name": entry,
                        "type": "directory" if os.path.isdir(entry_path) else "file"
                    })
                else:
                    if os.path.isdir(entry_path):
                        entries.append({
                            "name": entry,
                            "type": "directory",
                            "path": entry_path,
                        })
                    else:
                        entries.append({
                            "name": entry,
                            "type": "file",
                            "path": entry_path,
                            "size": os.path.getsize(entry_path),
                            "modified": os.path.getmtime(entry_path)
                        })
            
            return entries
        except Exception as e:
            raise ValueError(f"Error listing directory contents: {str(e)}")
    
    def create_folders(self, paths: List[str]) -> bool:
        """Create one or more directories.
        
        Args:
            paths: List of directory paths to create
            
        Returns:
            True if successful
        """
        try:
            for path in paths:
                os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            raise ValueError(f"Error creating directories: {str(e)}")
    
    def search_files(self, pattern: str, path: str = ".", recursive: bool = True) -> List[Dict[str, Any]]:
        """Search for files matching a pattern.
        
        Args:
            pattern: Text pattern to search for
            path: Directory to search in
            recursive: Whether to search in subdirectories
            
        Returns:
            List of matches
        """
        try:
            matches = []
            
            for root, dirs, files in os.walk(path):
                if not recursive and root != path:
                    continue
                    
                for filename in files:
                    file_path = os.path.join(root, filename)
                    if not os.path.isfile(file_path):
                        continue
                        
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()
                            
                        if pattern.lower() in content.lower():
                            # Extract relevant context around match
                            lines = content.split("\n")
                            line_matches = []
                            
                            for i, line in enumerate(lines):
                                if pattern.lower() in line.lower():
                                    start = max(0, i - 2)
                                    end = min(len(lines), i + 3)
                                    context = "\n".join(lines[start:end])
                                    line_matches.append({
                                        "line_number": i + 1,
                                        "context": context
                                    })
                            
                            if line_matches:
                                matches.append({
                                    "path": file_path,
                                    "matches": line_matches
                                })
                    except Exception:
                        # Skip files we can't read
                        pass
            
            return matches
        except Exception as e:
            raise ValueError(f"Error searching files: {str(e)}")
    
    def copy_entry(self, source_path: str, destination_path: str, overwrite: bool = False) -> bool:
        """Copy a file or directory.
        
        Args:
            source_path: Source path
            destination_path: Destination path
            overwrite: Whether to overwrite existing files
            
        Returns:
            True if successful
        """
        try:
            # Check if destination exists and we're not overwriting
            if os.path.exists(destination_path) and not overwrite:
                raise ValueError(f"Destination {destination_path} already exists and overwrite is False")
            
            if os.path.isdir(source_path):
                if os.path.exists(destination_path):
                    if overwrite:
                        shutil.rmtree(destination_path)
                    else:
                        raise ValueError(f"Directory {destination_path} already exists")
                        
                shutil.copytree(source_path, destination_path)
            else:
                # Create parent directories if needed
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.copy2(source_path, destination_path)
                
            return True
        except Exception as e:
            raise ValueError(f"Error copying {source_path} to {destination_path}: {str(e)}")
    
    def move_entry(self, source_path: str, destination_dir: str, overwrite: bool = False) -> bool:
        """Move a file or directory.
        
        Args:
            source_path: Source path
            destination_dir: Destination directory
            overwrite: Whether to overwrite existing files
            
        Returns:
            True if successful
        """
        try:
            # Make sure destination directory exists
            os.makedirs(destination_dir, exist_ok=True)
            
            # Get base name of source
            base_name = os.path.basename(source_path)
            destination_path = os.path.join(destination_dir, base_name)
            
            # Check if destination exists and we're not overwriting
            if os.path.exists(destination_path) and not overwrite:
                raise ValueError(f"Destination {destination_path} already exists and overwrite is False")
            
            # Move the file or directory
            shutil.move(source_path, destination_path)
            return True
        except Exception as e:
            raise ValueError(f"Error moving {source_path} to {destination_dir}: {str(e)}")
    
    # Test data operations
    
    def load_prompts(self, filepath: str = "data/prompts.json") -> List[Dict[str, Any]]:
        """Load test prompts from prompts.json.
        
        Args:
            filepath: Path to prompts file
            
        Returns:
            List of test prompts
        """
        try:
            prompt_path = Path(filepath)
            if not prompt_path.exists():
                return []
            
            with open(prompt_path, "r") as f:
                prompts = json.load(f)
                
            return prompts
        except Exception as e:
            print(f"Error loading prompts: {str(e)}")
            return []
    
    def save_test_results(self, model: str, results: List[TestResult]) -> None:
        """Save a batch of test results for a model.
        
        Args:
            model: Model identifier
            results: List of test results
        """
        # Create model-specific results directory
        model_dir = self.base_path / "results" / model.replace("/", "_")
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Save each result in the batch
        for result in results:
            # Extract timestamp from ISO format
            if isinstance(result.timestamp, str):
                try:
                    dt = datetime.fromisoformat(result.timestamp.replace('Z', '+00:00'))
                    timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
                except ValueError:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate filename with timestamp
            filename = f"{result.provider}_{result.prompt_id}_{timestamp_str}.json"
            result_file = model_dir / filename
            
            # Ensure the directory exists (redundant but safe)
            result_file.parent.mkdir(exist_ok=True, parents=True)
            
            # Save as JSON with preprocessing for safe serialization
            try:
                with open(result_file, "w") as f:
                    # Convert to dict, handling datetime objects
                    result_dict = json.loads(result.json())
                    json.dump(result_dict, f, indent=2)
                print(f"Saved test result to {result_file}")
            except Exception as e:
                print(f"Error saving test result: {str(e)}")
    
    def load_test_results(self, model: Optional[str] = None) -> List[TestResult]:
        """Load test results, optionally filtered by model.
        
        Args:
            model: Optional model identifier to filter by
            
        Returns:
            List of test results
        """
        results = []
        results_dir = self.base_path / "results"
        
        if not results_dir.exists():
            return []
            
        if model:
            # Load results for specific model
            model_dir = results_dir / model.replace("/", "_")
            if model_dir.exists() and model_dir.is_dir():
                for result_file in model_dir.glob("**/*.json"):  # Use recursive glob to find in subdirectories
                    try:
                        with open(result_file, "r") as f:
                            result_data = json.load(f)
                        results.append(TestResult(**result_data))
                    except Exception as e:
                        print(f"Error loading result file {result_file}: {str(e)}")
        else:
            # Load results for all models
            for model_dir in results_dir.iterdir():
                if model_dir.is_dir():
                    for result_file in model_dir.glob("**/*.json"):  # Use recursive glob
                        try:
                            with open(result_file, "r") as f:
                                result_data = json.load(f)
                            results.append(TestResult(**result_data))
                        except Exception as e:
                            print(f"Error loading result file {result_file}: {str(e)}")
        
        return results
    
    def generate_provider_summary(self, model: str) -> List[ProviderSummary]:
        """Generate summary statistics for all providers of a model.
        
        Args:
            model: Model identifier to generate summary for
            
        Returns:
            List of provider summaries
        """
        results = self.load_test_results(model)
        provider_stats = {}
        
        for result in results:
            provider = result.provider
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "total_attempts": 0,
                    "successful_attempts": 0,
                    "error_categories": {},
                    "response_times": []
                }
            
            provider_stats[provider]["total_attempts"] += 1
            
            if result.success:
                provider_stats[provider]["successful_attempts"] += 1
                
                # Track response times from successful requests
                if result.metrics and "latency_ms" in result.metrics:
                    provider_stats[provider]["response_times"].append(result.metrics["latency_ms"])
            else:
                # Track error categories
                category = result.error_category or "unknown_error"
                provider_stats[provider]["error_categories"][category] = \
                    provider_stats[provider]["error_categories"].get(category, 0) + 1
        
        # Convert to ProviderSummary objects
        summaries = []
        for provider, stats in provider_stats.items():
            failure_rate = 0
            if stats["total_attempts"] > 0:
                failure_rate = ((stats["total_attempts"] - stats["successful_attempts"]) / 
                               stats["total_attempts"] * 100)
            
            # Calculate average response time if available
            avg_response_time = None
            if stats["response_times"]:
                avg_response_time = sum(stats["response_times"]) / len(stats["response_times"])
            
            summary = ProviderSummary(
                provider=provider,
                model=model,
                total_attempts=stats["total_attempts"],
                successful_attempts=stats["successful_attempts"],
                failure_rate=failure_rate,
                error_categories=stats["error_categories"],
                avg_response_time=avg_response_time
            )
            summaries.append(summary)
        
        return summaries
    
    def save_report(self, report_name: str, content: str) -> None:
        """Save a report to the reports directory.
        
        Args:
            report_name: Name of the report file
            content: Report content
        """
        reports_dir = self.base_path / "reports"
        reports_dir.mkdir(exist_ok=True, parents=True)
        
        # Handle report names that might include subdirectories
        if '/' in report_name or '\\' in report_name:
            report_path = reports_dir / report_name
            report_path = report_path.with_suffix('.md')  # Add .md extension if not present
            report_path.parent.mkdir(exist_ok=True, parents=True)
        else:
            report_path = reports_dir / f"{report_name}.md"
        
        try:
            with open(report_path, "w") as f:
                f.write(content)
            print(f"Saved report to {report_path}")
        except Exception as e:
            print(f"Error saving report: {str(e)}")
    
    def list_models(self) -> List[str]:
        """List all models that have test results.
        
        Returns:
            List of unique model names
        """
        results_dir = self.base_path / "results"
        if not results_dir.exists():
            return []
        
        models = []
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir():
                # Convert directory name back to model name
                model_name = model_dir.name.replace("_", "/")
                models.append(model_name)
        
        return models
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics about the test data.
        
        Returns:
            Dictionary of statistics
        """
        # Load provider information from provider_config instead of directly
        from provider_config import ProviderConfig
        providers = ProviderConfig.load_providers()
        
        # Load prompts and results directly
        prompts = self.load_prompts()
        all_results = self.load_test_results()
        models = self.list_models()
        
        enabled_providers = sum(1 for p in providers if p.get("enabled", True))
        
        successful_tests = len([r for r in all_results if r.success])
        failed_tests = len([r for r in all_results if not r.success])
        
        return {
            "total_providers": len(providers),
            "enabled_providers": enabled_providers,
            "total_prompts": len(prompts),
            "models_tested": len(models),
            "total_test_results": len(all_results),
            "successful_tests": successful_tests,
            "failed_tests": failed_tests
        }
