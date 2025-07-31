"""Filesystem Test Helper

Provides test-specific filesystem operations for OpenRouter Provider Validator.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from client import FileSystemClient


class FileSystemTestHelper:
    """Helper class for filesystem operations in tests."""
    
    def __init__(self, base_path: str = ".", test_files_dir: Optional[Path] = None):
        """Initialize the test helper.
        
        Args:
            base_path: Base directory path for all operations
            test_files_dir: Optional custom directory for test files
        """
        self.base_path = Path(base_path)
        self.filesystem_client = FileSystemClient(base_path)
        
        # Use custom test files directory if provided, otherwise use default
        self.test_files_dir = test_files_dir or (self.base_path / "data/test_files")
        
        # Ensure test directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required test directories exist."""
        test_dirs = [
            self.test_files_dir,
            self.test_files_dir / "nested"
        ]
        
        for directory in test_dirs:
            directory.mkdir(exist_ok=True, parents=True)
    
    def initialize_test_files(self) -> None:
        """Initialize or reset test files for a clean test run."""
        # First, clear any existing test files
        if self.test_files_dir.exists():
            # Remove all contents but keep the directory
            for item in self.test_files_dir.iterdir():
                if item.is_dir() and item.name != "nested":  # Preserve the nested directory
                    shutil.rmtree(item)
                elif item.is_file():
                    item.unlink()
        
        # Create nested directory if it doesn't exist
        nested_dir = self.test_files_dir / "nested"
        nested_dir.mkdir(exist_ok=True, parents=True)
        
        # Clear contents of nested directory
        for item in nested_dir.iterdir():
            if item.is_file():
                item.unlink()
        
        # Check if we have a templates directory to copy from
        templates_dir = self.base_path / "data/test_files/templates"
        if templates_dir.exists() and templates_dir.is_dir():
            # Copy templates instead of creating hardcoded files
            for template_file in templates_dir.glob("**/*"):
                if template_file.is_file():
                    # Construct the relative path from templates_dir
                    rel_path = template_file.relative_to(templates_dir)
                    # Construct destination path in test_files_dir
                    dest_file = self.test_files_dir / rel_path
                    # Create parent directories if needed
                    dest_file.parent.mkdir(exist_ok=True, parents=True)
                    # Copy the template file
                    shutil.copy2(template_file, dest_file)
        else:
            # Create standard sample files if no templates exist
            with open(self.test_files_dir / "sample1.txt", "w") as f:
                f.write("This is sample file 1\nIt has multiple lines\nFor testing file reading operations.")
            
            with open(self.test_files_dir / "sample2.txt", "w") as f:
                f.write("Sample file 2 contains different content\nUseful for testing searching functionality.")
            
            with open(nested_dir / "sample3.txt", "w") as f:
                f.write("This is a nested file\nLocated in a subdirectory\nFor testing nested path operations.")
    
    def load_prompts(self, filepath: str = "data/prompts.json") -> List[Dict[str, Any]]:
        """Load all test prompts.
        
        Args:
            filepath: Path to prompts file
            
        Returns:
            List of test prompts
        """
        return self.filesystem_client.load_prompts(filepath)
    
    def load_prompt_sequence(self, prompt_id: str, filepath: str = "data/prompts.json") -> Optional[Dict[str, Any]]:
        """Load a specific prompt sequence by ID.
        
        Args:
            prompt_id: ID of the prompt sequence to load
            filepath: Path to prompts file
            
        Returns:
            Prompt sequence dictionary or None if not found
        """
        prompts = self.load_prompts(filepath)
        
        for prompt in prompts:
            if prompt["id"] == prompt_id:
                return prompt
                
        return None
    
    # File operation methods expected by the agent
    
    def list_files(self, directory: str) -> str:
        """List files in a directory.
        
        Args:
            directory: Directory to list files from
            
        Returns:
            Formatted list of files as a string
        """
        try:
            # If directory is a relative path within the test files directory structure,
            # resolve it against the test_files_dir
            if not os.path.isabs(directory) and not directory.startswith("data/") and not directory.startswith("./data/"):
                # Check if it might be a path relative to test_files_dir
                test_relative_path = self.test_files_dir / directory
                if test_relative_path.exists():
                    directory = str(test_relative_path)
            
            entries = self.filesystem_client.list_folder_contents(directory)
            result = []
            result.append(f"Contents of {directory}:")
            
            for entry in entries:
                entry_type = "📁" if entry["type"] == "directory" else "📄"
                result.append(f"{entry_type} {entry['name']}")
            
            return "\n".join(result)
        except Exception as e:
            return f"Error listing files in {directory}: {str(e)}"
    
    def read_file(self, file_path: str) -> str:
        """Read content from a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File content as a string
        """
        try:
            # Check if file_path is relative to test_files_dir
            if not os.path.isabs(file_path) and not file_path.startswith("data/") and not file_path.startswith("./data/"):
                test_relative_path = self.test_files_dir / file_path
                if test_relative_path.exists():
                    file_path = str(test_relative_path)
                    
            return self.filesystem_client.read_file(file_path)
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write content to a file.
        
        Args:
            file_path: Path to the file to write to
            content: Content to write to the file
            
        Returns:
            True if successful
        """
        try:
            # Resolve path against test_files_dir if it's a relative path
            if not os.path.isabs(file_path) and not file_path.startswith("data/") and not file_path.startswith("./data/"):
                file_path = str(self.test_files_dir / file_path)
                
            return self.filesystem_client.write_file(file_path, content)
        except Exception as e:
            raise ValueError(f"Error writing to file {file_path}: {str(e)}")
    
    def append_file(self, file_path: str, content: str) -> bool:
        """Append content to a file.
        
        Args:
            file_path: Path to the file to append to
            content: Content to append to the file
            
        Returns:
            True if successful
        """
        try:
            # Resolve path against test_files_dir if it's a relative path
            if not os.path.isabs(file_path) and not file_path.startswith("data/") and not file_path.startswith("./data/"):
                file_path = str(self.test_files_dir / file_path)
                
            return self.filesystem_client.append_to_file(file_path, content, create_if_missing=True)
        except Exception as e:
            raise ValueError(f"Error appending to file {file_path}: {str(e)}")
    
    def create_directory(self, directory: str) -> bool:
        """Create a new directory.
        
        Args:
            directory: Path of the directory to create
            
        Returns:
            True if successful
        """
        try:
            # Resolve path against test_files_dir if it's a relative path
            if not os.path.isabs(directory) and not directory.startswith("data/") and not directory.startswith("./data/"):
                directory = str(self.test_files_dir / directory)
                
            return self.filesystem_client.create_folders([directory])
        except Exception as e:
            raise ValueError(f"Error creating directory {directory}: {str(e)}")
    
    def copy_file(self, source: str, destination: str) -> bool:
        """Copy a file to a new location.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful
        """
        try:
            # Resolve paths against test_files_dir if they are relative paths
            if not os.path.isabs(source) and not source.startswith("data/") and not source.startswith("./data/"):
                source = str(self.test_files_dir / source)
            
            if not os.path.isabs(destination) and not destination.startswith("data/") and not destination.startswith("./data/"):
                destination = str(self.test_files_dir / destination)
                
            return self.filesystem_client.copy_entry(source, destination)
        except Exception as e:
            raise ValueError(f"Error copying file from {source} to {destination}: {str(e)}")
    
    def move_file(self, source: str, destination: str) -> bool:
        """Move a file to a new location.
        
        Args:
            source: Source file path
            destination: Destination directory
            
        Returns:
            True if successful
        """
        try:
            # Resolve paths against test_files_dir if they are relative paths
            if not os.path.isabs(source) and not source.startswith("data/") and not source.startswith("./data/"):
                source = str(self.test_files_dir / source)
            
            if not os.path.isabs(destination) and not destination.startswith("data/") and not destination.startswith("./data/"):
                destination = str(self.test_files_dir / destination)
            
            # The move_entry method expects a destination directory, not file
            dest_dir = os.path.dirname(destination)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
            
            # Move the file to the destination directory with the new name
            return self.filesystem_client.move_entry(source, dest_dir)
        except Exception as e:
            raise ValueError(f"Error moving file from {source} to {destination}: {str(e)}")
    
    def search_files(self, directory: str, pattern: str) -> str:
        """Search for content in files.
        
        Args:
            directory: Directory to search in
            pattern: Pattern to search for
            
        Returns:
            Formatted string with search results
        """
        try:
            # Resolve path against test_files_dir if it's a relative path
            if not os.path.isabs(directory) and not directory.startswith("data/") and not directory.startswith("./data/"):
                directory = str(self.test_files_dir / directory)
                
            results = self.filesystem_client.search_files(pattern, directory)
            
            if not results:
                return f"No matches found for '{pattern}' in {directory}"
            
            output = [f"Search results for '{pattern}' in {directory}:"]
            
            for result in results:
                output.append(f"\nFile: {result['path']}")
                for match in result['matches']:
                    output.append(f"Line {match['line_number']}:\n{match['context']}\n")
            
            return "\n".join(output)
        except Exception as e:
            return f"Error searching for '{pattern}' in {directory}: {str(e)}"
