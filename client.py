"""OpenRouter Provider Validator Client

Filesystem client for managing test data, configurations, and results.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for an OpenRouter provider."""
    name: str = Field(description="Provider name (e.g., 'anthropic', 'openai')")
    model: str = Field(description="Model identifier")
    description: Optional[str] = Field(default=None, description="Provider description")
    enabled: bool = Field(default=True, description="Whether provider is enabled for testing")


class TestPrompt(BaseModel):
    """Test prompt configuration."""
    id: str = Field(description="Unique prompt identifier")
    name: str = Field(description="Human-readable prompt name")
    prompt: str = Field(description="The actual prompt text")
    category: str = Field(description="Prompt category (e.g., 'tool_use', 'reasoning')")
    expected_tools: Optional[List[str]] = Field(default=None, description="Expected tools to be used")


class TestResult(BaseModel):
    """Individual test result."""
    provider: str = Field(description="Provider name")
    model: str = Field(description="Model used")
    prompt_id: str = Field(description="Prompt identifier")
    success: bool = Field(description="Whether test succeeded")
    response_id: Optional[str] = Field(default=None, description="OpenAI response ID")
    timestamp: datetime = Field(description="Test execution timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    error_category: Optional[str] = Field(default=None, description="Classified error category")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")
    response_data: Optional[Dict[str, Any]] = Field(default=None, description="Full response data")


class ProviderSummary(BaseModel):
    """Summary statistics for a provider."""
    provider: str = Field(description="Provider name")
    model: str = Field(description="Model identifier")
    total_attempts: int = Field(description="Total test attempts")
    successful_attempts: int = Field(description="Successful attempts")
    failure_rate: float = Field(description="Failure rate as percentage")
    error_categories: Dict[str, int] = Field(description="Error counts by category")
    avg_response_time: Optional[float] = Field(default=None, description="Average response time")


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
            self.base_path / "reports"
        ]
        for directory in directories:
            directory.mkdir(exist_ok=True)
    
    def load_providers(self) -> List[ProviderConfig]:
        """Load provider configurations from providers.json.
        
        Returns:
            List of provider configurations
        """
        providers_file = self.base_path / "data" / "providers.json"
        if not providers_file.exists():
            return []
        
        with open(providers_file, 'r') as f:
            data = json.load(f)
        
        return [ProviderConfig(**provider) for provider in data]
    
    def save_providers(self, providers: List[ProviderConfig]) -> None:
        """Save provider configurations to providers.json.
        
        Args:
            providers: List of provider configurations to save
        """
        providers_file = self.base_path / "data" / "providers.json"
        data = [provider.model_dump() for provider in providers]
        
        with open(providers_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_prompts(self) -> List[TestPrompt]:
        """Load test prompts from prompts.json.
        
        Returns:
            List of test prompts
        """
        prompts_file = self.base_path / "data" / "prompts.json"
        if not prompts_file.exists():
            return []
        
        with open(prompts_file, 'r') as f:
            data = json.load(f)
        
        return [TestPrompt(**prompt) for prompt in data]
    
    def save_prompts(self, prompts: List[TestPrompt]) -> None:
        """Save test prompts to prompts.json.
        
        Args:
            prompts: List of test prompts to save
        """
        prompts_file = self.base_path / "data" / "prompts.json"
        data = [prompt.model_dump() for prompt in prompts]
        
        with open(prompts_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_test_result(self, result: TestResult) -> None:
        """Save individual test result.
        
        Args:
            result: Test result to save
        """
        # Create model-specific results directory
        model_dir = self.base_path / "results" / result.model.replace("/", "_")
        model_dir.mkdir(exist_ok=True)
        
        # Save result with timestamp
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.provider}_{result.prompt_id}_{timestamp}.json"
        result_file = model_dir / filename
        
        with open(result_file, 'w') as f:
            json.dump(result.model_dump(mode='json'), f, indent=2, default=str)
    
    def load_test_results(self, model: Optional[str] = None) -> List[TestResult]:
        """Load test results, optionally filtered by model.
        
        Args:
            model: Optional model name to filter by
            
        Returns:
            List of test results
        """
        results = []
        results_dir = self.base_path / "results"
        
        if model:
            # Load results for specific model
            model_dir = results_dir / model.replace("/", "_")
            if model_dir.exists():
                for result_file in model_dir.glob("*.json"):
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    results.append(TestResult(**data))
        else:
            # Load all results
            for model_dir in results_dir.iterdir():
                if model_dir.is_dir():
                    for result_file in model_dir.glob("*.json"):
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        results.append(TestResult(**data))
        
        return results
    
    def generate_provider_summary(self, model: str) -> List[ProviderSummary]:
        """Generate summary statistics for all providers of a model.
        
        Args:
            model: Model name to generate summary for
            
        Returns:
            List of provider summaries
        """
        results = self.load_test_results(model)
        provider_stats = {}
        
        for result in results:
            provider = result.provider
            if provider not in provider_stats:
                provider_stats[provider] = {
                    'total_attempts': 0,
                    'successful_attempts': 0,
                    'error_categories': {}
                }
            
            provider_stats[provider]['total_attempts'] += 1
            if result.success:
                provider_stats[provider]['successful_attempts'] += 1
            else:
                category = result.error_category or 'unknown'
                provider_stats[provider]['error_categories'][category] = \
                    provider_stats[provider]['error_categories'].get(category, 0) + 1
        
        summaries = []
        for provider, stats in provider_stats.items():
            failure_rate = ((stats['total_attempts'] - stats['successful_attempts']) / 
                          stats['total_attempts'] * 100) if stats['total_attempts'] > 0 else 0
            
            summary = ProviderSummary(
                provider=provider,
                model=model,
                total_attempts=stats['total_attempts'],
                successful_attempts=stats['successful_attempts'],
                failure_rate=failure_rate,
                error_categories=stats['error_categories']
            )
            summaries.append(summary)
        
        return summaries
    
    def save_report(self, report_name: str, content: str) -> None:
        """Save a report to the reports directory.
        
        Args:
            report_name: Name of the report file
            content: Report content
        """
        report_file = self.base_path / "reports" / f"{report_name}.md"
        with open(report_file, 'w') as f:
            f.write(content)
    
    def list_models(self) -> List[str]:
        """List all models that have test results.
        
        Returns:
            List of model names
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
        
        return sorted(models)
    
    def clear_results(self, model: Optional[str] = None) -> None:
        """Clear test results, optionally for a specific model.
        
        Args:
            model: Optional model name to clear results for
        """
        results_dir = self.base_path / "results"
        
        if model:
            model_dir = results_dir / model.replace("/", "_")
            if model_dir.exists():
                for result_file in model_dir.glob("*.json"):
                    result_file.unlink()
                model_dir.rmdir()
        else:
            # Clear all results
            for model_dir in results_dir.iterdir():
                if model_dir.is_dir():
                    for result_file in model_dir.glob("*.json"):
                        result_file.unlink()
                    model_dir.rmdir()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics about the validator data.
        
        Returns:
            Dictionary with statistics
        """
        providers = self.load_providers()
        prompts = self.load_prompts()
        models = self.list_models()
        all_results = self.load_test_results()
        
        return {
            'total_providers': len(providers),
            'enabled_providers': len([p for p in providers if p.enabled]),
            'total_prompts': len(prompts),
            'models_tested': len(models),
            'total_test_results': len(all_results),
            'successful_tests': len([r for r in all_results if r.success]),
            'failed_tests': len([r for r in all_results if not r.success])
        }
