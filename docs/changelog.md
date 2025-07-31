# Changelog

## 2025-07-31

### Added
- Parallel provider testing capability
- Provider-specific test directories for isolated testing
- Command-line option `--sequential` to disable parallel testing
- Custom test directory support via environment variables
- Template-based test file initialization
- Documentation for parallel testing features
- Improved path resolution in FileSystemTestHelper

### Fixed
- Report file creation issues with multiple providers
- File conflicts during concurrent testing
- Path resolution issues when using relative paths
