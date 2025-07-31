# Changelog

## [1.2.0] - 2025-07-31

### Fixed
- Fixed directory structure for test results to follow a more logical hierarchy ([directory_structure_fix.md](directory_structure_fix.md))
- Resolved FileNotFoundError when trying to load results with the previous structure
- Updated client.py and agent.py to use consistent directory paths

### Added
- Added migration script for existing test results
- Added documentation for directory structure organization

## [1.1.0] - 2025-07-31

### Fixed
- Fixed message serialization issues with datetime objects ([serialization_fixes.md](serialization_fixes.md))
- Improved serialization for complex objects to ensure tests always save results

### Added
- Added `serialization_helper.py` with JSON serialization utilities
- Added documentation on message serialization fixes
- Added test script for verifying serialization functions

## [1.0.0] - 2025-07-30 (Initial Release)

### Added
- Initial implementation of OpenRouter Provider Validator
- Created file system test infrastructure
- Added agent and MCP server for testing providers
- Implemented reporting and analysis tools
- Added support for multiple prompt sequences
