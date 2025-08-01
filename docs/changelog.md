# Changelog

## 2025-08-01

### Added
- Enhanced validation error tracking to detect errors in successful tests
- New TestResult model fields for validation_errors, validation_error_count, and perfect_success
- Specialized validation error reporting to distinguish between perfect and partial success
- Detailed validation error patterns analysis in reports

### Changed
- Modified agent.py to detect and track validation errors in tool responses
- Updated test_runner.py to process validation errors in both successful and failed tests
- Enhanced report generation to include validation error statistics
- Improved JSON report to include perfect_success metrics

### Fixed
- Fixed issue where validation errors during successful tests were not being tracked
- Fixed misleading 100% success rates for models that had validation errors but completed steps