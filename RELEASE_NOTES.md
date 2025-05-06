# NeoMT3 Release Notes

## Version 0.1.0 (May 2025)

### Major Changes

- **Project Renamed**: Renamed from MT3 to NeoMT3 to reflect the new direction of the project
- **Framework Migration**: Migrated from seqio to transformers, making the project more accessible and compatible with modern ML workflows
- **Dependency Management**: Added Poetry for improved dependency management
- **Project Structure**: Completely restructured the project layout for better organization and maintainability

### Features

- New transformers-based implementation for more flexible model integration
- Updated colab notebook for easy music transcription with the new framework
- Simplified API for better developer experience
- Improved test suite for better code quality

### Bug Fixes

- Fixed spectrogram test failures
- Fixed issue with event codec implementation
- Fixed uint32 initialization issue in layers.py

### Infrastructure Improvements

- Added GitHub Actions workflows for continuous integration
- Added PyPI publishing workflow for automated releases
- Added linting tools:
  - Black for code formatting
  - isort for import sorting
  - pre-commit hooks for automated code quality checks

### Documentation

- Updated README with new maintainer information
- Improved documentation for the new architecture
- Added CONTRIBUTING.md with contribution guidelines

### Developer Notes

This is the first release of NeoMT3, a fork of the original MT3 project by Google Research, now maintained by Igor Bogicevic. The project continues to use the T5X framework for multi-instrument automatic music transcription, but with a modernized codebase and dependency structure.

For issues and feedback, please use the [GitHub repository](https://github.com/probablyrobot/neomt3). 