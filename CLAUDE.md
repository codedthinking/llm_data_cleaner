# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Data Cleaner is a Python package that automates the transformation of messy text columns into structured data using various language models via the LLM API. It eliminates complex regex patterns and manual parsing by leveraging LLMs with schema validation.

## Development Commands

This project uses Poetry for dependency management:

```bash
# Setup and dependencies
poetry install                    # Install all dependencies
poetry install --only=main      # Install only production dependencies

# Testing
poetry run pytest               # Run all tests
poetry run pytest tests/test_cleaner.py  # Run specific test file
poetry run pytest -v            # Verbose test output
poetry run pytest -k "test_name"  # Run specific test

# Code quality
poetry run black .              # Format code
poetry run isort .              # Sort imports  
poetry run flake8              # Lint code

# Package building
poetry build                    # Build package
poetry version patch|minor|major  # Bump version
```

## Core Architecture

The package has a clean modular structure:

- **`DataCleaner`** (`cleaner.py`): Main class handling LLM API interactions, batch processing, and data transformation
- **`utils.py`**: Contains `jsonize()` utility and type definitions (`InstructionField`, `InstructionSchema`)
- **YAML instruction loading**: Dynamic Pydantic model creation from YAML configuration files

### Key Technical Patterns

- Uses LLM API's structured output feature with Pydantic schemas for reliable JSON responses
- Implements batch processing with configurable sizes to respect API rate limits
- Dynamic Pydantic model creation allows flexible schema definitions
- Proper retry logic with exponential backoff for API resilience
- Model-agnostic design - works with any LLM API supported model

### Data Flow

1. Load DataFrame and cleaning instructions (programmatic or YAML)
2. Process columns in batches via LLM API with structured outputs
3. Validate responses against Pydantic schemas
4. Return enhanced DataFrame with `cleaned_*` prefixed columns alongside originals

## Configuration Requirements

- **Python 3.9+** required
- **LLM API** must be installed and configured (`pip install llm`)
- **Model setup** via LLM plugins (e.g., `llm install llm-openai` for OpenAI models)
- **API keys** managed by LLM API (e.g., `llm keys set openai`)
- Default model: `gpt-4o-mini` (configurable to any LLM-supported model)

## Important Files

- `pyproject.toml`: Poetry configuration and package metadata
- `instructions.yaml`: Example YAML configuration for cleaning rules
- `tests/test_data/sample.csv`: Test data fixture

## Version Management

The project uses Poetry's version management. Current version is 0.5.0 after migrating to LLM API. Maintain consistency between `pyproject.toml` and `CITATION.cff` when updating versions.

## Testing Strategy

Tests use mocked LLM API calls to avoid real API usage. The mocks patch `llm.get_model` and simulate model responses. Run tests before any API-related changes to ensure mocking remains effective. Test data fixtures are in `tests/test_data/`.