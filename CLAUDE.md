# CLAUDE.md

## Project Overview

**LLM Data Cleaner** is a Python package that automates the transformation of messy text columns into well-structured data using LLM APIs. It supports **OpenAI, OpenRouter, Anthropic, and 100+ other providers** through LiteLLM. The package eliminates the need for complex regular expressions or manual parsing while ensuring output conforms to a schema.

### Key Features
- **Provider-agnostic** - Works with OpenAI, OpenRouter, Anthropic, and 100+ providers via LiteLLM
- **Automated data cleaning** using language models
- **Schema validation** with Pydantic models
- **Batch processing** to respect API rate limits
- **YAML-based configuration** for reusable cleaning instructions
- **Consistent results** through structured output parsing

## Project Structure

```
llm_data_cleaner/
├── llm_data_cleaner/          # Main package
│   ├── __init__.py            # Package exports
│   ├── cleaner.py             # DataCleaner class and YAML loader
│   └── utils.py               # Utility functions and schema definitions
├── tests/                     # Test suite
│   ├── test_cleaner.py        # DataCleaner tests
│   ├── test_utils.py          # Utility function tests
│   └── test_data/             # Test data files
├── example.py                 # Usage example
├── instructions.yaml          # Example YAML instructions
├── pyproject.toml             # Poetry dependencies
└── README.md                  # User documentation
```

## Core Components

### 1. DataCleaner (`llm_data_cleaner/cleaner.py`)

The main class that orchestrates data cleaning operations.

**Key Responsibilities:**
- Batch processing of DataFrame columns
- Communication with LLM APIs using LiteLLM for provider-agnostic access
- Structured output parsing with JSON mode and Pydantic validation
- Retry logic for failed API calls
- Progress tracking with tqdm

**Important Methods:**
- `clean_dataframe(df, instructions)` - Main entry point for cleaning data
- `_process_batch()` - Processes a single batch of rows
- `_clean_batch()` - Makes API calls with retry logic using LiteLLM
- `_make_batch_model()` - Creates Pydantic models for batch responses

**Configuration Parameters:**
- `api_key`: API key for the LLM provider (optional if set via environment variables)
- `model`: Model name (default: "gpt-4o-2024-08-06"). Use provider prefixes for non-OpenAI models (e.g., "openrouter/anthropic/claude-3-opus")
- `batch_size`: Number of rows per API call (default: 10)
- `max_retries`: Retry attempts for failed calls (default: 3)
- `retry_delay`: Seconds between retries (default: 5)
- `temperature`: Model temperature (default: 0.0)
- `system_prompt`: Custom system prompt template (optional)
- `api_base`: Base URL for the API (e.g., "https://openrouter.ai/api/v1" for OpenRouter)
- `**litellm_kwargs`: Additional arguments to pass to litellm.completion()

### 2. Utilities (`llm_data_cleaner/utils.py`)

Helper functions and schema definitions.

**Key Components:**
- `jsonize()` - Converts Pydantic models and lists to JSON strings for storage
- `InstructionField` - Pydantic model for individual column instructions
- `InstructionSchema` - Root model for the complete instruction set

### 3. YAML Instructions Loader (`llm_data_cleaner/cleaner.py:162`)

`load_yaml_instructions()` reads cleaning instructions from YAML files.

**YAML Structure:**
```yaml
column_name:
  prompt: "Instruction for cleaning this column"
  schema:
    type: "object"
    properties:
      index:
        type: "integer"
      field_name:
        type: "string"
```

**Supported Types:**
- Primitives: `string`, `integer`, `number`, `bool`
- Collections: `array`, `object`
- Optional fields via `optional: true`
- Nested objects (automatically converted to Pydantic models)

## Dependencies

### Production
- `litellm ^1.0.0` - Unified API client for 100+ LLM providers (OpenAI, OpenRouter, Anthropic, etc.)
- `pydantic ^2.0.0` - Data validation and schema definition
- `pandas ^2.2.3` - DataFrame operations
- `pyyaml ^6.0.2` - YAML parsing
- `jsonschema ^4.0.0` - JSON schema validation
- `tqdm ^4.65.0` - Progress bars

### Development
- `pytest ^7.0.0` - Testing framework
- `black ^23.0.0` - Code formatting
- `isort ^5.0.0` - Import sorting
- `flake8 ^6.0.0` - Linting

## Development Guidelines

### Code Style
- Use **Black** for code formatting
- Use **isort** for import sorting
- Follow PEP 8 guidelines
- Use type hints for function signatures

### Naming Conventions
- **Classes**: PascalCase (e.g., `DataCleaner`, `InstructionField`)
- **Functions/Methods**: snake_case (e.g., `clean_dataframe`, `load_yaml_instructions`)
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: Leading underscore (e.g., `_process_batch`)

### Important Patterns

#### Schema Definition
Instructions use either Pydantic models or YAML schemas:

**Pydantic:**
```python
class MyItem(BaseModel):
    index: int
    field_name: str | None
```

**YAML:**
```yaml
column:
  prompt: "..."
  schema:
    type: "object"
    properties:
      index: {type: "integer"}
      field_name: {type: "string"}
```

#### Cleaned Column Names
The package adds cleaned data with the prefix `cleaned_`. For example, cleaning the `education` column creates `cleaned_education`.

#### System Prompt Template
Must contain `{column_prompt}` placeholder for column-specific instructions. Example:
```python
system_prompt="{column_prompt}. Return results in JSON format."
```

### Testing

Run tests with:
```bash
pytest
```

Test files are located in `tests/`:
- `test_cleaner.py` - DataCleaner functionality
- `test_utils.py` - Utility functions
- `test_data/` - Sample data for testing

### Common Issues

1. **API Key Not Set**: Set API key via environment variables (OPENAI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY) or pass directly to constructor
2. **Rate Limits**: Adjust `batch_size` to control API call frequency
3. **Schema Validation Errors**: Ensure Pydantic models include `index: int` field
4. **Missing Columns**: DataCleaner skips columns not in DataFrame with a warning
5. **Provider-specific Configuration**: Use `api_base` parameter for custom endpoints (e.g., OpenRouter)
6. **Model Naming**: Use provider prefixes for non-OpenAI models (e.g., "openrouter/model-name", "anthropic/model-name")

## API Changes

### Version 0.5.0
- **BREAKING**: Migrated from OpenAI library to LiteLLM for provider-agnostic support
- Now supports OpenAI, OpenRouter, Anthropic, and 100+ other providers
- Added `api_base` parameter for custom API endpoints
- Added `**litellm_kwargs` for provider-specific configuration
- API key is now optional in constructor (can use environment variables)
- Model parameter now supports provider prefixes (e.g., "openrouter/", "anthropic/")

### Version 0.4.x
- Migrated from deprecated `client.responses.parse()` to supported OpenAI methods
- Added `jsonize()` utility for consistent data serialization
- Harmonized spelling: "normalize" (US) throughout codebase

### Python Version Support
- Requires Python 3.9 or higher
- Tested with Python 3.9, 3.10, 3.11, 3.12

## Example Workflow

1. **Define schemas** (Pydantic or YAML)
2. **Create instructions** dictionary mapping columns to prompts and schemas
3. **Initialize DataCleaner** with API key, model name, and configuration
4. **Load data** into pandas DataFrame
5. **Call `clean_dataframe()`** to process
6. **Access results** in `cleaned_*` columns

### Provider-Specific Examples

**OpenAI (default):**
```python
cleaner = DataCleaner(api_key="sk-...", model="gpt-4o-2024-08-06")
```

**OpenRouter:**
```python
cleaner = DataCleaner(
    api_key="sk-or-...",
    model="openrouter/anthropic/claude-3-opus",
    api_base="https://openrouter.ai/api/v1"
)
```

**Anthropic:**
```python
cleaner = DataCleaner(api_key="sk-ant-...", model="claude-3-opus-20240229")
```

## Authors
- Miklós Koren (koren@codedthinking.com)
- Gergely Attila Kiss (kiss@codedthinking.com)

## Citation
See `CITATION.cff` for preferred citation format.

## License
MIT License
