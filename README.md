# LLM Data Cleaner

A Python package for cleaning data using OpenAI API. This package allows you to define cleaning instructions for each column in a CSV file and process them using OpenAI's language models.

## Installation

```bash
pip install git+https://github.com/codedthinking/llm_data_cleaner.git
```

Or with Poetry:

```bash
poetry add git+https://github.com/codedthinking/llm_data_cleaner.git
```

## Usage

```python
import pandas as pd
from llm_data_cleaner import DataCleaner

# Define cleaning instructions
cleaning_instructions = {
    "education": {
        "prompt": "The rows below contain education experience of individuals residing in Hungary. Extract the year of higher education degree (may be None) and the precise, non abbreviated name of the university (may be None).",
        "schema": {
            "type": "object",
            "properties": {
                "year": {"type": ["integer", "null"]},
                "university": {"type": ["string", "null"]}
            },
            "required": ["year", "university"]
        }
    },
    "job_title": {
        "prompt": "Standardize the job title according to industry standards. Return the standardized job title."
    }
}

# Initialize the cleaner
cleaner = DataCleaner(api_key="your-openai-api-key")

# Clean the data
df = pd.read_csv("your_data.csv")
result = cleaner.clean_dataframe(df, cleaning_instructions)

# Output results
print(result)
```

## Features

- Process CSV data using OpenAI API
- Define custom cleaning prompts for each column
- Optional JSON schema validation for responses
- Batch processing to handle rate limits
- Progress tracking with tqdm

## License

MIT
