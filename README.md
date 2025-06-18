# LLM Data Cleaner

LLM Data Cleaner automates the transformation of messy text columns into well structured data using OpenAI models. It eliminates the need for complex regular expressions or manual parsing while ensuring the output conforms to a schema.

## Why use it?

- **Less manual work** – delegate repetitive cleaning tasks to a language model.
- **Consistent results** – validate responses with Pydantic models.
- **Batch processing** – send rows in chunks to respect API rate limits.

## Installation

Requires **Python 3.9+**.

```bash
pip install git+https://github.com/codedthinking/llm_data_cleaner.git
```

Or with Poetry:

```bash
poetry add git+https://github.com/codedthinking/llm_data_cleaner.git
```

## Step by step

1. Create Pydantic models describing the cleaned values.
2. Define a dictionary of instructions mapping column names to a prompt and schema.
3. Instantiate `DataCleaner` with your OpenAI API key.
4. Load your raw CSV file with `pandas`.
5. Call `clean_dataframe(df, instructions)`.
6. Inspect the returned DataFrame which contains new `cleaned_*` columns.
7. Save or further process the cleaned data.

## Example: inline models

```python
import pandas as pd
from pydantic import BaseModel
from llm_data_cleaner import DataCleaner

class AddressItem(BaseModel):
    index: int
    city: str | None
    country: str | None
    postal_code: str | None

class TitleItem(BaseModel):
    index: int
    profession: str | None

instructions = {
    "address": {
        "prompt": "Extract city, country and postal code if present.",
        "schema": AddressItem,
    },
    "profession": {
        "prompt": "Normalize the profession to a standard job title.",
        "schema": TitleItem,
    },
}

cleaner = DataCleaner(api_key="YOUR_OPENAI_API_KEY")
raw_df = pd.DataFrame({
    "address": ["Budapest Váci út 1", "1200 Vienna Mariahilfer Straße 10"],
    "profession": ["dev", "data eng"]
})
cleaned = cleaner.clean_dataframe(raw_df, instructions)
print(cleaned)
```

## Example: loading YAML instructions

```python
from llm_data_cleaner import DataCleaner, load_yaml_instructions
import pandas as pd

instructions = load_yaml_instructions("instructions.yaml")
cleaner = DataCleaner(api_key="YOUR_OPENAI_API_KEY", system_prompt="{column_prompt}")
raw_df = pd.read_csv("data.csv")
result = cleaner.clean_dataframe(raw_df, instructions)
```

`load_yaml_instructions` reads the same structure shown above from a YAML file so
cleaning rules can be shared without modifying code.

## Authors

- Miklós Koren
- Gergely Attila Kiss

## Preferred citation

If you use **LLM Data Cleaner** in your research, please cite the project as
specified in [`CITATION.cff`](./CITATION.cff).

## License

MIT
