import os
import pandas as pd
from llm_data_cleaner import DataCleaner, load_yaml_instructions
from pydantic import BaseModel
from typing import Optional, List

# Define your models
class EducationItem(BaseModel):
    index: int
    year: Optional[List[str]]
    university: Optional[List[str]]

class JobTitleItem(BaseModel):
    index: int
    job_title: Optional[str]


yaml_instructions = load_yaml_instructions("instructions.yaml")
print(yaml_instructions)
#raise ValueError("Instructions loaded from YAML file.")

# Set your OpenAI API key, reading from .secrets/OPENAI_API_KEY
with open(".secrets/OPENAI_API_KEY", "r") as f:
    api_key = f.read().strip()
# Ensure the API key is set
if not api_key:
    raise ValueError("API key is not set. Please provide a valid OpenAI API key.")
# Create a sample DataFrame
data = {
    "education": [
        "BS from Technical University of Budapest, 2005",
        "Master's degree from Eotvos Lorand University in Budapest (ELTE), 2018",
        "No higher education",
        "PhD in Computer Science, CEU, 2010"
    ],
    "job_title": [
        "sw engineer",
        "Senior Software Architect",
        "Jr. Data Scientist",
        "Chief Technology Officer"
    ]
}

df = pd.DataFrame(data)

# Define cleaning instructions - schema is optional
instructions = {
    "education": {
        "prompt": (
            "Extract year (if present) and university name (if present) from the education strings. "
            "Return a list of year (int or None) and university (string or None) values matching input order."
        ),
        "schema": EducationItem,
    },
    "job_title": {
        "prompt": (
            "Standardize each job title string to an industry standard format. Return a list of job_title strings."
        ),
        "schema": JobTitleItem,
    },
}
# Initialize the cleaner with a batch size (default is 20)
cleaner = DataCleaner(api_key=api_key, batch_size=20)

# Clean the data
result2 = cleaner.clean_dataframe(df, yaml_instructions)

# Display results
print("Original Data:")
print(df)
print("\nCleaned Data:")
print(result2)

# You can also save the results to a CSV file
result2.to_csv("cleaned_data.csv", index=False)
