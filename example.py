import os
import pandas as pd
from llm_data_cleaner import DataCleaner, load_yaml_instructions
from pydantic import BaseModel
from typing import Optional, List


yaml_instructions = load_yaml_instructions("instructions.yaml")
file_path = "feor08_kozl_melleklet.pdf"

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

# Initialize the cleaner with a batch size (default is 20)
cleaner = DataCleaner(
    api_key=api_key, 
    batch_size=20, 
    system_prompt='Follow these instructions, but return the answers in Greek. {column_prompt}.',
    file=file_path
)

# Clean the data
result = cleaner.clean_dataframe(df, yaml_instructions)

# Display results
print("Original Data:")
print(df)
print("\nCleaned Data:")
print(result)

# You can also save the results to a CSV file
result.to_csv("cleaned_data.csv", index=False)
