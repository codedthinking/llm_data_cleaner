import os
import pandas as pd
from llm_data_cleaner import DataCleaner
from pydantic import BaseModel
from typing import Optional, List

# Define your models
class EducationItem(BaseModel):
    index: int
    year: Optional[List[str]]
    university: Optional[List[str]]

class EducationBatch(BaseModel):
    cleaned: List[Optional[EducationItem]]

class JobTitleItem(BaseModel):
    index: int
    job_title: Optional[str]

class JobTitleBatch(BaseModel):
    cleaned: List[Optional[JobTitleItem]]

# Set your OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY", "")

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
        "description": (
            "Extract year (if present) and university name (if present) from the education strings. "
            "Return a list of year (int or None) and university (string or None) values matching input order."
        ),
        "model": EducationBatch,
    },
    "job_title": {
        "description": (
            "Standardize each job title string to an industry standard format. Return a list of job_title strings."
        ),
        "model": JobTitleBatch,
    },
}
# Initialize the cleaner with a batch size (default is 20)
cleaner = DataCleaner(api_key=api_key, batch_size=20)

# Clean the data
result = cleaner.clean_dataframe(df,instructions)

# Display results
print("Original Data:")
print(df)
print("\nCleaned Data:")
print(result)

# You can also save the results to a CSV file
result.to_csv("cleaned_data.csv", index=False)
