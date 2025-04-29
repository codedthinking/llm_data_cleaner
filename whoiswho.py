import pandas as pd
from llm_data_cleaner import DataCleaner
from pydantic import BaseModel
from typing import Optional, List
import os

class Occupation(BaseModel):
    index: int
    occupation: Optional[str]

class Company(BaseModel):
    index: int
    company: Optional[str]

class Education(BaseModel):
    index: int
    year: Optional[List[str]]
    institute: Optional[List[str]]

class Career(BaseModel):
    index: int
    ccompany: Optional[List[str]]
    coccupation: Optional[List[str]]
    cyear: Optional[List[str]]

api_key = os.environ.get("OPENAI_API_KEY", "")


data = pd.read_csv("whoiswho100.csv")
data.columns = [col.lower() for col in data.columns]

instructions = {
    "occupation": {
        "description": (
            "Standardize each job title string to an industry standard format. Return a string of occupation name."
        ),
        "model": Occupation,
        },
    "company": {
        "description": (
            "Standardize each company name string to an industry standard format. Return a string of company name."
        ),
        "model": Company,
        },
    "education": {
        "description": (
            "Extract year (if present, keep intervals when possible) and institution name (if present) from the education strings."
            "Ignore non-education related text."
            "Return a list of year (string or None) and university (string or None) values matching input order."
        ),
        "model": Education,
        },
    "carrier": {
        "description": (
            "Extract year (if present, keep intervals when possible) and company name (if present) and occupation name (if present)"
            "from the carrier strings. Ignore non-carrier related text."
            "Return a list of year (string or None), company (string or None) and occupation (string or None) values matching input order."
        ),
        "model": Career,
        },
}

cleaner = DataCleaner(api_key=api_key, batch_size=50)

result = cleaner.clean_dataframe(data, instructions)

print("Original Data:")
print(data.head())
print("\nCleaned Data:")
print(result.head())

result.to_csv("whoiswho_cleaned.csv", index=False)
