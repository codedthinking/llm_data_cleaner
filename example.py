import os
import pandas as pd
from llm_data_cleaner import DataCleaner

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
cleaning_instructions = {
    "education": {
        "prompt": "The rows below contain education experience of individuals residing in Hungary. Extract the year of higher education degree (may be None) and the precise, non abbreviated name of the university (may be None).",
        # Optional schema
        "schema": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "array",
                    "items": {"type": ["integer", "null"]}
                },
                "university": {
                    "type": "array",
                    "items": {"type": ["string", "null"]}
                }
            },
            "required": ["year", "university"]
        }
    },
    "job_title": {
        "prompt": "Standardize the job title according to industry standards. Return the standardized job title."
        # No schema for this column
    }
}

# Initialize the cleaner with a batch size (default is 20)
cleaner = DataCleaner(
    api_key=api_key,
    batch_size=20  # Process up to 20 rows in a single API call
)

# Clean the data
result = cleaner.clean_dataframe(df, cleaning_instructions)

# Display results
print("Original Data:")
print(df)
print("\nCleaned Data:")
print(result)

# You can also save the results to a CSV file
result.to_csv("cleaned_data.csv", index=False)