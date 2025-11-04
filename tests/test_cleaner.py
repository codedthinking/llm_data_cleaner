import json
import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from llm_data_cleaner import DataCleaner
from llm_data_cleaner.utils import batch_dataframe, validate_instructions


class TestUtils(unittest.TestCase):
    def test_batch_dataframe(self):
        # Create a test DataFrame
        df = pd.DataFrame({"A": range(10)})
        
        # Test with batch size equal to DataFrame size
        batches = batch_dataframe(df, 10)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 10)
        
        # Test with batch size smaller than DataFrame size
        batches = batch_dataframe(df, 3)
        self.assertEqual(len(batches), 4)  # 3 + 3 + 3 + 1 = 10
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[-1]), 1)
        
        # Test with batch size larger than DataFrame size
        batches = batch_dataframe(df, 20)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 10)
    
    def test_validate_instructions(self):
        # Valid instructions
        valid_instructions = {
            "column1": {
                "prompt": "Clean this data"
            },
            "column2": {
                "prompt": "Clean this data",
                "schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"}
                    }
                }
            }
        }
        errors = validate_instructions(valid_instructions)
        self.assertEqual(len(errors), 0)
        
        # Invalid instructions
        invalid_instructions = {
            123: {  # Non-string column name
                "prompt": "Clean this data"
            },
            "column2": "not a dict",  # Non-dict instruction
            "column3": {  # Missing prompt
                "schema": {}
            },
            "column4": {  # Non-string prompt
                "prompt": 123
            },
            "column5": {  # Non-dict schema
                "prompt": "Clean this data",
                "schema": "invalid"
            }
        }
        errors = validate_instructions(invalid_instructions)
        self.assertEqual(len(errors), 5)


class TestDataCleaner(unittest.TestCase):
    @patch('llm_data_cleaner.cleaner.litellm.completion')
    def test_clean_dataframe(self, mock_completion):
        # Create a mock response matching LiteLLM's response structure
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "cleaned": [
                {"index": 0, "year": 2020, "university": "Example University"}
            ]
        })
        mock_completion.return_value = mock_response

        # Create a test DataFrame
        df = pd.DataFrame({
            "education": ["Graduated from Example University in 2020"]
        })

        # Create Pydantic model for testing
        from pydantic import BaseModel

        class EducationItem(BaseModel):
            index: int
            year: int | None
            university: str | None

        # Create cleaning instructions
        instructions = {
            "education": {
                "prompt": "Extract year and university name",
                "schema": EducationItem
            }
        }

        # Initialize the cleaner
        cleaner = DataCleaner(api_key="dummy-key")

        # Clean the data
        result = cleaner.clean_dataframe(df, instructions)

        # Assertions
        self.assertIn("cleaned_year", result.columns)
        self.assertIn("cleaned_university", result.columns)

        # Check the API was called
        mock_completion.assert_called()
        call_args = mock_completion.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4o-2024-08-06")
        self.assertEqual(call_args["temperature"], 0.0)
        self.assertEqual(call_args["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()