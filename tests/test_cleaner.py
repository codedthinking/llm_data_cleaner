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
    @patch('llm_data_cleaner.cleaner.OpenAI')
    def test_clean_dataframe(self, mock_openai):
        # Create a mock response
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = json.dumps({"year": 2020, "university": "Example University"})
        
        # Set up the mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client
        
        # Create a test DataFrame
        df = pd.DataFrame({
            "education": ["Graduated from Example University in 2020"]
        })
        
        # Create cleaning instructions
        instructions = {
            "education": {
                "prompt": "Extract year and university name",
                "schema": {
                    "type": "object",
                    "properties": {
                        "year": {"type": ["integer", "null"]},
                        "university": {"type": ["string", "null"]}
                    },
                    "required": ["year", "university"]
                }
            }
        }
        
        # Initialize the cleaner
        cleaner = DataCleaner(api_key="dummy-key")
        
        # Clean the data
        result = cleaner.clean_dataframe(df, instructions)
        
        # Assertions
        self.assertIn("original_education", result.columns)
        self.assertIn("cleaned_education", result.columns)
        
        # Check the API was called with the correct arguments
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4o")
        self.assertEqual(call_args["temperature"], 0.0)
        self.assertEqual(call_args["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()