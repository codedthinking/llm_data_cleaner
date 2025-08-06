import json
import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from llm_data_cleaner import DataCleaner
from llm_data_cleaner.utils import jsonize


class TestUtils(unittest.TestCase):
    def test_jsonize(self):
        from pydantic import BaseModel
        
        # Test with BaseModel
        class TestModel(BaseModel):
            name: str
            age: int
            
        model = TestModel(name="John", age=30)
        result = jsonize(model)
        self.assertIsInstance(result, str)
        
        # Test with primitive values
        self.assertEqual(jsonize("test"), "test")
        self.assertEqual(jsonize(42), 42)
        self.assertEqual(jsonize({"key": "value"}), {"key": "value"})


class TestDataCleaner(unittest.TestCase):
    @patch('llm_data_cleaner.cleaner.llm.get_model')
    def test_clean_dataframe(self, mock_get_model):
        # Create a mock model response
        mock_response = MagicMock()
        mock_response.text.return_value = json.dumps({
            "cleaned": [
                {"index": 0, "year": 2020, "university": "Example University"}
            ]
        })
        
        # Set up the mock model
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response
        mock_get_model.return_value = mock_model
        
        # Create a test DataFrame
        df = pd.DataFrame({
            "education": ["Graduated from Example University in 2020"]
        })
        
        # Create cleaning instructions using the new InstructionSchema format
        from pydantic import BaseModel, Field
        from typing import Optional
        
        class EducationData(BaseModel):
            index: int
            year: Optional[int] = None
            university: Optional[str] = None
        
        instructions = {
            "education": {
                "prompt": "Extract year and university name",
                "schema": EducationData
            }
        }
        
        # Initialize the cleaner (no api_key needed for LLM API)
        cleaner = DataCleaner(model="gpt-4o-mini")
        
        # Clean the data
        result = cleaner.clean_dataframe(df, instructions)
        
        # Check that the model was called
        mock_get_model.assert_called_once_with("gpt-4o-mini")
        mock_model.prompt.assert_called()
        
        # Verify the call arguments
        call_args = mock_model.prompt.call_args
        self.assertIn("schema", call_args[1])
        self.assertIn("temperature", call_args[1])
        self.assertEqual(call_args[1]["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()