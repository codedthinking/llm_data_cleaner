import json
import logging
import time
from typing import Dict, List, Optional, Union, Any

import pandas as pd
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm
from jsonschema import validate, ValidationError

from .utils import batch_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    A class for cleaning data using OpenAI API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-2024-08-06",
        max_retries: int = 3,
        retry_delay: int = 5,
        batch_size: int = 20,
        temperature: float = 0.0,
    ):
        """
        Initialize the DataCleaner.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Delay between retry attempts in seconds
            batch_size: Number of rows to process in a batch
            temperature: Temperature setting for the OpenAI API
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.temperature = temperature

    def clean_dataframe(
        self, df: pd.DataFrame, instructions: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Clean a DataFrame using OpenAI API.

        Args:
            df: Input DataFrame
            instructions: Dictionary mapping column names to cleaning instructions

        Returns:
            DataFrame with cleaned data and original values
        """
        results_df = pd.DataFrame()
        
        # Process each column specified in the instructions
        for column, instruction in instructions.items():
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame. Skipping.")
                continue
                
            logger.info(f"Processing column: {column}")
            
            # Create a working DataFrame with the index and the column to process
            working_df = df[[column]].copy()
            working_df.reset_index(inplace=True)
            
            # Get prompt, schema, and expected output structure from instructions
            prompt = instruction.get("prompt", "")
            schema = instruction.get("schema")  # This is optional
            
            # Include the expected output format in the prompt when there's a schema
            # result may be a literal string etc, this is in the schema. or it is an object or an array
            if schema:
                if schema.get("type") == "object":
                    # Add schema type to the prompt
                    prompt += f"\nThe expected output is a JSON object with the following structure:\n"
                # Extract required field names from schema
                required_fields = schema.get("required", [])
                properties = schema.get("properties", {})
                
                # Add output structure guidance to the prompt
                field_guidance = ""
                for field in required_fields:
                    field_type = properties.get(field, {}).get("type", "any")
                    field_guidance += f"- '{field}': {field_type}\n"
                
                output_guidance = f"\nYour response must be a JSON object with exactly these fields:\n{field_guidance}"
                prompt += output_guidance
            
            # Process in batches
            batches = batch_dataframe(working_df, self.batch_size)
            processed_batches = []
            
            for batch in tqdm(batches, desc=f"Cleaning {column}"):
                cleaned_batch = self._process_batch(batch, column, prompt, schema)
                processed_batches.append(cleaned_batch)
            
            # Combine processed batches
            column_results = pd.concat(processed_batches, ignore_index=True)
            
            # Rename columns for clarity
            column_results.rename(
                columns={
                    "index": f"original_index_{column}",
                    column: f"original_{column}",
                    f"cleaned_{column}": f"cleaned_{column}"
                },
                inplace=True
            )
            
            # Merge with results_df or create it if it's the first column
            if results_df.empty:
                results_df = column_results
            else:
                results_df = pd.merge(
                    results_df,
                    column_results,
                    left_on=f"original_index_{list(instructions.keys())[0]}",
                    right_on=f"original_index_{column}",
                    how="outer"
                )
        
        return results_df

    def _process_batch(
        self, 
        batch: pd.DataFrame, 
        column: str, 
        prompt: str, 
        schema: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Process a batch of rows for a specific column.

        Args:
            batch: Batch DataFrame
            column: Column name to process
            prompt: Cleaning prompt
            schema: JSON schema for validation (optional)

        Returns:
            Processed batch
        """
        result_batch = batch.copy()
        result_batch[f"cleaned_{column}"] = None
        
        for idx, row in batch.iterrows():
            value = row[column]
            
            if pd.isna(value):
                result_batch.at[idx, f"cleaned_{column}"] = json.dumps({"error": "Original value is NaN"})
                continue
            
            # Process individual value
            messages = [
                {"role": "system", "content": "You are a data cleaning assistant. Your task is to clean and structure data according to the instructions. Respond with valid JSON literal (string, number, boolean or null), or JSON array, or JSON object."},
                {"role": "user", "content": f"{prompt}\n\nData to clean: {value}"}
            ]
            
            cleaned_value = self._call_openai_with_retry(messages, schema)
            
            # Normalize keys if schema is provided
            if schema and "properties" in schema and isinstance(cleaned_value, dict):
                normalized_value = self._normalize_keys(cleaned_value, schema)
                
                # Validate the normalized value
                try:
                    validate(instance=normalized_value, schema=schema)
                    cleaned_value = normalized_value
                except ValidationError as e:
                    cleaned_value = {"error": f"Schema validation failed: {str(e)}", "data": cleaned_value}
            
            result_batch.at[idx, f"cleaned_{column}"] = json.dumps(cleaned_value)
        
        return result_batch
        
    def _normalize_keys(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize keys in the response to match the schema.
        This handles cases where the model returns different key names than the schema expects.
        
        Args:
            data: The data to normalize
            schema: The JSON schema with expected keys
            
        Returns:
            Normalized data with keys that match the schema
        """
        if not isinstance(data, dict) or not schema or "properties" not in schema:
            return data
            
        required_keys = schema.get("properties", {}).keys()
        
        # Simple case: keys already match the schema
        if all(key in data for key in required_keys):
            return data
            
        # Try to normalize common key variations
        normalized = {}
        key_mapping = {}
        
        # Generate potential mappings from actual keys to schema keys
        for schema_key in required_keys:
            # Various common transformations of keys
            variations = [
                schema_key,                           # exact match
                schema_key.lower(),                   # lowercase
                schema_key.upper(),                   # uppercase
                schema_key.replace("_", ""),          # no underscores
                f"{schema_key}_value",                # with _value suffix
                f"{schema_key}_name",                 # with _name suffix
                f"{schema_key}_date",                 # with _date suffix
                schema_key.replace("_", " "),         # spaces instead of underscores
                f"{schema_key.split('_')[0]}",        # first part of compound key
                f"{schema_key.split('_')[-1]}"        # last part of compound key
            ]
            
            # Look for similar keys in the data
            for data_key in data.keys():
                if data_key in variations or schema_key in data_key.lower():
                    key_mapping[data_key] = schema_key
                    break
                    
        # If we have mappings, apply them
        for data_key, schema_key in key_mapping.items():
            normalized[schema_key] = data[data_key]
            
        # For any remaining schema keys, check if there are obvious matches in the data
        for schema_key in required_keys:
            if schema_key not in normalized:
                for data_key in data.keys():
                    # Try to match keys based on similarity
                    if (data_key.lower() in schema_key.lower() or 
                        schema_key.lower() in data_key.lower()):
                        normalized[schema_key] = data[data_key]
                        break
                        
        # If we still don't have all required keys, set the missing ones to None
        for schema_key in required_keys:
            if schema_key not in normalized:
                normalized[schema_key] = None
                
        return normalized

    def _call_openai_with_retry(
        self, 
        messages: List[Dict[str, str]], 
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call OpenAI API with retry logic.

        Args:
            messages: List of message dictionaries
            schema: JSON schema for validation (optional)

        Returns:
            Cleaned data as a dictionary
        """
        for attempt in range(self.max_retries):
            try:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                try:
                    return json.loads(response.choices[0].message.content)
                except json.JSONDecodeError:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Failed to parse JSON response. Retrying ({attempt + 1}/{self.max_retries})...")
                        time.sleep(self.retry_delay)
                    else:
                        return {"error": "Failed to parse JSON response"}
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"API call failed: {str(e)}. Retrying ({attempt + 1}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    return {"error": f"API call failed: {str(e)}"}
        
        # If we get here, all retries failed
        return {"error": "Failed after maximum retries"}