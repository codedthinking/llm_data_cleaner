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
        model: str = "gpt-4o",
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
            
            # Get prompt and schema from instructions
            prompt = instruction.get("prompt", "")
            schema = instruction.get("schema")  # This is optional
            
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
        
        # Filter out NaN values
        valid_data = [(idx, str(val)) for idx, val in batch[column].items() if not pd.isna(val)]
        
        # Set NaN results for empty values
        for idx, row in batch.iterrows():
            if pd.isna(row[column]):
                result_batch.at[idx, f"cleaned_{column}"] = json.dumps({"error": "Original value is NaN"})
        
        if not valid_data:
            return result_batch
            
        # Process the valid data as a batch
        indices, values = zip(*valid_data)
        
        # Prepare the message for the API with the entire batch
        messages = [
            {"role": "system", "content": "You are a data cleaning assistant. Your task is to clean and structure data according to the instructions. Respond with valid JSON."},
            {"role": "user", "content": f"{prompt}\n\nData to clean:\n{json.dumps(list(values))}\n\nRespond with a valid JSON array containing one result object per input item, in the same order."}
        ]
        
        # Call the OpenAI API with retry logic
        cleaned_values = self._call_openai_with_retry(messages, schema, len(indices))
        
        # Store the results
        for i, idx in enumerate(indices):
            if i < len(cleaned_values):
                result = cleaned_values[i]
                
                # Validate schema if provided
                if schema:
                    try:
                        validate(instance=result, schema=schema)
                    except ValidationError as e:
                        result = {"error": f"Schema validation failed: {str(e)}", "data": result}
                
                result_batch.at[idx, f"cleaned_{column}"] = json.dumps(result)
            else:
                result_batch.at[idx, f"cleaned_{column}"] = json.dumps({"error": "No result returned from API"})
            
        return result_batch

    def _call_openai_with_retry(
        self, 
        messages: List[Dict[str, str]], 
        schema: Optional[Dict[str, Any]] = None,
        batch_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Call OpenAI API with retry logic.

        Args:
            messages: List of message dictionaries
            schema: JSON schema for validation (optional)
            batch_size: Number of items we expect in the response

        Returns:
            Cleaned data as a list of dictionaries
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
                    parsed_response = json.loads(response.choices[0].message.content)
                    
                    # Handle different response formats
                    if isinstance(parsed_response, list):
                        return parsed_response
                    elif isinstance(parsed_response, dict) and 'results' in parsed_response:
                        return parsed_response['results']
                    else:
                        # If we only have one item or the response is not properly formatted
                        if batch_size == 1:
                            return [parsed_response]
                        else:
                            # If this was supposed to be a batch but we got a single object
                            logger.warning("Expected batch results but got a single object. Attempting another retry.")
                            if attempt < self.max_retries - 1:
                                time.sleep(self.retry_delay)
                                continue
                            else:
                                # Return the single response duplicated as a last resort
                                return [parsed_response] * batch_size
                
                except json.JSONDecodeError:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Failed to parse JSON response. Retrying ({attempt + 1}/{self.max_retries})...")
                        time.sleep(self.retry_delay)
                    else:
                        return [{"error": "Failed to parse JSON response"}] * batch_size
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"API call failed: {str(e)}. Retrying ({attempt + 1}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    return [{"error": f"API call failed: {str(e)}"}] * batch_size
        
        # If we get here, all retries failed
        return [{"error": "Failed after maximum retries"}] * batch_size