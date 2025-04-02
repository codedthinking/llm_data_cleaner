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
        api_batch_size: int = 5,
        temperature: float = 0.0,
    ):
        """
        Initialize the DataCleaner.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Delay between retry attempts in seconds
            batch_size: Number of rows to process in a dataframe batch
            api_batch_size: Number of rows to process in a single API call
            temperature: Temperature setting for the OpenAI API
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.api_batch_size = api_batch_size
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
        
        # Split the batch into API batch chunks to send multiple rows per API call
        api_batches = [
            batch.iloc[i:i+self.api_batch_size] 
            for i in range(0, len(batch), self.api_batch_size)
        ]
        
        for api_batch in api_batches:
            # Prepare data for this API batch
            batch_data = []
            batch_indices = []
            
            for idx, row in api_batch.iterrows():
                original_value = row[column]
                if pd.isna(original_value):
                    result_batch.at[idx, f"cleaned_{column}"] = json.dumps({"error": "Original value is NaN"})
                else:
                    batch_data.append(str(original_value))
                    batch_indices.append(idx)
            
            # If we have data to process in this API batch
            if batch_data:
                # Prepare the message for the API with multiple items
                messages = [
                    {"role": "system", "content": "You are a data cleaning assistant. Your task is to clean and structure data according to the instructions. Respond with valid JSON for each item in the array."},
                    {"role": "user", "content": f"{prompt}\n\nData to clean (process each item separately):\n{json.dumps(batch_data)}\n\nRespond with a JSON array containing one cleaned result per input item, in the same order. Each result should be a JSON object."}
                ]
                
                # Call the OpenAI API with retry logic
                response_data = self._call_openai_with_retry(messages, schema, batch_size=len(batch_data))
                
                # Process the results
                if isinstance(response_data, list) and len(response_data) == len(batch_indices):
                    # Store each result with its corresponding index
                    for i, (idx, result) in enumerate(zip(batch_indices, response_data)):
                        # Validate against schema if provided
                        if schema and not isinstance(result, dict):
                            result = {"error": "Response is not a valid JSON object"}
                        elif schema:
                            try:
                                validate(instance=result, schema=schema)
                            except ValidationError as e:
                                result = {"error": f"Schema validation failed: {str(e)}", "data": result}
                        
                        result_batch.at[idx, f"cleaned_{column}"] = json.dumps(result)
                else:
                    # Handle error cases where we don't get the expected response format
                    error_msg = {"error": "Invalid API response format"}
                    for idx in batch_indices:
                        result_batch.at[idx, f"cleaned_{column}"] = json.dumps(error_msg)
            
        return result_batch

    def _call_openai_with_retry(
        self, 
        messages: List[Dict[str, str]], 
        schema: Optional[Dict[str, Any]] = None,
        batch_size: int = 1
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Call OpenAI API with retry logic.

        Args:
            messages: List of message dictionaries
            schema: JSON schema for validation (optional)
            batch_size: Number of items we expect in the response

        Returns:
            Cleaned data as a list of dictionaries (for batch) or a single dictionary
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
                    response_text = response.choices[0].message.content
                    cleaned_data = json.loads(response_text)
                    
                    # For batch responses, ensure we have a list of the right size
                    if batch_size > 1:
                        if isinstance(cleaned_data, list) and len(cleaned_data) == batch_size:
                            return cleaned_data
                        elif isinstance(cleaned_data, dict) and 'results' in cleaned_data and isinstance(cleaned_data['results'], list):
                            # Handle case where API returns {"results": [...]}
                            return cleaned_data['results']
                        else:
                            # Try to interpret as a list if it's not already
                            logger.warning(f"Expected list of {batch_size} items but got different format. Attempting to fix.")
                            if isinstance(cleaned_data, dict):
                                # Sometimes the API returns a dictionary with numbered keys
                                if all(str(i) in cleaned_data for i in range(batch_size)):
                                    return [cleaned_data[str(i)] for i in range(batch_size)]
                            
                            # If we can't interpret as a list, log error and retry
                            if attempt < self.max_retries - 1:
                                logger.warning(f"Received wrong response format. Retrying ({attempt + 1}/{self.max_retries})...")
                                time.sleep(self.retry_delay)
                                continue
                            else:
                                return [{"error": "Failed to get proper batch response"} for _ in range(batch_size)]
                    
                    return cleaned_data
                    
                except json.JSONDecodeError:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Failed to parse JSON response. Retrying ({attempt + 1}/{self.max_retries})...")
                        time.sleep(self.retry_delay)
                    else:
                        if batch_size > 1:
                            return [{"error": "Failed to parse JSON response"} for _ in range(batch_size)]
                        else:
                            return {"error": "Failed to parse JSON response"}
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"API call failed: {str(e)}. Retrying ({attempt + 1}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    if batch_size > 1:
                        return [{"error": f"API call failed: {str(e)}"} for _ in range(batch_size)]
                    else:
                        return {"error": f"API call failed: {str(e)}"}
        
        # If we get here, all retries failed
        if batch_size > 1:
            return [{"error": "Failed after maximum retries"} for _ in range(batch_size)]
        else:
            return {"error": "Failed after maximum retries"}