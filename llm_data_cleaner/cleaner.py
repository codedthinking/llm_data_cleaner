import os
import pandas as pd
from typing import Dict, Any, Type, List, Optional
from openai import OpenAI
from pydantic import BaseModel, create_model, ConfigDict
from llm_data_cleaner.utils import InstructionField, InstructionSchema
import time
from tqdm import tqdm
from .utils import jsonize


class DataCleaner:
    """
    Batch DataCleaner that uses OpenAI's responses.parse method with auto-generated prompts.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-2024-08-06",
        max_retries: int = 3,
        retry_delay: int = 5,
        batch_size: int = 10,
        system_prompt: str = None,
        temperature: float = 0.0,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.temperature = temperature

        # General system prompt format, set once for all tasks (you may tweak further)
        if system_prompt:
            # check that "{column_prompt}" is present in the prompt
            if "{column_prompt}" not in system_prompt:
                raise ValueError("System prompt must contain '{column_prompt}' placeholder to load column-specific instructions.")
            self.system_prompt_template = system_prompt
        else:        
            self.system_prompt_template = (
                "You are a data cleaning assistant. Your task is to clean and structure data according to the instructions. "
                "{column_prompt}."
                "The first element of each tuple is the row index, the second element is the value to be cleaned. "
                "Respond with a JSON list of objects, one per tuple, each including the input row's index and the extracted"
                "fields (index, year, university, etc.). If extraction fails, output null or set the fields to null, but always"
                    "preserve the input order and list length."
            )

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        instructions: InstructionSchema,
    ) -> pd.DataFrame:
        """
        Cleans each column specified in instructions, in batches.
        instructions: dict of {column: {"prompt": "...", "schema": pydantic.BaseModel}}
        Returns a DataFrame with cleaned columns appended with 'cleaned_{column}' prefix.
        """
        result_df = pd.DataFrame()
        for column, instruction in instructions.items():
            if column not in df.columns:
                print(f"Column '{column}' not found in DataFrame, skipping.")
                continue

            task_description = instruction["prompt"]
            pyd_model_batch: Type[BaseModel] = self._make_batch_model(instruction["schema"])
            if not issubclass(pyd_model_batch, BaseModel):
                raise ValueError(f"Invalid schema for column '{column}'. Must be a Pydantic model.")
            cleaned_batches = []

            print(f"Cleaning column '{column}' in batches...")
            for batch in tqdm(self._batch_dataframe(df[[column]], self.batch_size), desc=f"Cleaning {column}"):
                cleaned_batch = self._process_batch(batch, column, task_description, pyd_model_batch)
                cleaned_batches.append(cleaned_batch)

            # Concatenate all batch results and sort by index to original DataFrame
            full_cleaned = pd.concat(cleaned_batches)
            full_cleaned = full_cleaned.loc[df.index] # ensure original order
            result_df = pd.concat([result_df, full_cleaned], axis=1)
        return result_df

    def _batch_dataframe(self, df: pd.DataFrame, batch_size: int):
        for i in range(0, len(df), batch_size):
            yield df.iloc[i:i + batch_size]

    def _process_batch(
        self,
        batch: pd.DataFrame,
        column: str,
        task_description: str,
        pyd_model_batch: Type[BaseModel]
    ) -> pd.DataFrame:
        """
        Clean one batch of a column.
        """
        result_batch = batch.copy()
        tuples = list(zip(batch.index, batch[column].fillna("")))
        n_elements = len(tuples)
        system_msg = {
            "role": "system",
            "content": self.system_prompt_template.format(
                column_prompt=task_description, n_elements=n_elements
                )
            }
        user_msg = {
            "role": "user",
            "content": str(tuples)
        }
        cleaned = self._clean_batch([system_msg, user_msg], pyd_model_batch)
        if cleaned is None:
            print("No cleaned data returned, returning empty DataFrame.")
            return result_batch

        cleaned_list = cleaned.cleaned 

        for item in cleaned_list:
            if item is None:
                continue
            index = item.index
            for fname in item.model_fields:
                if fname != "index":
                    colname = f"cleaned_{fname}"
                    if colname not in result_batch.columns:
                        result_batch[colname] = None
                    value = getattr(item, fname, None)
                    # if value is a BaseModel, convert to JSON string
                    value = jsonize(value)
                    if index in result_batch.index:
                        result_batch.at[index, colname] = value

        return result_batch

    def _clean_batch(
        self,
        messages: list,
        pyd_model_batch: Type[BaseModel]
    ):
        for attempt in range(self.max_retries):
            try:
                resp = self.client.responses.parse(
                    model=self.model,
                    input=messages,
                    text_format=pyd_model_batch,
                    temperature=self.temperature,
                )
                return resp.output_parsed
            except Exception as e:
                print(f"Batch cleaning error: {e} (attempt {attempt+1}/{self.max_retries})")
                time.sleep(self.retry_delay)
        print("Failed batch cleaning after retries.")
        return None

    def _make_batch_model(self, model: Type[BaseModel], batch_name: str = None) -> Type[BaseModel]:
        if batch_name is None:
            batch_name = f"Batch{model.__name__}"
        return create_model(
            batch_name,
            cleaned=(List[Optional[model]], ...)
        )
    
def load_yaml_instructions(yaml_path:str = None) -> InstructionSchema:
    """
    Load models from YAML file.
    """
    import yaml

    type_names = dict(
        str=str,
        int=int,
        float=float,
        bool=bool,
        list=list,
        dict=dict,
        object=dict,
        array=list,
        integer=int,
        number=float,
        string=str,
        null=None,
        any=Any,)

    def parse_type(props):
        typ = props["type"]
        optional = props.get("optional", False)

        if isinstance(typ, str) and typ in type_names:
            typ = type_names[typ]

        if typ == list:
            items = props["items"]
            if isinstance(items, dict):
                typ = List[parse_type(items)]
            else:
                typ = List[parse_type({"type": items})]

        if typ == dict:
            # generate BaseModel on the fly
            properties = props.get("properties", {})
            annotations = {}
            for field, field_props in properties.items():
                annotations[field] = parse_type(field_props)

            # fix error:     raise PydanticUserError(pydantic.errors.PydanticUserError: A non-annotated attribute was detected: `year = <class 'str'>`. All model fields require a type annotation; if `year` is not meant to be a field, you may be able to resolve this error by annotating it as a `ClassVar` or updating `model_config['ignored_types']`.
            # to avoid this, we need to use create_model with annotations
            # and not BaseModel
            annotations = {k: (v, ...) for k, v in annotations.items()}
            annotations["__config__"] = ConfigDict(extra="forbid", json_schema_extra={"additionalProperties": False})

            # create a new model with the properties
            typ = create_model("DynamicModel", **annotations)

        if optional:
            typ = Optional[typ]

        return typ

    with open(yaml_path, "r") as f:
        schema = yaml.safe_load(f)

    instructions = {}

    for name, instruction in schema.items():
        prompt = instruction["prompt"]
        model_def = instruction.get("schema", {})

        # schema is a dictonary with its "type" key having value "object". we also need a "properties" key
        if not ("properties" in model_def and "type" in model_def and model_def["type"] == "object"):
            raise ValueError(f"Schema for {name} must be a dictionary with 'type' as 'object' and 'properties' key") 
        
        fields = model_def["properties"]
        annotations = {}
        defaults = {}
        for field, props in fields.items():
            t = parse_type(props)
            if props.get("optional", False):
                defaults[field] = None
                annotations[field] = (t, None)
            else:
                annotations[field] = (t, ...)
        instructions[name] = dict(prompt=prompt, schema_class=create_model(name, **annotations, __config__=ConfigDict(extra="forbid", json_schema_extra={"additionalProperties": False})))
    return InstructionSchema(root=instructions)
